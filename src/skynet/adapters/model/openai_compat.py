"""OpenAI-compatible chat completions client.

Ollama and OpenRouter both speak this dialect, so the wire logic lives here once
and each provider is a thin subclass that supplies a base URL, credentials and
its capability profile. Adding a third OpenAI-compatible provider is a subclass
plus a registry entry.

Translation between the harness's own message/tool types and the wire format
happens here and nowhere else, which is what keeps provider vocabulary out of
the core.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from skynet.domain.errors import ProviderError
from skynet.domain.ids import CallId
from skynet.domain.messages import Message, Role, ToolCall
from skynet.domain.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ToolChoice,
    Usage,
)
from skynet.domain.tools import ToolSpec
from skynet.ports.model import ProviderCapabilities

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Thinking models expose their reasoning under different names depending on the
# provider. None of them is the OpenAI spec, so all of them are guesses.
_REASONING_FIELDS = ("reasoning", "reasoning_content", "thinking")


def _reasoning_of(message: dict[str, Any]) -> str | None:
    for field in _REASONING_FIELDS:
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


class OpenAICompatProvider:
    """Base implementation of the ``ModelProvider`` port over HTTP."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        capabilities: ProviderCapabilities | None = None,
        timeout_s: float = 300.0,
        default_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._capabilities = capabilities or ProviderCapabilities(native_tools=True)
        self._headers = {"Content-Type": "application/json", **(default_headers or {})}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = client
        self._timeout = timeout_s

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self.build_payload(request)
        data = await self._post("/chat/completions", payload)
        return self.parse_response(data)

    # -- wire translation -----------------------------------------------------

    def build_payload(self, request: ModelRequest) -> dict[str, Any]:
        params = request.params
        payload: dict[str, Any] = {
            "model": params.model,
            "messages": [self._encode_message(m) for m in request.messages],
            "temperature": params.temperature,
            "stream": False,
        }
        if params.top_p is not None:
            payload["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            payload["max_tokens"] = params.max_output_tokens
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed

        if request.tools and self._capabilities.native_tools:
            payload["tools"] = [self._encode_tool(t) for t in request.tools]
            if request.tool_choice is not ToolChoice.AUTO:
                payload["tool_choice"] = str(request.tool_choice)

        if request.response_schema and self._capabilities.json_schema_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.response_schema},
            }

        payload.update(params.extra)
        return payload

    @staticmethod
    def _encode_tool(spec: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    @staticmethod
    def _encode_message(message: Message) -> dict[str, Any]:
        if message.role is Role.TOOL:
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }

        encoded: dict[str, Any] = {"role": str(message.role), "content": message.content}
        if message.tool_calls:
            encoded["tool_calls"] = [
                OpenAICompatProvider._encode_tool_call(call) for call in message.tool_calls
            ]
        return encoded

    @staticmethod
    def _encode_tool_call(call: ToolCall) -> dict[str, Any]:
        encoded: dict[str, Any] = {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
        if extra := call.provider_data.get("extra_content"):
            encoded["extra_content"] = extra
        return encoded

    def parse_response(self, data: dict[str, Any]) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"provider returned no choices: {data}")
        choice = choices[0]
        raw_message = choice.get("message") or {}

        tool_calls = tuple(
            self._decode_tool_call(index, item)
            for index, item in enumerate(raw_message.get("tool_calls") or [])
        )
        finish = _FINISH_REASONS.get(choice.get("finish_reason") or "stop", FinishReason.STOP)
        if tool_calls and finish is FinishReason.STOP:
            finish = FinishReason.TOOL_CALLS

        usage = data.get("usage") or {}
        return ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=raw_message.get("content") or "",
                reasoning=_reasoning_of(raw_message),
                tool_calls=tool_calls,
            ),
            finish_reason=finish,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            model=str(data.get("model") or ""),
            provider_response_id=data.get("id"),
        )

    @staticmethod
    def _decode_tool_call(index: int, item: dict[str, Any]) -> ToolCall:
        function = item.get("function") or {}
        raw_arguments = function.get("arguments")
        arguments: dict[str, Any]
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                parsed = json.loads(raw_arguments or "{}")
                arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                # Keep the raw text: the tool's own validation will explain the
                # problem to the model far better than a crash here would.
                arguments = {}
        # Round-trip anything the provider attached to the call. Gemini 3
        # rejects a conversation whose function calls come back without their
        # `thought_signature`, and the only safe policy is to return whatever we
        # were given, unread.
        extra = item.get("extra_content")
        return ToolCall(
            id=CallId(str(item.get("id") or f"call-{index}")),
            name=str(function.get("name") or ""),
            arguments=arguments,
            raw=raw_arguments if isinstance(raw_arguments, str) else None,
            provider_data={"extra_content": extra} if isinstance(extra, dict) else {},
        )

    # -- transport ------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload, headers=self._headers)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"model request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"model request failed: {exc}", retryable=True) from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"model provider returned {response.status_code}: {response.text[:500]}",
                retryable=response.status_code in _RETRYABLE_STATUS,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"model provider returned non-JSON: {exc}") from exc


__all__ = ["OpenAICompatProvider"]
