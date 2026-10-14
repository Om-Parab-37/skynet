"""Tool calling for models that lack it natively.

Rather than making each adapter cope, one decorator wraps any ``ModelProvider``
and simulates tool calling by describing the tools in the prompt and parsing a
fenced JSON block back out. Selection happens once, by capability negotiation,
so the core is unaware of which mode is in play.

A parse failure is surfaced as ``FinishReason.PARSE_ERROR`` rather than retried
inside the adapter: that keeps the failure in the event log, in Temporal
history, and inside the run's budget.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from skynet.domain.ids import CallId
from skynet.domain.messages import Message, Role, ToolCall, system
from skynet.domain.models import FinishReason, ModelRequest, ModelResponse
from skynet.domain.tools import ToolSpec
from skynet.ports.clock import IdGenerator
from skynet.ports.model import ModelProvider, ProviderCapabilities

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL)


class ParsedCalls(BaseModel):
    model_config = ConfigDict(frozen=True)

    calls: tuple[ToolCall, ...] = ()
    text: str = ""
    error: str | None = None


@runtime_checkable
class ToolCallCodec(Protocol):
    """Pure translation between prose and tool calls. Swappable per model family."""

    def augment(self, request: ModelRequest) -> ModelRequest: ...

    def parse(self, content: str, ids: IdGenerator) -> ParsedCalls: ...

    def format_reminder(self) -> str: ...


class JsonBlockCodec:
    """One action per turn, expressed as a single fenced JSON object.

    Deliberately simple: 7B-class models handle exactly one well-signposted
    format far better than they handle nested or repeated structures.
    """

    INSTRUCTIONS = """You have access to the following tools:

{tool_docs}

To use a tool, reply with EXACTLY ONE fenced JSON block and nothing else:

```json
{{"thought": "<one short sentence>", "tool": "<tool name>", "arguments": {{...}}}}
```

Rules:
- Emit exactly one JSON block per reply.
- Use only the tool names listed above; never invent one.
- When you have the final answer, call the `final_answer` tool.
"""

    REMINDER = (
        "Your previous reply was not a valid tool call. Reply with exactly one "
        'fenced JSON block of the form {"thought": "...", "tool": "...", '
        '"arguments": {...}} and nothing else.'
    )

    def augment(self, request: ModelRequest) -> ModelRequest:
        if not request.tools:
            return request
        instructions = self.INSTRUCTIONS.format(tool_docs=self._describe(request.tools))
        return request.model_copy(
            update={
                "messages": (*request.messages, system(instructions)),
                "tools": (),
            }
        )

    def parse(self, content: str, ids: IdGenerator) -> ParsedCalls:
        payload, remainder = self._extract(content)
        if payload is None:
            return ParsedCalls(text=content, error="no JSON object found in the reply")

        name = payload.get("tool") or payload.get("name")
        if not name:
            return ParsedCalls(text=content, error="JSON block has no 'tool' field")

        arguments = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(arguments, dict):
            return ParsedCalls(text=content, error="'arguments' must be an object")

        thought = str(payload.get("thought") or "")
        return ParsedCalls(
            calls=(
                ToolCall(
                    id=CallId(ids.new_id("call")),
                    name=str(name),
                    arguments=arguments,
                    raw=content,
                ),
            ),
            text=thought or remainder,
        )

    def format_reminder(self) -> str:
        return self.REMINDER

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _describe(tools: tuple[ToolSpec, ...]) -> str:
        lines = []
        for spec in tools:
            properties = spec.parameters.get("properties", {})
            required = set(spec.parameters.get("required", []))
            args = ", ".join(
                f"{key}: {value.get('type', 'string')}" + ("" if key in required else " (optional)")
                for key, value in properties.items()
            )
            lines.append(f"- {spec.name}({args})\n    {spec.description}")
        return "\n".join(lines)

    @staticmethod
    def _extract(content: str) -> tuple[dict[str, Any] | None, str]:
        for pattern in (_FENCED, _BARE):
            for match in pattern.finditer(content):
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    remainder = (content[: match.start()] + content[match.end() :]).strip()
                    return parsed, remainder
        return None, content


class EmulatedToolCallingProvider:
    """Wraps a provider so that it appears to support native tool calling."""

    def __init__(self, inner: ModelProvider, codec: ToolCallCodec, ids: IdGenerator) -> None:
        self._inner = inner
        self._codec = codec
        self._ids = ids

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._inner.capabilities.model_copy(
            update={
                "native_tools": True,
                "emulated_tools": True,
                "parallel_tool_calls": False,
            }
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not request.tools:
            return await self._inner.generate(request)

        response = await self._inner.generate(self._codec.augment(request))
        parsed = self._codec.parse(response.message.content, self._ids)

        if parsed.error:
            return response.model_copy(
                update={
                    "finish_reason": FinishReason.PARSE_ERROR,
                    "parse_error": parsed.error,
                }
            )

        return response.model_copy(
            update={
                "message": Message(
                    role=Role.ASSISTANT, content=parsed.text, tool_calls=parsed.calls
                ),
                "finish_reason": (FinishReason.TOOL_CALLS if parsed.calls else FinishReason.STOP),
            }
        )


def wrap_for_tool_calling(
    provider: ModelProvider,
    ids: IdGenerator,
    *,
    codec: ToolCallCodec | None = None,
    force_emulation: bool = False,
) -> ModelProvider:
    """Use native tool calling when available, emulate it otherwise."""
    if provider.capabilities.native_tools and not force_emulation:
        return provider
    return EmulatedToolCallingProvider(provider, codec or JsonBlockCodec(), ids)


__all__ = [
    "EmulatedToolCallingProvider",
    "JsonBlockCodec",
    "ParsedCalls",
    "ToolCallCodec",
    "wrap_for_tool_calling",
]
