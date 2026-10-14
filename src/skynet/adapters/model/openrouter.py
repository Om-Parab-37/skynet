"""OpenRouter provider.

Hosted, multi-vendor and OpenAI-compatible. Tool-calling support varies by the
model selected, so ``native_tools`` is configurable rather than assumed; when it
is off, the runtime wraps the provider in emulated tool calling automatically.
"""

from __future__ import annotations

from skynet.adapters.model.openai_compat import OpenAICompatProvider
from skynet.ports.model import ProviderCapabilities

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_CAPABILITIES = ProviderCapabilities(
    native_tools=True,
    parallel_tool_calls=True,
    streaming=True,
    json_schema_output=True,
)


class OpenRouterProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        capabilities: ProviderCapabilities | None = None,
        referer: str = "https://github.com/skynet",
        title: str = "Skynet Agent Harness",
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            capabilities=capabilities or OPENROUTER_CAPABILITIES,
            default_headers={"HTTP-Referer": referer, "X-Title": title},
            **kwargs,
        )


__all__ = ["DEFAULT_BASE_URL", "OPENROUTER_CAPABILITIES", "OpenRouterProvider"]
