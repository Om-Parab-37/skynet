"""Google Gemini provider.

Gemini exposes an OpenAI-compatible surface, so this is the whole adapter: a
base URL, a key, and a capability profile. That it takes twenty lines is the
point -- adding a model provider must not require touching the agent runtime.
"""

from __future__ import annotations

from skynet.adapters.model.openai_compat import OpenAICompatProvider
from skynet.ports.model import ProviderCapabilities

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

GEMINI_CAPABILITIES = ProviderCapabilities(
    native_tools=True,
    parallel_tool_calls=True,
    streaming=True,
    json_schema_output=True,
)


class GeminiProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        capabilities: ProviderCapabilities | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            capabilities=capabilities or GEMINI_CAPABILITIES,
            **kwargs,
        )


__all__ = ["DEFAULT_BASE_URL", "GEMINI_CAPABILITIES", "GeminiProvider"]
