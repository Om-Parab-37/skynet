"""Ollama provider.

Ollama exposes an OpenAI-compatible endpoint at ``/v1`` which returns proper
``tool_calls``, so it needs no bespoke transport -- only its own base URL and
capability profile.
"""

from __future__ import annotations

from skynet.adapters.model.openai_compat import OpenAICompatProvider
from skynet.ports.model import ProviderCapabilities

DEFAULT_BASE_URL = "http://localhost:11434/v1"

OLLAMA_CAPABILITIES = ProviderCapabilities(
    native_tools=True,
    parallel_tool_calls=True,
    streaming=True,
    json_schema_output=True,
)


class OllamaProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        capabilities: ProviderCapabilities | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=None,
            capabilities=capabilities or OLLAMA_CAPABILITIES,
            **kwargs,
        )


__all__ = ["DEFAULT_BASE_URL", "OLLAMA_CAPABILITIES", "OllamaProvider"]
