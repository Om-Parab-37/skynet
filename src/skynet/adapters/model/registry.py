"""Provider selection.

The extension point for model support: a new provider is a class plus one entry
here. Nothing in the core changes, and nothing here knows what an agent is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from skynet.adapters.model.gemini import GeminiProvider
from skynet.adapters.model.ollama import OllamaProvider
from skynet.adapters.model.openai_compat import OpenAICompatProvider
from skynet.adapters.model.openrouter import OpenRouterProvider
from skynet.domain.errors import ConfigurationError
from skynet.ports.model import ModelProvider

ProviderFactory = Callable[..., ModelProvider]

PROVIDERS: Mapping[str, ProviderFactory] = {
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
    "gemini": GeminiProvider,
    "openai_compat": OpenAICompatProvider,
}


def create_provider(name: str, **kwargs) -> ModelProvider:
    try:
        factory = PROVIDERS[name]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown model provider '{name}'; available: {', '.join(sorted(PROVIDERS))}"
        ) from exc
    return factory(**kwargs)


__all__ = ["PROVIDERS", "ProviderFactory", "create_provider"]
