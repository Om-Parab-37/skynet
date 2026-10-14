"""Embedder selection. A new embedder is a class plus one entry here."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from skynet.adapters.embedding.hashing import HashingEmbedder, NullEmbedder
from skynet.adapters.embedding.http import (
    GeminiEmbedder,
    OllamaEmbedder,
    OpenAICompatEmbedder,
)
from skynet.domain.errors import ConfigurationError
from skynet.ports.embedding import Embedder

EmbedderFactory = Callable[..., Embedder]

EMBEDDERS: Mapping[str, EmbedderFactory] = {
    "hashing": HashingEmbedder,
    "null": NullEmbedder,
    "ollama": OllamaEmbedder,
    "gemini": GeminiEmbedder,
    "openai_compat": OpenAICompatEmbedder,
}


def create_embedder(name: str, **kwargs) -> Embedder:
    try:
        factory = EMBEDDERS[name]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown embedder '{name}'; available: {', '.join(sorted(EMBEDDERS))}"
        ) from exc
    return factory(**kwargs)


__all__ = ["EMBEDDERS", "EmbedderFactory", "create_embedder"]
