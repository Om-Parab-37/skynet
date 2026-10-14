"""HTTP embedders.

The wire logic lives here once; each provider is a thin subclass supplying a URL
shape, a payload and a way to read vectors back out -- the same arrangement the
model adapters use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from skynet.domain.errors import ProviderError

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpEmbedder:
    """Base ``Embedder`` over JSON/HTTP."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key
        self._timeout = timeout_s
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        payload = await self._post(self._url(), self._payload(list(texts)))
        vectors = self._vectors(payload)
        if len(vectors) != len(texts):
            raise ProviderError(f"embedder returned {len(vectors)} vectors for {len(texts)} inputs")
        for vector in vectors:
            if len(vector) != self._dimensions:
                # Silence here would present later as total recall loss with
                # nothing in any log to explain it.
                raise ProviderError(
                    f"embedder '{self._model}' returned {len(vector)} dimensions, "
                    f"expected {self._dimensions}"
                )
        return vectors

    # -- to be supplied by subclasses -----------------------------------------

    def _url(self) -> str:
        raise NotImplementedError

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    def _vectors(self, payload: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
        raise NotImplementedError

    # -- transport -------------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload, headers=self._headers)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"embedding request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"embedding request failed: {exc}", retryable=True) from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"embedding provider returned {response.status_code}: {response.text[:300]}",
                retryable=response.status_code in _RETRYABLE_STATUS,
            )
        return response.json()


class OllamaEmbedder(HttpEmbedder):
    """Ollama's *native* embedding endpoint.

    ``/api/embed``, not the OpenAI-compatible ``/v1`` surface -- and the base URL
    commonly configured for chat ends in ``/v1``, which would 404 here. Stripping
    it means one ``base_url`` serves both.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        **kwargs,
    ) -> None:
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[: -len("/v1")]
        super().__init__(base_url=cleaned, model=model, dimensions=dimensions, **kwargs)

    def _url(self) -> str:
        return f"{self._base_url}/api/embed"

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {"model": self._model, "input": texts}

    def _vectors(self, payload: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(x) for x in row) for row in payload.get("embeddings", []))


class OpenAICompatEmbedder(HttpEmbedder):
    """The ``/embeddings`` dialect, which most hosted providers speak."""

    def _url(self) -> str:
        return f"{self._base_url}/embeddings"

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {"model": self._model, "input": texts, "dimensions": self._dimensions}

    def _vectors(self, payload: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
        rows = sorted(payload.get("data", []), key=lambda row: row.get("index", 0))
        return tuple(tuple(float(x) for x in row.get("embedding", [])) for row in rows)


class GeminiEmbedder(OpenAICompatEmbedder):
    """Gemini embeddings, truncated to the configured width.

    Its native width is 3072, above pgvector's 2000-dimension index ceiling. The
    ``dimensions`` parameter is Matryoshka truncation, which is what makes it and
    a 768-dimension local model interchangeable without touching the schema.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            **kwargs,
        )


__all__ = ["GeminiEmbedder", "HttpEmbedder", "OllamaEmbedder", "OpenAICompatEmbedder"]
