"""A deterministic embedder that needs nothing to be running.

This is what keeps the full test suite -- and ``docker compose up`` on a laptop
with no models pulled -- working without a network. It is signed feature hashing
over word tokens *and* character trigrams, projected into the same dimension a
real embedder uses, so the vector column and every query path get exercised for
real rather than skipped.

Two details are load-bearing.

**``blake2b``, not Python's ``hash()``.** ``hash()`` is salted per process by
``PYTHONHASHSEED``, so hash-based vectors would differ between test runs, and
between the API and worker containers reading the same table. That single choice
is the difference between deterministic and haunted.

**Character trigrams as well as words.** A token-only hash scores "pangolin"
against "pangolins" at exactly zero, which is the one behaviour a hybrid
retrieval test most needs to exercise. Trigrams buy partial credit for
morphological variants.

It is not semantic: "car" will not find "vehicle". It is a stand-in that is
honest about being one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from hashlib import blake2b

DEFAULT_DIMENSIONS = 768
_WORD = re.compile(r"[a-z0-9]+")


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Index and sign for one feature. Signed hashing cancels collisions rather
    than letting them all add up, which is what keeps unrelated text apart."""
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, 1.0 if value >> 63 & 1 else -1.0


def _features(text: str) -> list[str]:
    lowered = text.lower()
    words = _WORD.findall(lowered)
    normalised = f" {' '.join(words)} "
    trigrams = [normalised[i : i + 3] for i in range(max(0, len(normalised) - 2))]
    return [f"w:{word}" for word in words] + [f"g:{gram}" for gram in trigrams]


class HashingEmbedder:
    """Offline, deterministic, and identical in every process."""

    def __init__(self, *, dimensions: int = DEFAULT_DIMENSIONS, model: str = "hashing") -> None:
        self._dimensions = dimensions
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    def _vector(self, text: str) -> tuple[float, ...]:
        weights = [0.0] * self._dimensions
        for feature in _features(text):
            index, sign = _bucket(feature, self._dimensions)
            weights[index] += sign
        norm = math.sqrt(sum(weight * weight for weight in weights))
        if norm == 0.0:
            return tuple(weights)
        return tuple(weight / norm for weight in weights)


class NullEmbedder:
    """No vectors at all -- lexical search only.

    Not a degenerate case: it is the correct choice when the corpus is small and
    keyword search is genuinely enough, and it is what the store falls back to
    when embedding is switched off.
    """

    def __init__(self, *, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return "null"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(() for _ in texts)


__all__ = ["DEFAULT_DIMENSIONS", "HashingEmbedder", "NullEmbedder"]
