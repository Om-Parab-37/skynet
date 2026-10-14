"""Embedding port.

Separate from ``ModelProvider`` on purpose: the two have different failure
policies. A chat failure is an observation the agent reasons about; an embedding
failure is invisible to the agent and must degrade to "this record is lexically
searchable but not semantically", never to an exception that fails a run.

``dimensions`` is on the port because a store's schema depends on it -- a
mismatch between what was indexed and what is being queried presents as 100%
recall loss with no error anywhere, so it must be assertable at boot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """One vector per input, in input order. Batched: embedding is per-round-trip."""
        ...


__all__ = ["Embedder"]
