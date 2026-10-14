"""Memory port.

Five methods, and the omissions are the design.

``list`` is absent: "the recent episodic memories" is ``search(text="")`` with a
kind filter. A second retrieval path is a second thing to keep consistent with
the first.

``update`` is absent: bi-temporality means a fact is never mutated in place.
Offering it would let an adapter quietly destroy history and silently break
``as_of`` reads.

``expire`` is absent from *this* protocol: correctness already comes from the
read-time ``as_of`` filter, so reclaiming rows is maintenance, not retrieval. It
lives on ``MemoryMaintenance`` below, which a CLI may use and an executor may
not -- interface segregation, so a store that cannot vacuum is still a store.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from skynet.domain.ids import MemoryId
from skynet.domain.memory import MemoryHit, MemoryQuery, MemoryRecord, MemoryScope


@runtime_checkable
class MemoryStore(Protocol):
    async def put(self, record: MemoryRecord) -> MemoryId:
        """Write a record. Idempotent by id, so a consolidation retry is free."""
        ...

    async def get(self, scope: MemoryScope, memory_id: MemoryId) -> MemoryRecord | None:
        """Fetch by id, invalidated or not -- history stays readable."""
        ...

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        """Candidates across every scope in the query, in one round trip.

        Returns raw ``relevance`` and leaves fusion to
        ``application.memory_ranking``: a store reports similarity, the
        application decides what is worth remembering.
        """
        ...

    async def invalidate(
        self,
        scope: MemoryScope,
        memory_id: MemoryId,
        *,
        at: datetime,
        superseded_by: MemoryId | None = None,
    ) -> bool:
        """Stamp ``valid_to``. The record survives; only the belief ends."""
        ...

    async def delete(self, scope: MemoryScope, memory_id: MemoryId) -> bool:
        """Erasure, for compliance. Deliberately not part of an agent's vocabulary."""
        ...


@runtime_checkable
class MemoryMaintenance(Protocol):
    """Optional housekeeping. Duck-typed on with ``isinstance`` at the edges."""

    async def purge_expired(self, *, before: datetime) -> int:
        """Physically remove rows already dead by ``before``. Returns the count."""
        ...


__all__ = ["MemoryMaintenance", "MemoryStore"]
