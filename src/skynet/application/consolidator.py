"""Running one consolidation pass.

The pure decisions live in ``memory_consolidation``; this composes them with the
ports that do the work. It exists so the in-process executor and the durable
activity run *identical* logic -- if they diverged, the same run would remember
different things depending on which engine executed it, which is the kind of bug
that only shows up in production and cannot be reproduced in a test.

It is a collaborator, not a port, for the same reason ``WorkspaceArchiver`` is:
it introduces no new I/O boundary. Every effect it has already belongs to
``ModelProvider`` or ``MemoryStore``, so it stays traced by the existing spans
and swappable by the existing configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from skynet.application.memory_consolidation import (
    consolidation_request,
    parse_operations,
    plan_writes,
)
from skynet.application.memory_ranking import candidate_limit, rank
from skynet.domain.decisions import ConsolidateMemory
from skynet.domain.errors import ErrorInfo
from skynet.domain.ids import MemoryId, RunId
from skynet.domain.memory import MemoryOperation, MemoryRecord
from skynet.domain.messages import Message
from skynet.ports.clock import Clock
from skynet.ports.memory import MemoryStore
from skynet.ports.model import ModelProvider


class ConsolidationOutcome(BaseModel):
    """What one pass managed to do. Errors are data, never exceptions."""

    model_config = ConfigDict(frozen=True)

    operations: tuple[MemoryOperation, ...] = ()
    written: tuple[MemoryId, ...] = ()
    invalidated: tuple[MemoryId, ...] = ()
    rejected: int = 0
    error: ErrorInfo | None = None


class Consolidator:
    def __init__(self, memory: MemoryStore, model: ModelProvider, clock: Clock) -> None:
        self._memory = memory
        self._model = model
        self._clock = clock

    async def run(
        self,
        decision: ConsolidateMemory,
        *,
        run_id: RunId,
        objective: str = "",
        messages: Sequence[Message] = (),
        default_ttl_hours: float | None = None,
    ) -> ConsolidationOutcome:
        """Retrieve, ask, apply -- one step with one failure mode.

        Kept whole rather than composed from a model call and separate writes
        because a partial failure between "the model says this is contradicted"
        and "the store wrote the replacement" would need compensation logic
        nothing else in this runtime has. Retry safety comes instead from the
        decision's ``idempotency_key``: every written id derives from it, so a
        replay writes the same rows rather than a second copy.
        """
        now = self._clock.now()
        try:
            candidates = await self._candidates(decision, now)
            response = await self._model.generate(
                consolidation_request(
                    decision.mode,
                    objective=objective,
                    messages=messages,
                    candidates=candidates,
                    params=decision.params,
                )
            )
            operations = parse_operations(
                response.message.text, known_ids=[record.id for record in candidates]
            )
            plan = plan_writes(
                operations,
                key=decision.idempotency_key,
                scope=decision.scope,
                at=now,
                candidates=candidates,
                source_run_id=run_id,
                default_ttl_hours=default_ttl_hours,
            )
            # Invalidate first. An update is invalidate-then-add, and the other
            # order leaves both the old and the new belief live in between --
            # briefly, but long enough for a concurrent read to see both.
            invalidated: list[MemoryId] = []
            for target, superseded_by in plan.invalidations:
                if await self._memory.invalidate(
                    decision.scope, target, at=now, superseded_by=superseded_by
                ):
                    invalidated.append(target)
            written = [await self._memory.put(record) for record in plan.writes]
        except Exception as exc:  # noqa: BLE001 - memory must never fail a run
            return ConsolidationOutcome(error=ErrorInfo.from_exc(exc))

        return ConsolidationOutcome(
            operations=operations,
            written=tuple(written),
            invalidated=tuple(invalidated),
            rejected=plan.rejected,
        )

    async def _candidates(
        self, decision: ConsolidateMemory, now: datetime
    ) -> tuple[MemoryRecord, ...]:
        """What the model reconciles against. Empty when none was requested."""
        query = decision.retrieval
        if query is None:
            return ()
        hits = await self._memory.search(
            query.model_copy(update={"limit": candidate_limit(query.limit), "as_of": now})
        )
        return tuple(hit.record for hit in rank(hits, at=now, limit=query.limit))


__all__ = ["ConsolidationOutcome", "Consolidator"]
