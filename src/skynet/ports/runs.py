"""Run persistence port.

Event-sourced: ``append`` is the write path and ``load`` rebuilds state by
replaying. A snapshot is an optimisation, never the source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from skynet.domain.events import RunEvent
from skynet.domain.ids import RunId
from skynet.domain.run import AgentRun


@runtime_checkable
class RunStore(Protocol):
    async def append(self, run_id: RunId, events: Sequence[RunEvent]) -> None: ...

    async def events(self, run_id: RunId) -> tuple[RunEvent, ...]: ...

    async def snapshot(self, run: AgentRun) -> None: ...

    async def load(self, run_id: RunId) -> AgentRun | None: ...

    async def list_runs(self, limit: int = 50) -> tuple[AgentRun, ...]: ...


__all__ = ["RunStore"]
