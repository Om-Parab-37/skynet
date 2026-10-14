"""The effect port -- the single seam between pure logic and the world.

Every non-deterministic operation an agent performs passes through here. That is
the whole reason the same ``AgentLoop`` can run in a plain event loop and inside
a Temporal workflow: only this object differs.

Implementations must stamp ``seq`` and ``at`` on the events they return, since
both come from the host rather than from the domain.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from skynet.domain.decisions import Decision
from skynet.domain.events import RunEvent
from skynet.domain.run import AgentRun


@runtime_checkable
class EffectExecutor(Protocol):
    async def execute(self, run: AgentRun, decision: Decision) -> Sequence[RunEvent]: ...


__all__ = ["EffectExecutor"]
