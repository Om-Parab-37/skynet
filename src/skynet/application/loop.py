"""The agent execution loop.

This is the entire driver. It alternates a pure decision, an effectful
execution, and a pure fold -- and it does nothing else. There is no branching on
infrastructure and no knowledge of any specific capability, model or backend.

Because the only ``await`` is the ``EffectExecutor`` call, this exact class runs
unchanged inside a Temporal workflow: the workflow simply supplies an executor
whose effects are activities and child workflows.
"""

from __future__ import annotations

from skynet.application.policy import AgentPolicy
from skynet.domain.errors import LoopStalled
from skynet.domain.run import AgentRun, apply
from skynet.ports.effects import EffectExecutor


class AgentLoop:
    def __init__(
        self,
        policy: AgentPolicy,
        effects: EffectExecutor,
        *,
        max_decisions: int = 500,
    ) -> None:
        self._policy = policy
        self._effects = effects
        self._max_decisions = max_decisions

    async def run(self, run: AgentRun) -> AgentRun:
        """Advance the run until it is terminal or suspended.

        Suspension (awaiting approval, paused) returns normally rather than
        blocking, so the caller decides how to wait -- an ``asyncio.Event``
        in-process, a signal under Temporal.
        """
        decisions = 0
        while not (run.status.is_terminal or run.status.is_suspended):
            if decisions >= self._max_decisions:
                raise LoopStalled(f"run {run.run_id} exceeded {self._max_decisions} decisions")
            decisions += 1

            decision = self._policy.decide(run)
            events = await self._effects.execute(run, decision)
            if not events:
                raise LoopStalled(
                    f"decision {type(decision).__name__} produced no events; "
                    "the run cannot make progress"
                )
            for event in events:
                run = apply(run, event)
        return run


__all__ = ["AgentLoop"]
