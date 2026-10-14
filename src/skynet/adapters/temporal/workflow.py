"""The agent workflow.

Note what is *not* here: no agent logic. The workflow constructs the same
``AgentPolicy`` and the same ``AgentLoop`` the in-process runner uses, hands
them a Temporal-flavoured effect executor, and folds signals in with the same
``apply`` function. Temporal supplies durability; it does not define the agent.

The outer loop is the resume protocol, identical in shape to the in-process
runner's: run until suspended, wait for an external event, fold it in, continue.
"""

from __future__ import annotations

from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from skynet.adapters.temporal.effects import WorkflowEffectExecutor
    from skynet.adapters.temporal.registry import workflow_agent_catalog, workflow_tool_catalog
    from skynet.application.context import DefaultContextBuilder
    from skynet.application.loop import AgentLoop
    from skynet.application.policy import AgentPolicy
    from skynet.domain.agent import FinishMode
    from skynet.domain.errors import ErrorInfo
    from skynet.domain.events import (
        ApprovalResolved,
        RunCancelled,
        RunEvent,
        RunPaused,
        RunResumed,
        RunStarted,
    )
    from skynet.domain.run import AgentRun, AgentStatus, apply, new_run
    from skynet.domain.task import AgentResult
    from skynet.ports.agents import RunRequest

DEFAULT_TASK_QUEUE = "skynet-agents"


@workflow.defn(name="AgentWorkflow")
class AgentWorkflow:
    def __init__(self) -> None:
        self._pending: list[RunEvent] = []
        self._run: AgentRun | None = None
        self.catalog = workflow_tool_catalog()
        self.catalog_agents = workflow_agent_catalog()
        self.task_queue = DEFAULT_TASK_QUEUE

    @workflow.run
    async def run(self, request: RunRequest) -> AgentResult:
        self.task_queue = workflow.info().task_queue

        run = new_run(
            spec=request.spec,
            task=request.task,
            context=request.context,
            finish_mode=FinishMode.NO_TOOL_CALLS,
        )
        effects = WorkflowEffectExecutor(self)
        started = RunStarted(
            run_id=run.run_id,
            seq=1,
            at=workflow.now(),
            spec=request.spec,
            task=request.task,
            context=request.context,
        )
        # Publish before applying: RunStarted carries the spec, task and
        # context, which is what lets an observer rebuild the run from the log
        # alone after a restart.
        await effects.publish(started)
        run = apply(run, started)

        loop = AgentLoop(
            policy=AgentPolicy(self.catalog, DefaultContextBuilder()),
            effects=effects,
        )

        while True:
            run = await loop.run(run)
            self._run = run
            if run.status.is_terminal:
                break

            # Suspended: park until a signal arrives, then fold it in and go on.
            await workflow.wait_condition(lambda: bool(self._pending))
            for event in self._drain():
                run = apply(run, event.model_copy(update={"seq": run.seq + 1}))
            self._run = run

        if run.result is not None:
            return run.result
        error = run.last_error or ErrorInfo.of("unknown", f"run ended as {run.status}")
        raise ApplicationError(error.message, error.code, non_retryable=True)

    # -- signals: the durable equivalent of the in-process suspension queue ----

    @workflow.signal
    def approve(self, call_id: str, granted: bool, actor: str = "user") -> None:
        self._enqueue(
            ApprovalResolved(run_id=self._run_id(), call_id=call_id, granted=granted, actor=actor)
        )

    @workflow.signal
    def pause(self, reason: str = "") -> None:
        self._enqueue(RunPaused(run_id=self._run_id(), reason=reason))

    @workflow.signal
    def resume(self) -> None:
        self._enqueue(RunResumed(run_id=self._run_id()))

    @workflow.signal
    def cancel(self, reason: str = "") -> None:
        self._enqueue(RunCancelled(run_id=self._run_id(), reason=reason))

    # -- queries --------------------------------------------------------------

    @workflow.query
    def status(self) -> str:
        return str(self._run.status) if self._run else str(AgentStatus.CREATED)

    @workflow.query
    def iteration(self) -> int:
        return self._run.iteration if self._run else 0

    @workflow.query
    def progress(self) -> dict:
        if self._run is None:
            return {"status": str(AgentStatus.CREATED)}
        return {
            "status": str(self._run.status),
            "iteration": self._run.iteration,
            "tool_calls": self._run.tool_call_count,
            "tokens": self._run.usage.total,
            "children": len(self._run.child_runs),
        }

    # -- helpers --------------------------------------------------------------

    def _enqueue(self, event: RunEvent) -> None:
        self._pending.append(event)

    def _drain(self) -> list[RunEvent]:
        drained, self._pending = self._pending, []
        return drained

    def _run_id(self) -> str:
        return str(self._run.run_id) if self._run else workflow.info().workflow_id


__all__ = ["DEFAULT_TASK_QUEUE", "AgentWorkflow"]
