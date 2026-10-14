"""Temporal-backed ``AgentRunner``.

Implements the same port as the in-process runner, so the API and CLI cannot
tell which engine is in use. Switching between them is one environment variable.

Signals carry the same domain events the in-process runner delivers through an
``asyncio.Event`` -- same protocol, durable transport.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio.client import Client, WorkflowHandle
from temporalio.common import RetryPolicy

from skynet.adapters.temporal.workflow import AgentWorkflow
from skynet.domain.errors import ErrorInfo
from skynet.domain.events import RunCancelled, RunCompleted, RunFailed, RunStarted
from skynet.domain.ids import CallId, RunId
from skynet.domain.models import Usage
from skynet.domain.run import AgentRun, AgentStatus, apply, new_run
from skynet.ports.agents import RunRequest

# Temporal's lifecycle vocabulary mapped onto the domain's, at the boundary, so
# the domain never learns Temporal's status names.
#
# Keyed on the enum *name*: WorkflowExecutionStatus is an IntEnum, so str() of
# it yields the number, not the label.
_STATUS_MAP = {
    "RUNNING": AgentStatus.RUNNING,
    "COMPLETED": AgentStatus.COMPLETED,
    "FAILED": AgentStatus.FAILED,
    "CANCELED": AgentStatus.CANCELLED,
    "TERMINATED": AgentStatus.CANCELLED,
    "TIMED_OUT": AgentStatus.FAILED,
    "CONTINUED_AS_NEW": AgentStatus.RUNNING,
}


def workflow_status(raw: object) -> AgentStatus:
    """Translate a Temporal execution status into the domain's status."""
    name = getattr(raw, "name", None) or str(raw)
    return _STATUS_MAP.get(name, AgentStatus.RUNNING)


class TemporalRunHandle:
    def __init__(self, handle: WorkflowHandle, request: RunRequest) -> None:
        self._handle = handle
        self._request = request

    @property
    def run_id(self) -> RunId:
        return self._request.context.run_id

    async def result(self) -> AgentRun:
        run = _initial(self._request)
        try:
            result = await self._handle.result()
        except Exception as exc:  # noqa: BLE001 - surface as a failed run, not a crash
            run = await self._merge_progress(run)
            return apply(
                run, RunFailed(run_id=run.run_id, seq=run.seq + 1, error=ErrorInfo.from_exc(exc))
            )
        run = await self._merge_progress(run)
        return apply(run, RunCompleted(run_id=run.run_id, seq=run.seq + 1, result=result))

    async def snapshot(self) -> AgentRun:
        """Read the run's current state from Temporal.

        ``describe`` gives the authoritative lifecycle status and a query gives
        the agent's own counters. Neither waits for completion, which is what
        makes this safe to call from a status endpoint.
        """
        run = await self._merge_progress(_initial(self._request))
        try:
            description = await self._handle.describe()
        except Exception:  # noqa: BLE001
            return run

        status = workflow_status(description.status)
        if status is AgentStatus.COMPLETED:
            try:
                result = await self._handle.result()
            except Exception as exc:  # noqa: BLE001
                return apply(
                    run,
                    RunFailed(run_id=run.run_id, seq=run.seq + 1, error=ErrorInfo.from_exc(exc)),
                )
            return apply(run, RunCompleted(run_id=run.run_id, seq=run.seq + 1, result=result))
        if status is AgentStatus.FAILED:
            return apply(
                run,
                RunFailed(
                    run_id=run.run_id,
                    seq=run.seq + 1,
                    error=ErrorInfo.of(
                        "workflow_failed", getattr(description.status, "name", "FAILED")
                    ),
                ),
            )
        if status is AgentStatus.CANCELLED:
            return apply(run, RunCancelled(run_id=run.run_id, seq=run.seq + 1))
        return run

    async def _merge_progress(self, run: AgentRun) -> AgentRun:
        """Fill in counters the workflow tracked but the result does not carry.

        The workflow owns the authoritative run state; a query is how a separate
        process reads it. Queries still answer after a workflow closes, so this
        works for finished runs too.
        """
        try:
            progress = await self._handle.query(AgentWorkflow.progress)
        except Exception:  # noqa: BLE001 - progress is informational only
            return run
        return run.model_copy(
            update={
                "iteration": int(progress.get("iteration", 0)),
                "tool_call_count": int(progress.get("tool_calls", 0)),
                "usage": Usage(output_tokens=int(progress.get("tokens", 0))),
                "child_runs": tuple(
                    (CallId(f"child-{i}"), RunId(f"{run.run_id}-child-{i}"))
                    for i in range(int(progress.get("children", 0)))
                ),
            }
        )

    async def progress(self) -> dict:
        return await self._handle.query(AgentWorkflow.progress)

    async def cancel(self, reason: str = "") -> None:
        await self._handle.signal(AgentWorkflow.cancel, reason)

    async def approve(self, call_id: CallId, granted: bool, actor: str = "user") -> None:
        await self._handle.signal(AgentWorkflow.approve, args=[call_id, granted, actor])

    async def pause(self, reason: str = "") -> None:
        await self._handle.signal(AgentWorkflow.pause, reason)

    async def resume(self) -> None:
        await self._handle.signal(AgentWorkflow.resume)


class TemporalAgentRunner:
    def __init__(
        self,
        client: Client,
        *,
        task_queue: str = "skynet-agents",
        execution_timeout_s: int = 3600,
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._timeout = execution_timeout_s

    async def start(self, request: RunRequest) -> TemporalRunHandle:
        handle = await self._client.start_workflow(
            AgentWorkflow.run,
            request,
            id=str(request.context.run_id),
            task_queue=self._task_queue,
            execution_timeout=timedelta(seconds=self._timeout),
            # The agent's own policy owns retries; restarting the whole workflow
            # would replay side effects the run has already observed.
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        return TemporalRunHandle(handle, request)

    async def run(self, request: RunRequest) -> AgentRun:
        handle = await self.start(request)
        return await handle.result()

    def attach(self, run_id: RunId, request: RunRequest) -> TemporalRunHandle:
        """Reconnect to an existing run -- the durability story, from the API."""
        return TemporalRunHandle(self._client.get_workflow_handle(str(run_id)), request)


def _initial(request: RunRequest) -> AgentRun:
    run = new_run(spec=request.spec, task=request.task, context=request.context)
    return apply(
        run,
        RunStarted(
            run_id=run.run_id,
            seq=1,
            spec=request.spec,
            task=request.task,
            context=request.context,
        ),
    )


__all__ = ["TemporalAgentRunner", "TemporalRunHandle", "workflow_status"]
