"""In-process agent execution.

The default runner: no infrastructure, no workflow engine, just an event loop.
It is what the CLI and the whole test suite use, and it is the reference against
which the Temporal runner's behaviour is compared.

It implements ``SubAgentSpawner`` as well as ``AgentRunner``, which is how a
parent agent's delegation tool reaches a child run. Nesting therefore works to
arbitrary depth (bounded by ``limits.max_depth``) with no extra machinery.
"""

from __future__ import annotations

import asyncio
import logging

from skynet.application.artifacts import WorkspaceArchiver
from skynet.application.effects import InProcessEffectExecutor
from skynet.application.loop import AgentLoop
from skynet.application.policy import AgentPolicy
from skynet.application.services import RuntimeServices
from skynet.domain.agent import AgentContext, FinishMode
from skynet.domain.errors import SubAgentFailed
from skynet.domain.events import (
    ApprovalResolved,
    RunCancelled,
    RunEvent,
    RunStarted,
)
from skynet.domain.ids import CallId, RunId
from skynet.domain.run import AgentRun, AgentStatus, apply, new_run
from skynet.domain.task import AgentResult, Task
from skynet.ports.agents import RunRequest
from skynet.ports.model import ProviderCapabilities
from skynet.ports.tracing import SpanKind

logger = logging.getLogger(__name__)


class _Suspensions:
    """Parks suspended runs until an external event arrives.

    The in-process analogue of a Temporal signal. Same protocol, different
    transport -- which is exactly the point.
    """

    def __init__(self) -> None:
        self._events: dict[RunId, asyncio.Event] = {}
        self._pending: dict[RunId, list[RunEvent]] = {}

    def deliver(self, run_id: RunId, event: RunEvent) -> None:
        self._pending.setdefault(run_id, []).append(event)
        self._events.setdefault(run_id, asyncio.Event()).set()

    async def wait(self, run_id: RunId) -> None:
        event = self._events.setdefault(run_id, asyncio.Event())
        await event.wait()
        event.clear()

    def drain(self, run_id: RunId) -> tuple[RunEvent, ...]:
        return tuple(self._pending.pop(run_id, []))

    def has_pending(self, run_id: RunId) -> bool:
        return bool(self._pending.get(run_id))


class InProcessRunHandle:
    def __init__(
        self,
        runner: InProcessAgentRunner,
        run_id: RunId,
        task: asyncio.Task[AgentRun],
        initial: AgentRun,
    ) -> None:
        self._runner = runner
        self._run_id = run_id
        self._task = task
        self._initial = initial

    @property
    def run_id(self) -> RunId:
        return self._run_id

    async def result(self) -> AgentRun:
        return await self._task

    async def snapshot(self) -> AgentRun:
        if self._task.done() and not self._task.cancelled() and self._task.exception() is None:
            return self._task.result()
        stored = await self._runner.services.runs.load(self._run_id)
        return stored if stored is not None else self._initial

    async def cancel(self, reason: str = "") -> None:
        self._runner.signal(self._run_id, RunCancelled(run_id=self._run_id, reason=reason))

    async def approve(self, call_id: CallId, granted: bool, actor: str = "user") -> None:
        self._runner.signal(
            self._run_id,
            ApprovalResolved(run_id=self._run_id, call_id=call_id, granted=granted, actor=actor),
        )


class InProcessAgentRunner:
    def __init__(self, services: RuntimeServices) -> None:
        self._services = services
        self._suspensions = _Suspensions()
        self._tasks: dict[RunId, asyncio.Task[AgentRun]] = {}

    # -- AgentRunner ----------------------------------------------------------

    async def start(self, request: RunRequest) -> InProcessRunHandle:
        task = asyncio.create_task(self.run(request))
        self._tasks[request.context.run_id] = task
        initial = new_run(spec=request.spec, task=request.task, context=request.context).model_copy(
            update={"status": AgentStatus.RUNNING}
        )
        return InProcessRunHandle(self, request.context.run_id, task, initial)

    @property
    def services(self) -> RuntimeServices:
        return self._services

    async def run(self, request: RunRequest) -> AgentRun:
        services = self._services
        run = new_run(
            spec=request.spec,
            task=request.task,
            context=request.context,
            finish_mode=self._finish_mode(services.model.capabilities),
        )
        loop = AgentLoop(
            policy=AgentPolicy(services.registry, services.context_builder),
            effects=InProcessEffectExecutor(
                model=services.model,
                prompts=services.prompts,
                registry=services.registry,
                memory=services.memory,
                objects=services.objects,
                spawner=self,
                tracer=services.tracer,
                sink=services.sink,
                clock=services.clock,
                ids=services.ids,
                approvals=services.approvals,
                archiver=WorkspaceArchiver(services.sandboxes, services.objects),
            ),
        )

        started = RunStarted(
            run_id=run.run_id,
            seq=run.seq + 1,
            at=services.clock.now(),
            spec=request.spec,
            task=request.task,
            context=request.context,
        )
        await services.sink.publish(started)
        await services.runs.append(run.run_id, (started,))
        run = apply(run, started)

        async with services.tracer.span(
            f"agent.{request.spec.name}",
            kind=SpanKind.AGENT,
            parent=request.context.trace,
            attributes={"run_id": str(run.run_id), "objective": request.task.objective},
        ) as span:
            # The outer loop is the resume protocol: the inner loop returns when
            # the run suspends, we wait for the external event, fold it in, and
            # continue. The Temporal workflow has the identical shape.
            while True:
                run = await loop.run(run)
                await services.runs.snapshot(run)
                if run.status.is_terminal:
                    break
                await self._suspensions.wait(run.run_id)
                for event in self._suspensions.drain(run.run_id):
                    run = apply(run, event)

            span.set_output(run.result.output if run.result else str(run.status))
            span.set_attributes(status=str(run.status), iterations=run.iteration)

        await services.runs.snapshot(run)
        await self._release_workspace(run)
        return run

    async def _release_workspace(self, run: AgentRun) -> None:
        """End the run's isolated environment.

        A workspace intentionally outlives each individual tool call, so
        something has to close it; otherwise every finished run leaves its
        sandbox behind.
        """
        sandboxes = self._services.sandboxes
        if sandboxes is None:
            return
        try:
            await sandboxes.release(run.context.workspace_id or str(run.run_id))
        except Exception:  # noqa: BLE001 - cleanup must not fail a finished run
            logger.warning("could not release workspace for %s", run.run_id, exc_info=True)

    def signal(self, run_id: RunId, event: RunEvent) -> None:
        """Deliver an external event (approval, cancellation) to a parked run."""
        self._suspensions.deliver(run_id, event)

    # -- SubAgentSpawner ------------------------------------------------------

    async def spawn(
        self, *, agent: str, task: Task, parent: AgentContext, call_id: CallId
    ) -> tuple[RunId, AgentResult]:
        spec = self._services.agents.get(agent)
        child_context = AgentContext(
            run_id=RunId(self._services.ids.new_id("run")),
            agent_name=agent,
            # A child can only ever hold a subset of its parent's authority.
            permissions=parent.permissions.narrowed(spec.default_permissions.grants),
            depth=parent.depth + 1,
            parent_run_id=parent.run_id,
            trace=parent.trace.child(),
            workspace_id=parent.workspace_id,
            deadline=parent.deadline,
            variables=parent.variables,
        )
        child = await self.run(RunRequest(spec=spec, task=task, context=child_context))
        if child.status is not AgentStatus.COMPLETED or child.result is None:
            raise SubAgentFailed(
                f"sub-agent '{agent}' ended as {child.status}: "
                f"{child.last_error.message if child.last_error else 'no result'}"
            )
        return child.run_id, child.result

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _finish_mode(capabilities: ProviderCapabilities) -> FinishMode:
        return (
            FinishMode.FINAL_ANSWER_TOOL
            if capabilities.emulated_tools
            else FinishMode.NO_TOOL_CALLS
        )


__all__ = ["InProcessAgentRunner", "InProcessRunHandle"]
