"""The Temporal effect executor.

This file is the entire cost of making the agent durable. It implements the same
``EffectExecutor`` port as the in-process one, but every side effect becomes an
activity or a child workflow. The agent loop, the policy and the reducer are
imported unchanged from the core.

Determinism rules observed here:
  * time and ids come from ``workflow.now()`` / ``workflow.uuid4()``;
  * fan-out results are folded in argument order, never completion order;
  * nothing does I/O directly -- every effect is a Temporal command.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import timedelta
from itertools import chain
from typing import TYPE_CHECKING, Any, assert_never

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from skynet.adapters.temporal.activities import (
        CallModelInput,
        CallModelOutput,
        ConsolidateMemoryInput,
        HarvestWorkspaceInput,
        InvokeToolInput,
        PublishEventsInput,
        RecallMemoryInput,
        RecallMemoryOutput,
        ReleaseWorkspaceInput,
        ResolvePromptInput,
    )
    from skynet.application.consolidator import ConsolidationOutcome
    from skynet.domain.decisions import (
        CallModel,
        CompactHistory,
        Complete,
        ConsolidateMemory,
        Decision,
        Fail,
        InvokeTools,
        RecallMemory,
        RequestApproval,
        ResolvePrompt,
    )
    from skynet.domain.errors import ErrorInfo
    from skynet.domain.events import (
        ApprovalRequested,
        ChildRunStarted,
        HistoryCompacted,
        MemoryConsolidated,
        MemoryRecalled,
        ModelFailed,
        ModelResponded,
        PromptResolved,
        RunCompleted,
        RunEvent,
        RunFailed,
        ToolCompleted,
    )
    from skynet.domain.ids import RunId, TaskId
    from skynet.domain.messages import ToolCall
    from skynet.domain.models import ModelRequest
    from skynet.domain.prompts import RenderedPrompt
    from skynet.domain.run import AgentRun
    from skynet.domain.storage import ObjectRef
    from skynet.domain.task import AgentResult, Task
    from skynet.domain.tools import ToolKind, ToolResult, ToolStatus
    from skynet.ports.agents import RunRequest

if TYPE_CHECKING:
    from skynet.adapters.temporal.workflow import AgentWorkflow

# Retry policies differ by effect: transport failures deserve retries, but the
# agent's own decisions must not be silently repeated.
_MODEL_RETRY = RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=1))
_TOOL_RETRY = RetryPolicy(maximum_attempts=1)
_QUICK_RETRY = RetryPolicy(maximum_attempts=3)


class WorkflowEffectExecutor:
    """``EffectExecutor`` whose effects are Temporal commands."""

    def __init__(self, host: AgentWorkflow) -> None:
        self._host = host

    async def execute(self, run: AgentRun, decision: Decision) -> Sequence[RunEvent]:
        events = await self._dispatch(run, decision)
        stamped = tuple(
            event.model_copy(update={"seq": run.seq + index + 1, "at": workflow.now()})
            for index, event in enumerate(events)
        )
        await self._publish(stamped)
        return stamped

    async def _dispatch(self, run: AgentRun, decision: Decision) -> Sequence[RunEvent]:
        match decision:
            case ResolvePrompt(ref=ref, variables=variables):
                prompt = await workflow.execute_activity(
                    "resolve_prompt",
                    ResolvePromptInput(ref=ref, variables=variables),
                    result_type=RenderedPrompt,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_QUICK_RETRY,
                )
                return (PromptResolved(run_id=run.run_id, prompt=prompt),)

            case RecallMemory(query=query):
                # Recall now means embed plus a vector search across several
                # scopes on a possibly cold pool, so the old 30s is too tight.
                # It is a read, which makes retries free.
                recalled: RecallMemoryOutput = await workflow.execute_activity(
                    "recall_memory",
                    RecallMemoryInput(query=query, trace=run.context.trace),
                    result_type=RecallMemoryOutput,
                    start_to_close_timeout=timedelta(seconds=60),
                    schedule_to_close_timeout=timedelta(seconds=120),
                    retry_policy=_QUICK_RETRY,
                )
                return (
                    MemoryRecalled(
                        run_id=run.run_id,
                        query=query,
                        hits=recalled.hits,
                        as_of=recalled.as_of,
                    ),
                )

            case CallModel(request=request):
                outcome: CallModelOutput = await workflow.execute_activity(
                    "call_model",
                    CallModelInput(run_id=run.run_id, request=request, trace=run.context.trace),
                    result_type=CallModelOutput,
                    start_to_close_timeout=timedelta(seconds=600),
                    retry_policy=_MODEL_RETRY,
                )
                if outcome.error is not None or outcome.response is None:
                    return (
                        ModelFailed(
                            run_id=run.run_id,
                            error=outcome.error or ErrorInfo.of("model", "no response"),
                        ),
                    )
                return (
                    ModelResponded(
                        run_id=run.run_id,
                        response=outcome.response,
                        latency_ms=outcome.latency_ms,
                    ),
                )

            case CompactHistory(request=request, through_index=through):
                return await self._compact(run, request, through)

            case ConsolidateMemory():
                return await self._consolidate(run, decision)

            case InvokeTools(calls=calls):
                # Fan out concurrently. asyncio.gather returns results in
                # argument order regardless of completion order, so the fold
                # stays deterministic under replay.
                grouped = await asyncio.gather(*(self._start_tool(run, call) for call in calls))
                return tuple(chain.from_iterable(grouped))

            case RequestApproval(calls=calls, reason=reason):
                return (
                    ApprovalRequested(
                        run_id=run.run_id,
                        call_ids=tuple(call.id for call in calls),
                        reason=reason,
                    ),
                )

            case Complete(result=result):
                artifacts = await self._harvest_workspace(run)
                if artifacts:
                    result = result.model_copy(
                        update={"artifacts": (*result.artifacts, *artifacts)}
                    )
                await self._release_workspace(run)
                return (RunCompleted(run_id=run.run_id, result=result),)

            case Fail(error=error):
                # A failed run's files are often the most interesting ones.
                await self._harvest_workspace(run)
                await self._release_workspace(run)
                return (RunFailed(run_id=run.run_id, error=error),)

            case _:  # pragma: no cover
                assert_never(decision)

    # -- history ---------------------------------------------------------------

    async def _compact(self, run: AgentRun, request: ModelRequest, through_index: int):
        """Summarise through the activity the executor already has.

        Reusing ``call_model`` is why compaction needs no activity of its own.
        Failures degrade to an event carrying the error: emitting ``ModelFailed``
        would spend the run's model-failure budget on housekeeping and could
        ultimately fail a run over a summary.
        """
        try:
            outcome: CallModelOutput = await workflow.execute_activity(
                "call_model",
                CallModelInput(
                    run_id=run.run_id,
                    request=request,
                    trace=run.context.trace,
                    span_name="history.compact",
                ),
                result_type=CallModelOutput,
                start_to_close_timeout=timedelta(seconds=600),
                retry_policy=_MODEL_RETRY,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a run over a summary
            return (
                HistoryCompacted(
                    run_id=run.run_id,
                    through_index=through_index,
                    error=ErrorInfo.of("compaction", str(exc)),
                ),
            )

        if outcome.error is not None or outcome.response is None:
            return (
                HistoryCompacted(
                    run_id=run.run_id,
                    through_index=through_index,
                    error=outcome.error or ErrorInfo.of("compaction", "no summary returned"),
                ),
            )

        summary = outcome.response.message.text.strip()
        return (
            HistoryCompacted(
                run_id=run.run_id,
                through_index=through_index,
                summary=summary,
                replaced=through_index,
                error=None if summary else ErrorInfo.of("compaction", "empty summary returned"),
            ),
        )

    async def _consolidate(self, run: AgentRun, decision: ConsolidateMemory):
        """One activity rather than a composed model call plus writes.

        A partial failure between "the model says this fact is contradicted" and
        "the store wrote the replacement" would need compensation logic nothing
        else here has; keeping it atomic means the only retry semantics needed
        are the decision's ``idempotency_key``, from which every written id
        derives. Two attempts, because a retry is safe but a third is just a
        slower way to reach the same answer.
        """
        try:
            outcome: ConsolidationOutcome = await workflow.execute_activity(
                "consolidate_memory",
                ConsolidateMemoryInput(
                    run_id=run.run_id,
                    decision=decision,
                    objective=run.task.objective,
                    messages=run.messages[decision.from_index : decision.through_index],
                    default_ttl_hours=run.spec.memory.default_ttl_hours,
                    trace=run.context.trace,
                ),
                result_type=ConsolidationOutcome,
                start_to_close_timeout=timedelta(seconds=180),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception as exc:  # noqa: BLE001 - memory must never fail a run
            outcome = ConsolidationOutcome(error=ErrorInfo.of("memory", str(exc)))

        return (
            MemoryConsolidated(
                run_id=run.run_id,
                mode=decision.mode,
                scope=decision.scope,
                through_index=decision.through_index,
                operations=outcome.operations,
                written=outcome.written,
                invalidated=outcome.invalidated,
                error=outcome.error,
            ),
        )

    # -- tools ----------------------------------------------------------------

    def _start_tool(self, run: AgentRun, call: ToolCall):
        spec = self._host.catalog.spec(call.name)

        if run.approvals.get(call.id) is False:
            return _immediate(run, _error(call, ToolStatus.DENIED, "denied by operator"))

        if spec is not None and spec.kind is ToolKind.DELEGATION:
            return self._delegate(run, call)

        return self._tool_activity(run, call, spec)

    async def _tool_activity(self, run: AgentRun, call: ToolCall, spec: Any):
        timeout = min(
            spec.timeout_s if spec else run.spec.limits.tool_timeout_s,
            run.spec.limits.tool_timeout_s,
        )
        result: ToolResult = await workflow.execute_activity(
            "invoke_tool",
            InvokeToolInput(
                run_id=run.run_id,
                agent_name=run.spec.name,
                call=call,
                permissions=run.context.permissions,
                depth=run.context.depth,
                workspace_id=run.context.workspace_id,
                timeout_s=timeout,
                trace=run.context.trace,
            ),
            result_type=ToolResult,
            start_to_close_timeout=timedelta(seconds=timeout + 30),
            retry_policy=_TOOL_RETRY,
        )
        return (ToolCompleted(run_id=run.run_id, result=result),)

    async def _delegate(self, run: AgentRun, call: ToolCall):
        """A sub-agent is a child workflow.

        Nesting and parallel children come from Temporal itself; the agent loop
        is unaware that any of this is happening.
        """
        if run.context.depth + 1 > run.spec.limits.max_depth:
            return (
                ToolCompleted(
                    run_id=run.run_id,
                    result=_error(
                        call,
                        ToolStatus.DENIED,
                        f"maximum delegation depth ({run.spec.limits.max_depth}) reached",
                    ),
                ),
            )

        agent = str(call.arguments.get("agent", ""))
        if agent not in run.spec.sub_agents:
            return (
                ToolCompleted(
                    run_id=run.run_id,
                    result=_error(
                        call,
                        ToolStatus.DENIED,
                        f"agent '{agent}' is not delegable from '{run.spec.name}'. "
                        f"Allowed: {', '.join(run.spec.sub_agents) or 'none'}",
                    ),
                ),
            )

        try:
            child_spec = self._host.catalog_agents.get(agent)
        except Exception as exc:  # noqa: BLE001
            return (
                ToolCompleted(
                    run_id=run.run_id,
                    result=_error(call, ToolStatus.ERROR, str(exc)),
                ),
            )

        child_run_id = RunId(f"{run.run_id}-{call.id}")
        child_context = run.context.model_copy(
            update={
                "run_id": child_run_id,
                "agent_name": agent,
                "permissions": run.context.permissions.narrowed(
                    child_spec.default_permissions.grants
                ),
                "depth": run.context.depth + 1,
                "parent_run_id": run.run_id,
            }
        )
        child_task = Task(
            id=TaskId(f"{run.task.id}-{call.id}"),
            objective=str(call.arguments.get("objective", "")),
            metadata={"parent_run": str(run.run_id)},
        )

        try:
            result = await workflow.execute_child_workflow(
                "AgentWorkflow",
                RunRequest(spec=child_spec, task=child_task, context=child_context),
                result_type=AgentResult,
                id=str(child_run_id),
                task_queue=self._host.task_queue,
            )
        except Exception as exc:  # noqa: BLE001 - a child failure is an observation
            return (
                ToolCompleted(
                    run_id=run.run_id,
                    result=_error(call, ToolStatus.ERROR, f"sub-agent '{agent}' failed: {exc}"),
                ),
            )

        return (
            ChildRunStarted(
                run_id=run.run_id,
                call_id=call.id,
                child_run_id=child_run_id,
                agent=agent,
            ),
            ToolCompleted(
                run_id=run.run_id,
                result=ToolResult(
                    call_id=call.id,
                    name=call.name,
                    status=ToolStatus.OK,
                    content=result.output,
                    data=result.data,
                    artifacts=result.artifacts,
                ),
            ),
        )

    # -- events ---------------------------------------------------------------

    async def _harvest_workspace(self, run: AgentRun) -> tuple[ObjectRef, ...]:
        """Preserve the run's files. Best-effort: never fails a finished run."""
        try:
            refs = await workflow.execute_activity(
                "harvest_workspace",
                HarvestWorkspaceInput(
                    run_id=run.run_id,
                    workspace_id=run.context.workspace_id or str(run.run_id),
                ),
                result_type=list[ObjectRef],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            return tuple(refs)
        except Exception:  # noqa: BLE001
            workflow.logger.warning("workspace harvest failed", exc_info=True)
            return ()

    async def _release_workspace(self, run: AgentRun) -> None:
        """Discard the run's sandbox. Best-effort: never fails a finished run."""
        try:
            await workflow.execute_activity(
                "release_workspace",
                ReleaseWorkspaceInput(workspace_id=run.context.workspace_id or str(run.run_id)),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        except Exception:  # noqa: BLE001
            workflow.logger.warning("workspace release failed", exc_info=True)

    async def publish(self, *events: RunEvent) -> None:
        """Publish events that did not come from a decision (e.g. RunStarted)."""
        await self._publish(events)

    async def _publish(self, events: Sequence[RunEvent]) -> None:
        """Mirror events out for the API stream. Never allowed to fail a run."""
        try:
            await workflow.execute_local_activity(
                "publish_events",
                PublishEventsInput(events=[e.model_dump(mode="json") for e in events]),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
        except Exception:  # noqa: BLE001
            workflow.logger.debug("event publication failed", exc_info=True)


async def _immediate(run: AgentRun, result: ToolResult):
    return (ToolCompleted(run_id=run.run_id, result=result),)


def _error(call: ToolCall, status: ToolStatus, message: str) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        name=call.name,
        status=status,
        content=f"ERROR: {message}",
        error=ErrorInfo.of(str(status), message),
    )


__all__ = ["WorkflowEffectExecutor"]
