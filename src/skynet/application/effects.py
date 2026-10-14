"""In-process effect execution.

Every side effect the agent performs is dispatched here, and nowhere else. The
Temporal executor is a sibling of this file, not a replacement for the loop.

The governing invariant: **tool problems are observations, not exceptions.** An
unknown tool, malformed arguments, a denied permission, a timeout, or a crash
inside a tool all become a ``ToolResult`` that goes back to the model so it can
correct itself. Only budget exhaustion and non-retryable model failures end a
run. This is what turns a fragile agent into a resilient one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from itertools import chain
from typing import assert_never

from skynet.application.artifacts import WorkspaceArchiver
from skynet.application.consolidator import Consolidator
from skynet.application.memory_ranking import candidate_limit, rank
from skynet.application.observability import conversation_input
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
from skynet.domain.errors import ErrorInfo, SkynetError
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
from skynet.domain.ids import TaskId
from skynet.domain.memory import MemoryQuery
from skynet.domain.messages import ToolCall
from skynet.domain.models import ModelRequest
from skynet.domain.prompts import render
from skynet.domain.run import AgentRun
from skynet.domain.storage import ObjectRef
from skynet.domain.task import Task
from skynet.domain.tools import ToolContext, ToolKind, ToolResult, ToolSpec, ToolStatus
from skynet.ports.agents import SubAgentSpawner
from skynet.ports.approvals import ApprovalGate
from skynet.ports.clock import Clock, IdGenerator
from skynet.ports.events import EventSink
from skynet.ports.memory import MemoryStore
from skynet.ports.model import ModelProvider
from skynet.ports.prompts import PromptProvider
from skynet.ports.storage import ObjectStore
from skynet.ports.tools import ToolRegistry
from skynet.ports.tracing import SpanKind, Tracer


class InProcessEffectExecutor:
    def __init__(
        self,
        *,
        model: ModelProvider,
        prompts: PromptProvider,
        registry: ToolRegistry,
        memory: MemoryStore,
        objects: ObjectStore,
        spawner: SubAgentSpawner,
        tracer: Tracer,
        sink: EventSink,
        clock: Clock,
        ids: IdGenerator,
        approvals: ApprovalGate,
        archiver: WorkspaceArchiver | None = None,
        max_tool_output_chars: int = 8_000,
    ) -> None:
        self._model = model
        self._prompts = prompts
        self._registry = registry
        self._memory = memory
        self._consolidator = Consolidator(memory, model, clock)
        self._objects = objects
        self._spawner = spawner
        self._tracer = tracer
        self._sink = sink
        self._clock = clock
        self._ids = ids
        self._approvals = approvals
        self._archiver = archiver
        self._max_output = max_tool_output_chars

    async def execute(self, run: AgentRun, decision: Decision) -> Sequence[RunEvent]:
        events = await self._dispatch(run, decision)
        stamped = tuple(
            event.model_copy(update={"seq": run.seq + index + 1, "at": self._clock.now()})
            for index, event in enumerate(events)
        )
        # Observability happens at the seam, so the loop stays free of it.
        for event in stamped:
            await self._sink.publish(event)
        return stamped

    async def _dispatch(self, run: AgentRun, decision: Decision) -> Sequence[RunEvent]:
        match decision:
            case ResolvePrompt(ref=ref, variables=variables):
                template = await self._prompts.fetch(ref)
                return (PromptResolved(run_id=run.run_id, prompt=render(template, variables)),)

            case RecallMemory(query=query):
                return await self._recall(run, query)

            case CallModel(request=request):
                return await self._call_model(run, request)

            case CompactHistory(request=request, through_index=through):
                return await self._compact(run, request, through)

            case ConsolidateMemory():
                return await self._consolidate(run, decision)

            case InvokeTools(calls=calls):
                # Fan out concurrently, but fold results in *argument* order.
                # asyncio.gather preserves that order regardless of completion
                # order, which keeps the event log deterministic.
                grouped = await asyncio.gather(*(self._invoke_one(run, call) for call in calls))
                return tuple(chain.from_iterable(grouped))

            case RequestApproval(calls=calls, reason=reason):
                call_ids = tuple(call.id for call in calls)
                await self._approvals.request(run.run_id, call_ids, reason)
                return (ApprovalRequested(run_id=run.run_id, call_ids=call_ids, reason=reason),)

            case Complete(result=result):
                # Harvest before the run is terminal, so the references are part
                # of the result and therefore part of the replayable log.
                artifacts = await self._harvest(run)
                if artifacts:
                    result = result.model_copy(
                        update={"artifacts": (*result.artifacts, *artifacts)}
                    )
                return (RunCompleted(run_id=run.run_id, result=result),)

            case Fail(error=error):
                # A failed run's files are often the most interesting ones.
                await self._harvest(run)
                return (RunFailed(run_id=run.run_id, error=error),)

            case _:  # pragma: no cover - exhaustiveness guard
                assert_never(decision)

    # -- memory ---------------------------------------------------------------

    async def _recall(self, run: AgentRun, query: MemoryQuery) -> Sequence[RunEvent]:
        """One read across every scope the decision authorised, then rank once.

        The clock is read a single time so the same instant governs liveness
        filtering and recency decay; two readings would let a memory be live for
        the filter and stale for the score. That instant travels in the event,
        so a replay ranks identically.
        """
        now = self._clock.now()
        limit = query.limit
        async with self._tracer.span(
            "memory.recall",
            kind=SpanKind.INTERNAL,
            parent=run.context.trace,
            input={"text": query.text, "scopes": list(query.namespaces())},
            attributes={"limit": limit, "kinds": [kind.value for kind in query.kinds]},
        ) as span:
            candidates = await self._memory.search(
                query.model_copy(update={"limit": candidate_limit(limit), "as_of": now})
            )
            hits = rank(candidates, at=now, limit=limit)
            span.set_output([{"content": h.record.content, "score": h.score} for h in hits])
            span.set_attributes(candidates=len(candidates), returned=len(hits))
        return (MemoryRecalled(run_id=run.run_id, query=query, hits=hits, as_of=now),)

    async def _compact(
        self, run: AgentRun, request: ModelRequest, through_index: int
    ) -> Sequence[RunEvent]:
        """Summarise the head of the conversation.

        Degrades rather than propagating. Emitting ``ModelFailed`` here would
        let a summarisation hiccup consume the run's model-failure budget and
        ultimately fail it -- and a run must never die of housekeeping.
        """
        async with self._tracer.span(
            "history.compact",
            kind=SpanKind.GENERATION,
            parent=run.context.trace,
            input=conversation_input(request),
            attributes={"model": request.params.model, "through_index": through_index},
        ) as span:
            try:
                response = await self._model.generate(request)
            except Exception as exc:  # noqa: BLE001 - adapters may raise anything
                span.record_error(exc)
                return (
                    HistoryCompacted(
                        run_id=run.run_id,
                        through_index=through_index,
                        error=ErrorInfo.from_exc(exc),
                    ),
                )
            summary = response.message.text.strip()
            span.set_output(summary)
            return (
                HistoryCompacted(
                    run_id=run.run_id,
                    through_index=through_index,
                    summary=summary,
                    replaced=through_index,
                    error=(
                        None if summary else ErrorInfo.of("compaction", "empty summary returned")
                    ),
                ),
            )

    async def _consolidate(self, run: AgentRun, decision: ConsolidateMemory) -> Sequence[RunEvent]:
        """One consolidation pass, reported as an event whatever happens.

        Nothing here can fail the run: every error becomes an event that still
        advances the watermark, because a memory subsystem able to kill an
        otherwise finished run is worse than one that occasionally forgets.
        """
        async with self._tracer.span(
            "memory.consolidate",
            kind=SpanKind.GENERATION,
            parent=run.context.trace,
            attributes={"mode": decision.mode.value, "scope": decision.scope.namespace()},
        ) as span:
            outcome = await self._consolidator.run(
                decision,
                run_id=run.run_id,
                objective=run.task.objective,
                messages=run.messages[decision.from_index : decision.through_index],
                default_ttl_hours=run.spec.memory.default_ttl_hours,
            )
            span.set_output([operation.model_dump() for operation in outcome.operations])
            span.set_attributes(
                written=len(outcome.written),
                invalidated=len(outcome.invalidated),
                rejected=outcome.rejected,
                error=outcome.error.message if outcome.error else "",
            )
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

    # -- model ----------------------------------------------------------------

    async def _call_model(self, run: AgentRun, request) -> Sequence[RunEvent]:
        async with self._tracer.span(
            "llm.generate",
            kind=SpanKind.GENERATION,
            parent=run.context.trace,
            input=conversation_input(request),
            attributes={
                "model": request.params.model,
                "tools": [t.name for t in request.tools],
            },
        ) as span:
            started = self._clock.now()
            try:
                response = await self._model.generate(request)
            except SkynetError as exc:
                span.record_error(exc)
                return (ModelFailed(run_id=run.run_id, error=ErrorInfo.from_exc(exc)),)
            except Exception as exc:  # noqa: BLE001 - adapters may raise anything
                span.record_error(exc)
                return (ModelFailed(run_id=run.run_id, error=ErrorInfo.from_exc(exc)),)

            span.set_output(response.message.text)
            span.set_attributes(
                finish_reason=str(response.finish_reason),
                model=response.model or request.params.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            elapsed = int((self._clock.now() - started).total_seconds() * 1000)
            return (ModelResponded(run_id=run.run_id, response=response, latency_ms=elapsed),)

    # -- tools ----------------------------------------------------------------

    async def _invoke_one(self, run: AgentRun, call: ToolCall) -> Sequence[RunEvent]:
        spec = self._registry.spec(call.name)
        if spec is None:
            # Echo the valid names: small models hallucinate tool names often,
            # and being shown the real list recovers most of those turns.
            return self._failed(
                run,
                call,
                ToolStatus.NOT_FOUND,
                f"unknown tool '{call.name}'. Available tools: {', '.join(self._registry.names())}",
            )

        if run.approvals.get(call.id) is False:
            return self._failed(run, call, ToolStatus.DENIED, "denied by operator")

        missing = run.context.permissions.missing(spec.required_permissions)
        if missing:
            return self._failed(
                run,
                call,
                ToolStatus.DENIED,
                f"missing permission(s): {', '.join(sorted(missing))}",
            )

        if spec.kind is ToolKind.DELEGATION:
            return await self._delegate(run, call, spec)

        return await self._run_tool(run, call, spec)

    async def _run_tool(self, run: AgentRun, call: ToolCall, spec: ToolSpec) -> Sequence[RunEvent]:
        ctx = ToolContext(
            run_id=run.run_id,
            agent_name=run.spec.name,
            permissions=run.context.permissions,
            depth=run.context.depth,
            workspace_id=run.context.workspace_id,
            deadline=run.context.deadline,
            variables=run.context.variables,
        )
        timeout = min(spec.timeout_s, run.spec.limits.tool_timeout_s)
        started = self._clock.now()

        async with self._tracer.span(
            f"tool.{call.name}",
            kind=SpanKind.TOOL,
            parent=run.context.trace,
            input=dict(call.arguments),
            attributes={"tool": call.name, "call_id": str(call.id)},
        ) as span:
            try:
                async with asyncio.timeout(timeout):
                    result = await self._registry.get(call.name).invoke(call, ctx)
            except TimeoutError:
                result = self._error_result(
                    call, ToolStatus.TIMEOUT, f"tool timed out after {timeout}s"
                )
            except Exception as exc:  # noqa: BLE001 - a tool must never kill a run
                span.record_error(exc)
                result = self._error_result(call, ToolStatus.ERROR, str(exc))

            result = await self._spill(run, result)
            elapsed = int((self._clock.now() - started).total_seconds() * 1000)
            result = result.model_copy(update={"duration_ms": elapsed})
            span.set_output(result.content)
            span.set_attributes(status=str(result.status))

        return (ToolCompleted(run_id=run.run_id, result=result),)

    async def _delegate(self, run: AgentRun, call: ToolCall, spec: ToolSpec) -> Sequence[RunEvent]:
        if run.context.depth + 1 > run.spec.limits.max_depth:
            return self._failed(
                run,
                call,
                ToolStatus.DENIED,
                f"maximum delegation depth ({run.spec.limits.max_depth}) reached",
            )

        agent = str(call.arguments.get("agent", ""))
        if agent not in run.spec.sub_agents:
            return self._failed(
                run,
                call,
                ToolStatus.DENIED,
                f"agent '{agent}' is not delegable from '{run.spec.name}'. "
                f"Allowed: {', '.join(run.spec.sub_agents) or 'none'}",
            )

        child_task = Task(
            id=TaskId(self._ids.new_id("task")),
            objective=str(call.arguments.get("objective", "")),
            input=call.arguments.get("input") or {},
            metadata={"parent_run": str(run.run_id), "parent_call": str(call.id)},
        )

        async with self._tracer.span(
            f"subagent.{agent}",
            kind=SpanKind.AGENT,
            parent=run.context.trace,
            input={"agent": agent, "objective": child_task.objective},
            attributes={"depth": run.context.depth + 1},
        ) as span:
            try:
                child_id, result = await self._spawner.spawn(
                    agent=agent, task=child_task, parent=run.context, call_id=call.id
                )
            except Exception as exc:  # noqa: BLE001 - a child failure is an observation
                span.record_error(exc)
                return self._failed(
                    run, call, ToolStatus.ERROR, f"sub-agent '{agent}' failed: {exc}"
                )
            span.set_output(result.output)

        return (
            ChildRunStarted(run_id=run.run_id, call_id=call.id, child_run_id=child_id, agent=agent),
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

    # -- helpers --------------------------------------------------------------

    async def _harvest(self, run: AgentRun) -> tuple[ObjectRef, ...]:
        if self._archiver is None:
            return ()
        return await self._archiver.harvest(
            run_id=run.run_id,
            workspace_id=run.context.workspace_id or str(run.run_id),
        )

    async def _spill(self, run: AgentRun, result: ToolResult) -> ToolResult:
        """Keep oversized output out of the context window but not lost.

        Anything past the cap is written to object storage and referenced from
        the result, so the agent can still reach it via a file tool.
        """
        if len(result.content) <= self._max_output:
            return result

        key = f"runs/{run.run_id}/tool-output/{result.call_id}.txt"
        try:
            ref = await self._objects.put(key, result.content.encode(), content_type="text/plain")
        except Exception:  # noqa: BLE001 - storage must not break a run
            ref = None

        head = result.content[: self._max_output]
        note = f"\n\n[output truncated at {self._max_output} chars"
        note += f"; full output stored at {ref.uri}]" if ref else "]"
        return result.model_copy(
            update={
                "content": head + note,
                "truncated": True,
                "artifacts": (*result.artifacts, ref) if ref else result.artifacts,
            }
        )

    def _failed(
        self, run: AgentRun, call: ToolCall, status: ToolStatus, message: str
    ) -> Sequence[RunEvent]:
        return (
            ToolCompleted(
                run_id=run.run_id,
                result=self._error_result(call, status, message),
            ),
        )

    @staticmethod
    def _error_result(call: ToolCall, status: ToolStatus, message: str) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=call.name,
            status=status,
            content=f"ERROR: {message}",
            error=ErrorInfo.of(str(status), message),
        )


__all__ = ["InProcessEffectExecutor"]
