"""Temporal activities -- the side effects of an agent run.

Each activity is one non-deterministic operation lifted out of the workflow.
They hold the real adapters (HTTP clients, Docker, MinIO); the workflow holds
none of that, which is what makes it replayable.

Activities are constructed around a ``RuntimeServices`` bundle rather than
reaching for globals, so a worker's dependencies stay explicit and swappable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from temporalio import activity

from skynet.application.artifacts import WorkspaceArchiver
from skynet.application.consolidator import ConsolidationOutcome, Consolidator
from skynet.application.memory_ranking import candidate_limit, rank
from skynet.application.observability import conversation_input
from skynet.application.services import RuntimeServices
from skynet.domain.agent import TraceContext
from skynet.domain.decisions import ConsolidateMemory
from skynet.domain.errors import ErrorInfo, SkynetError
from skynet.domain.ids import RunId
from skynet.domain.memory import MemoryHit, MemoryQuery
from skynet.domain.messages import Message, ToolCall
from skynet.domain.models import ModelRequest, ModelResponse
from skynet.domain.permissions import PermissionSet
from skynet.domain.prompts import PromptRef, RenderedPrompt, render
from skynet.domain.storage import ObjectRef
from skynet.domain.tools import ToolContext, ToolResult, ToolStatus
from skynet.ports.tracing import SpanKind

logger = logging.getLogger(__name__)


# -- activity payloads --------------------------------------------------------


class ResolvePromptInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    ref: PromptRef
    variables: dict[str, str] = {}


class RecallMemoryInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    query: MemoryQuery
    trace: TraceContext = TraceContext()
    """The whole query, scopes included. The workflow already decided what this
    run may read; re-deriving it here would be a second opinion on authority."""


class RecallMemoryOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    hits: tuple[MemoryHit, ...] = ()
    as_of: datetime | None = None
    """Returned so the event records the instant ranking actually used."""


class ConsolidateMemoryInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: RunId
    decision: ConsolidateMemory
    objective: str = ""
    messages: tuple[Message, ...] = ()
    """The slice to read. Sent with the payload so the activity needs no access
    to run state, which is what keeps it a plain function of its input."""
    default_ttl_hours: float | None = None
    trace: TraceContext = TraceContext()


class CallModelInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: RunId
    request: ModelRequest
    trace: TraceContext = TraceContext()
    span_name: str = "llm.generate"
    """Compaction reuses this activity, and a trace that labelled its
    summarisation call the same as a reasoning turn would leave "why did this
    run make an extra model call?" unanswerable under Temporal while it stays
    answerable in-process."""


class CallModelOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    response: ModelResponse | None = None
    error: ErrorInfo | None = None
    latency_ms: int = 0


class InvokeToolInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: RunId
    agent_name: str
    call: ToolCall
    permissions: PermissionSet
    depth: int = 0
    workspace_id: str | None = None
    timeout_s: float = 60.0
    max_output_chars: int = 8_000
    trace: TraceContext = TraceContext()


class ReleaseWorkspaceInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    workspace_id: str


class HarvestWorkspaceInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: RunId
    workspace_id: str


class PublishEventsInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    events: list[dict[str, Any]] = []


class AgentActivities:
    """Bound activity implementations for one worker."""

    def __init__(self, services: RuntimeServices) -> None:
        self._services = services

    def all(self) -> list[Any]:
        return [
            self.resolve_prompt,
            self.recall_memory,
            self.consolidate_memory,
            self.call_model,
            self.invoke_tool,
            self.publish_events,
            self.release_workspace,
            self.harvest_workspace,
        ]

    @activity.defn(name="resolve_prompt")
    async def resolve_prompt(self, payload: ResolvePromptInput) -> RenderedPrompt:
        template = await self._services.prompts.fetch(payload.ref)
        return render(template, payload.variables)

    @activity.defn(name="recall_memory")
    async def recall_memory(self, payload: RecallMemoryInput) -> RecallMemoryOutput:
        now = self._services.clock.now()
        query = payload.query
        async with self._services.tracer.span(
            "memory.recall",
            kind=SpanKind.INTERNAL,
            parent=payload.trace,
            input={"text": query.text, "scopes": list(query.namespaces())},
            attributes={"limit": query.limit},
        ) as span:
            candidates = await self._services.memory.search(
                query.model_copy(update={"limit": candidate_limit(query.limit), "as_of": now})
            )
            span.set_output([hit.record.content for hit in candidates[: query.limit]])
            span.set_attributes(candidates=len(candidates))
        # Ranking is pure but runs here, not in the workflow, so ``as_of`` is
        # read from the clock exactly once and travels in the result. Ranking on
        # the workflow side would have the in-process and durable engines score
        # the same run at subtly different instants.
        return RecallMemoryOutput(hits=rank(candidates, at=now, limit=query.limit), as_of=now)

    @activity.defn(name="consolidate_memory")
    async def consolidate_memory(self, payload: ConsolidateMemoryInput) -> ConsolidationOutcome:
        """Errors are returned, never raised.

        Raising would let Temporal retry a *partially applied* pass, and would
        ultimately fail the workflow -- over housekeeping. The same
        ``Consolidator`` the in-process engine uses runs here, so a run
        remembers the same things whichever engine executed it.
        """
        services = self._services
        decision = payload.decision
        async with services.tracer.span(
            "memory.consolidate",
            kind=SpanKind.GENERATION,
            parent=payload.trace,
            attributes={"mode": decision.mode.value, "scope": decision.scope.namespace()},
        ) as span:
            outcome = await Consolidator(services.memory, services.model, services.clock).run(
                decision,
                run_id=payload.run_id,
                objective=payload.objective,
                messages=payload.messages,
                default_ttl_hours=payload.default_ttl_hours,
            )
            span.set_output([operation.model_dump() for operation in outcome.operations])
            span.set_attributes(
                written=len(outcome.written),
                invalidated=len(outcome.invalidated),
                rejected=outcome.rejected,
                error=outcome.error.message if outcome.error else "",
            )
        return outcome

    @activity.defn(name="call_model")
    async def call_model(self, payload: CallModelInput) -> CallModelOutput:
        """Model failures are returned, not raised.

        Retrying is the agent policy's decision, so it stays visible in the
        event log. Letting Temporal retry silently would hide it.
        """
        services = self._services
        started = services.clock.now()
        async with services.tracer.span(
            payload.span_name,
            kind=SpanKind.GENERATION,
            parent=payload.trace,
            input=conversation_input(payload.request),
            attributes={
                "model": payload.request.params.model,
                "tools": [t.name for t in payload.request.tools],
            },
        ) as span:
            try:
                response = await services.model.generate(payload.request)
            except SkynetError as exc:
                span.record_error(exc)
                return CallModelOutput(error=ErrorInfo.from_exc(exc))
            except Exception as exc:  # noqa: BLE001
                span.record_error(exc)
                return CallModelOutput(error=ErrorInfo.from_exc(exc))
            span.set_output(response.message.text)
            span.set_attributes(
                model=response.model or payload.request.params.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=str(response.finish_reason),
            )

        elapsed = int((services.clock.now() - started).total_seconds() * 1000)
        return CallModelOutput(response=response, latency_ms=elapsed)

    @activity.defn(name="invoke_tool")
    async def invoke_tool(self, payload: InvokeToolInput) -> ToolResult:
        services = self._services
        call = payload.call
        spec = services.registry.spec(call.name)
        if spec is None:
            return _error(
                call,
                ToolStatus.NOT_FOUND,
                f"unknown tool '{call.name}'. Available tools: "
                f"{', '.join(services.registry.names())}",
            )

        missing = payload.permissions.missing(spec.required_permissions)
        if missing:
            return _error(
                call, ToolStatus.DENIED, f"missing permission(s): {', '.join(sorted(missing))}"
            )

        ctx = ToolContext(
            run_id=payload.run_id,
            agent_name=payload.agent_name,
            permissions=payload.permissions,
            depth=payload.depth,
            workspace_id=payload.workspace_id,
        )

        async with services.tracer.span(
            f"tool.{call.name}",
            kind=SpanKind.TOOL,
            parent=payload.trace,
            input=dict(call.arguments),
            attributes={"tool": call.name, "call_id": str(call.id)},
        ) as span:
            try:
                result = await services.registry.get(call.name).invoke(call, ctx)
            except Exception as exc:  # noqa: BLE001 - a tool must never kill a run
                span.record_error(exc)
                result = _error(call, ToolStatus.ERROR, str(exc))
            span.set_output(result.content)

        return await _spill(services, payload, result)

    @activity.defn(name="harvest_workspace")
    async def harvest_workspace(self, payload: HarvestWorkspaceInput) -> list[ObjectRef]:
        """Copy a run's workspace into object storage before it is discarded."""
        archiver = WorkspaceArchiver(self._services.sandboxes, self._services.objects)
        refs = await archiver.harvest(run_id=payload.run_id, workspace_id=payload.workspace_id)
        if refs:
            logger.info("stored %d artifact(s) for run %s", len(refs), payload.run_id)
        return list(refs)

    @activity.defn(name="release_workspace")
    async def release_workspace(self, payload: ReleaseWorkspaceInput) -> None:
        """End a run's isolated environment once the run is over."""
        sandboxes = self._services.sandboxes
        if sandboxes is None:
            return
        try:
            await sandboxes.release(payload.workspace_id)
        except Exception as exc:  # noqa: BLE001 - cleanup must not fail a run
            logger.warning("could not release workspace %s: %s", payload.workspace_id, exc)

    @activity.defn(name="publish_events")
    async def publish_events(self, payload: PublishEventsInput) -> None:
        from pydantic import TypeAdapter

        from skynet.domain.events import RunEvent

        adapter = TypeAdapter(RunEvent)
        for raw in payload.events:
            try:
                await self._services.sink.publish(adapter.validate_python(raw))
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to publish event: %s", exc)


def _error(call: ToolCall, status: ToolStatus, message: str) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        name=call.name,
        status=status,
        content=f"ERROR: {message}",
        error=ErrorInfo.of(str(status), message),
    )


async def _spill(
    services: RuntimeServices, payload: InvokeToolInput, result: ToolResult
) -> ToolResult:
    """Keep oversized output out of the workflow history and the context window."""
    if len(result.content) <= payload.max_output_chars:
        return result
    key = f"runs/{payload.run_id}/tool-output/{result.call_id}.txt"
    try:
        ref = await services.objects.put(key, result.content.encode(), content_type="text/plain")
    except Exception:  # noqa: BLE001
        ref = None
    head = result.content[: payload.max_output_chars]
    note = f"\n\n[output truncated at {payload.max_output_chars} chars"
    note += f"; full output stored at {ref.uri}]" if ref else "]"
    return result.model_copy(
        update={
            "content": head + note,
            "truncated": True,
            "artifacts": (*result.artifacts, ref) if ref else result.artifacts,
        }
    )


__all__ = [
    "AgentActivities",
    "CallModelInput",
    "CallModelOutput",
    "HarvestWorkspaceInput",
    "InvokeToolInput",
    "PublishEventsInput",
    "RecallMemoryInput",
    "ReleaseWorkspaceInput",
    "ResolvePromptInput",
]
