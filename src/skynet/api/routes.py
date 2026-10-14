"""HTTP routes.

Deliberately thin: each handler translates a request into a port call and a
result back into a DTO. There is no agent logic here, which is why adding a
gRPC or WebSocket front end later would not touch the runtime.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from skynet.api.schemas import (
    AgentView,
    ApprovalRequest,
    CancelRequest,
    EventView,
    RunDetail,
    RunSummary,
    SubmitRunRequest,
    SubmitRunResponse,
    ToolView,
)
from skynet.domain.agent import AgentContext, TraceContext
from skynet.domain.errors import AgentNotFound, InvalidTransition
from skynet.domain.events import RunStarted
from skynet.domain.ids import CallId, RunId, TaskId, TraceId
from skynet.domain.run import AgentStatus, new_run, replay
from skynet.domain.task import Task
from skynet.ports.agents import RunRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def runtime(request: Request):
    return request.app.state.runtime


@router.get("/health", tags=["system"])
async def health(request: Request) -> dict[str, Any]:
    rt = runtime(request)
    return {
        "status": "ok",
        "engine": rt.settings.execution.engine,
        "model": f"{rt.settings.model.provider}/{rt.settings.model.name}",
        "sandbox": rt.settings.sandbox.backend,
        "tracing": rt.settings.tracing.backend,
        "storage": rt.settings.storage.backend,
    }


@router.get("/v1/agents", response_model=list[AgentView], tags=["catalog"])
async def list_agents(request: Request) -> list[AgentView]:
    return [
        AgentView(
            name=spec.name,
            description=spec.description,
            tools=list(spec.tools),
            sub_agents=list(spec.sub_agents),
            permissions=sorted(spec.default_permissions.grants),
        )
        for spec in runtime(request).services.agents.all()
    ]


@router.get("/v1/tools", response_model=list[ToolView], tags=["catalog"])
async def list_tools(request: Request) -> list[ToolView]:
    registry = runtime(request).services.registry
    views = []
    for name in registry.names():
        spec = registry.spec(name)
        if spec is None:
            continue
        views.append(
            ToolView(
                name=spec.name,
                description=spec.description,
                kind=str(spec.kind),
                required_permissions=sorted(spec.required_permissions),
                requires_approval=spec.requires_approval,
                parameters=spec.parameters,
            )
        )
    return views


@router.post("/v1/runs", response_model=SubmitRunResponse, status_code=202, tags=["runs"])
async def submit_run(request: Request, body: SubmitRunRequest) -> SubmitRunResponse:
    rt = runtime(request)
    try:
        spec = rt.services.agents.get(body.agent)
    except AgentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    ids = rt.services.ids
    run_id = RunId(ids.new_id("run"))
    run_request = RunRequest(
        spec=spec,
        task=Task(
            id=TaskId(ids.new_id("task")),
            objective=body.objective,
            input=body.input,
            metadata=body.metadata,
        ),
        context=AgentContext(
            run_id=run_id,
            agent_name=spec.name,
            permissions=spec.default_permissions,
            # Minted here, before execution, so activities in other processes
            # can attach their spans to the same trace.
            trace=TraceContext(
                trace_id=TraceId(ids.new_trace_id()),
                baggage={"trace_name": f"agent.{spec.name}", "session_id": str(run_id)},
            ),
        ),
    )

    rt.requests[run_id] = run_request
    rt.handles[run_id] = await rt.runner.start(run_request)
    return SubmitRunResponse(run_id=str(run_id), agent=spec.name, status=AgentStatus.RUNNING)


@router.get("/v1/runs", response_model=list[RunSummary], tags=["runs"])
async def list_runs(request: Request, limit: int = 50) -> list[RunSummary]:
    runs = await runtime(request).services.runs.list_runs(limit)
    return [RunSummary.of(run) for run in runs]


@router.get("/v1/runs/{run_id}", response_model=RunDetail, tags=["runs"])
async def get_run(request: Request, run_id: str) -> RunDetail:
    run = await _load(runtime(request), RunId(run_id))
    return RunDetail.of(run)


@router.get("/v1/runs/{run_id}/events", response_model=list[EventView], tags=["runs"])
async def get_events(request: Request, run_id: str) -> list[EventView]:
    events = runtime(request).events.history(RunId(run_id))
    if not events:
        raise HTTPException(status_code=404, detail=f"no events for run {run_id}")
    return [_event_view(event) for event in events]


@router.get("/v1/runs/{run_id}/stream", tags=["runs"])
async def stream_events(request: Request, run_id: str) -> StreamingResponse:
    """Server-sent events.

    The streaming seam: the event log already exists, so streaming is a
    projection of it rather than a parallel mechanism.
    """
    rt = runtime(request)

    async def publish():
        async for event in rt.events.subscribe(RunId(run_id)):
            yield f"event: {event.kind}\ndata: {_event_view(event).model_dump_json()}\n\n"
            if await request.is_disconnected():
                return
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        publish(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/v1/runs/{run_id}/approve", status_code=204, tags=["runs"])
async def approve(request: Request, run_id: str, body: ApprovalRequest) -> None:
    handle = _handle(runtime(request), RunId(run_id))
    await handle.approve(CallId(body.call_id), body.granted, body.actor)


@router.post("/v1/runs/{run_id}/cancel", status_code=204, tags=["runs"])
async def cancel(request: Request, run_id: str, body: CancelRequest) -> None:
    handle = _handle(runtime(request), RunId(run_id))
    await handle.cancel(body.reason)


@router.get("/v1/runs/{run_id}/artifacts", tags=["runs"])
async def list_artifacts(request: Request, run_id: str) -> list[dict[str, Any]]:
    refs = await runtime(request).services.objects.list(f"runs/{run_id}/")
    return [
        {"uri": ref.uri, "key": ref.key, "size": ref.size, "content_type": ref.content_type}
        for ref in refs
    ]


# -- helpers ------------------------------------------------------------------


def _handle(rt, run_id: RunId):
    handle = rt.handles.get(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return handle


async def _load(rt, run_id: RunId):
    """Current state of a run, from whichever source can answer.

    The event-log path is the interesting one: it needs no live handle and no
    session state, so a restarted API can still describe runs a worker executed
    -- which is the same reconstruction Temporal itself performs on replay.
    """
    # The event log is the richest source -- it reconstructs the full step and
    # tool history -- and it needs no session state, so it is tried first.
    if (rebuilt := _rebuild_from_events(rt, run_id)) is not None:
        return rebuilt

    handle = rt.handles.get(run_id)
    if handle is not None:
        return await handle.snapshot()

    run = await rt.services.runs.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
    return run


def _rebuild_from_events(rt, run_id: RunId):
    events = rt.events.history(run_id)
    if not events or not isinstance(events[0], RunStarted):
        return None
    opening = events[0]
    initial = new_run(spec=opening.spec, task=opening.task, context=opening.context)
    try:
        return replay(initial, events)
    except InvalidTransition:
        # A torn or partially written log should degrade, not 500.
        logger.warning("could not fully replay run %s", run_id)
        return None


def _event_view(event) -> EventView:
    payload = event.model_dump(mode="json")
    for key in ("kind", "seq", "at", "run_id"):
        payload.pop(key, None)
    return EventView(kind=event.kind, seq=event.seq, at=event.at, detail=payload)


__all__ = ["router"]
