"""Construction helpers for tests and demos.

Shipped in the package so that adapter test-suites and examples build runs the
same way the core suite does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from skynet.application.registry import InMemoryToolRegistry
from skynet.application.services import RuntimeServices
from skynet.domain.agent import AgentContext, AgentSpec
from skynet.domain.ids import CallId, RunId, TaskId
from skynet.domain.messages import ToolCall, assistant
from skynet.domain.models import FinishReason, ModelResponse, Usage
from skynet.domain.permissions import Permission, PermissionSet
from skynet.domain.prompts import PromptRef
from skynet.domain.run import AgentRun, new_run
from skynet.domain.task import Task
from skynet.testing.fakes import (
    AutoApproveGate,
    FixedClock,
    InMemoryMemoryStore,
    InMemoryObjectStore,
    InMemoryRunStore,
    RecordingEventSink,
    RecordingTracer,
    ScriptedModelProvider,
    SequentialIdGenerator,
    StaticAgentCatalog,
    StaticPromptProvider,
)

DEFAULT_PROMPT = PromptRef(name="skynet/agent")
# Permissive by default: a test that wants to exercise denial narrows the set
# explicitly, which reads better than a test failing for a reason it never
# mentioned. Derived from Permission so a new grant cannot be forgotten here.
DEFAULT_GRANTS = tuple(sorted(Permission.all()))


def make_spec(name: str = "demo", **kwargs: Any) -> AgentSpec:
    kwargs.setdefault("prompt", DEFAULT_PROMPT)
    kwargs.setdefault("default_permissions", PermissionSet.of(*DEFAULT_GRANTS))
    return AgentSpec(name=name, **kwargs)


def make_context(run_id: str = "run-1", **kwargs: Any) -> AgentContext:
    kwargs.setdefault("permissions", PermissionSet.of(*DEFAULT_GRANTS))
    kwargs.setdefault("agent_name", "demo")
    return AgentContext(run_id=RunId(run_id), **kwargs)


def make_task(objective: str = "do the thing", **kwargs: Any) -> Task:
    kwargs.setdefault("id", TaskId("task-1"))
    return Task(objective=objective, **kwargs)


def make_run(spec: AgentSpec | None = None, **kwargs: Any) -> AgentRun:
    return new_run(
        spec=spec or make_spec(),
        task=kwargs.pop("task", None) or make_task(),
        context=kwargs.pop("context", None) or make_context(),
        **kwargs,
    )


def call(name: str, call_id: str = "call-1", **arguments: Any) -> ToolCall:
    return ToolCall(id=CallId(call_id), name=name, arguments=arguments)


def says(content: str) -> ModelResponse:
    """A plain assistant turn: under native tool calling this means 'done'."""
    return ModelResponse(
        message=assistant(content),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def calls(*tool_calls: ToolCall, content: str = "") -> ModelResponse:
    return ModelResponse(
        message=assistant(content, tool_calls=tool_calls),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def build_services(
    *,
    responses: Sequence[ModelResponse] = (),
    tools: Sequence[Any] = (),
    agents: Sequence[AgentSpec] = (),
    **overrides: Any,
) -> RuntimeServices:
    """A fully wired runtime with every port backed by an in-memory fake."""
    defaults: dict[str, Any] = {
        "model": ScriptedModelProvider(responses),
        "prompts": StaticPromptProvider({"skynet/agent": "You are {{agent_name}}."}),
        "registry": InMemoryToolRegistry(tools),
        "memory": InMemoryMemoryStore(),
        "objects": InMemoryObjectStore(),
        "tracer": RecordingTracer(),
        "sink": RecordingEventSink(),
        "clock": FixedClock(),
        "ids": SequentialIdGenerator(),
        "approvals": AutoApproveGate(),
        "runs": InMemoryRunStore(),
        "agents": StaticAgentCatalog(agents),
    }
    defaults.update(overrides)
    return RuntimeServices(**defaults)


__all__ = [
    "DEFAULT_GRANTS",
    "DEFAULT_PROMPT",
    "build_services",
    "call",
    "calls",
    "make_context",
    "make_run",
    "make_spec",
    "make_task",
    "says",
]
