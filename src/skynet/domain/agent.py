"""Agent configuration and per-run context.

``AgentSpec`` is the static definition of an agent (which prompt, which model,
which tools, which budgets). ``AgentContext`` is the per-run envelope: identity,
authority, position in the agent hierarchy, and trace linkage.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from skynet.domain.ids import RunId, SpanId, TraceId
from skynet.domain.memory import MemoryKind, ScopeLevel
from skynet.domain.models import ModelParams
from skynet.domain.permissions import PermissionSet
from skynet.domain.prompts import PromptRef


class FinishMode(StrEnum):
    """How the runtime decides an agent is done."""

    NO_TOOL_CALLS = "no_tool_calls"
    """Native tool calling: an assistant turn without tool calls means done."""
    FINAL_ANSWER_TOOL = "final_answer"
    """Emulated tool calling: completion must be an explicit `final_answer` call,
    because weaker models chat rather than stop."""


class ResourceLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iterations: int = 12
    max_tool_calls: int = 40
    max_total_tokens: int | None = None
    max_model_failures: int = 2
    max_parse_retries: int = 2
    max_depth: int = 3
    """Sub-agent nesting depth."""
    tool_timeout_s: float = 60.0

    max_context_chars: int = 24_000
    """One budget, read by both the compaction trigger and the history trimmer.

    Two numbers here would let the trimmer truncate history the summariser was
    about to fold, or let compaction fire on a request that already fits.
    """
    max_compactions: int = 4
    max_consolidations: int = 3


class MemoryConfig(BaseModel):
    """Per-agent memory policy.

    Private by default: an agent reads its own scopes unless a grant says
    otherwise. Sharing is opt-in because the failure mode of the reverse default
    -- one agent quietly reasoning from another's half-formed notes -- is
    invisible in the event log until it produces a wrong answer.
    """

    model_config = ConfigDict(frozen=True)

    auto_recall: bool = False
    recall_limit: int = 8
    recall_kinds: tuple[MemoryKind, ...] = ()
    """Empty means every kind."""

    share_project: bool = False
    share_user: bool = False
    project: str = ""
    user: str = ""
    """Default owners. ``AgentContext.variables`` overrides them, so one agent
    definition serves many projects."""

    write_level: ScopeLevel = ScopeLevel.AGENT
    """Where ``memory_write`` and consolidation land."""

    consolidate: bool = False
    reflect_every: int = 0
    """Iterations between reflections. 0 disables it."""
    default_ttl_hours: float | None = None
    model: ModelParams | None = None
    """A cheaper model for memory work. ``None`` reuses the agent's own."""


class TraceContext(BaseModel):
    """Trace linkage carried through the run and inherited by sub-agents.

    Making this a domain concept (rather than something the tracer adapter owns)
    is what keeps the agent -> model -> tool -> sub-agent hierarchy intact when
    the observability backend is swapped.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: TraceId | None = None
    parent_span_id: SpanId | None = None
    baggage: dict[str, str] = {}

    @property
    def is_linked(self) -> bool:
        return self.trace_id is not None

    def child(self, span_id: SpanId | None = None) -> TraceContext:
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=span_id or self.parent_span_id,
            baggage=self.baggage,
        )


class AgentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    prompt: PromptRef
    instructions: str = ""
    model: ModelParams = ModelParams(model="qwen3.5:9b")
    tools: tuple[str, ...] = ()
    """Explicit allowlist. No wildcards: authority must be readable."""
    sub_agents: tuple[str, ...] = ()
    """Agents this one may delegate to."""
    limits: ResourceLimits = ResourceLimits()
    output_schema: dict[str, Any] | None = None
    memory: MemoryConfig = MemoryConfig()
    default_permissions: PermissionSet = PermissionSet()
    description: str = ""


class AgentContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: RunId
    agent_name: str
    permissions: PermissionSet = PermissionSet()
    depth: int = 0
    parent_run_id: RunId | None = None
    trace: TraceContext = TraceContext()
    workspace_id: str | None = None
    deadline: datetime | None = None
    variables: dict[str, str] = {}


__all__ = [
    "AgentContext",
    "AgentSpec",
    "FinishMode",
    "MemoryConfig",
    "ResourceLimits",
    "TraceContext",
]
