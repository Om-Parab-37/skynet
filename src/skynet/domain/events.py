"""The run event log -- the canonical record of what happened.

Everything else is derived from this: ``AgentRun`` is a fold over events, the
API's stream is a projection of them, and a crashed run is recovered by
replaying them. Events are append-only and totally ordered by ``seq``.

``seq`` and ``at`` are stamped by the ``EffectExecutor``, never by the policy,
because both are non-deterministic and must come from the host (the system
clock in-process, ``workflow.now()`` under Temporal).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from skynet.domain.agent import AgentContext, AgentSpec
from skynet.domain.errors import ErrorInfo
from skynet.domain.ids import CallId, MemoryId, RunId
from skynet.domain.memory import (
    ConsolidationMode,
    MemoryHit,
    MemoryOperation,
    MemoryQuery,
    MemoryScope,
)
from skynet.domain.models import ModelResponse
from skynet.domain.prompts import RenderedPrompt
from skynet.domain.task import AgentResult, Task
from skynet.domain.tools import ToolResult


class RunEventBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: RunId
    seq: int = 0
    at: datetime | None = None


class RunStarted(RunEventBase):
    kind: Literal["run.started"] = "run.started"
    spec: AgentSpec
    task: Task
    context: AgentContext


class PromptResolved(RunEventBase):
    kind: Literal["prompt.resolved"] = "prompt.resolved"
    prompt: RenderedPrompt


class MemoryRecalled(RunEventBase):
    kind: Literal["memory.recalled"] = "memory.recalled"
    query: MemoryQuery
    hits: tuple[MemoryHit, ...] = ()
    as_of: datetime | None = None
    """The instant the store was read at. Carried so a replay ranks identically."""


class MemoryConsolidated(RunEventBase):
    """What one consolidation pass decided and what it managed to write.

    Carries ``error`` rather than raising: memory must never fail a run, and a
    consolidation that achieved nothing still has to advance ``through_index``
    or the policy would ask for it again forever.
    """

    kind: Literal["memory.consolidated"] = "memory.consolidated"
    mode: ConsolidationMode
    scope: MemoryScope
    through_index: int = 0
    operations: tuple[MemoryOperation, ...] = ()
    written: tuple[MemoryId, ...] = ()
    invalidated: tuple[MemoryId, ...] = ()
    error: ErrorInfo | None = None


class HistoryCompacted(RunEventBase):
    """The head of the conversation, replaced by a summary.

    ``summary`` travels in the event, so ``replay`` reproduces ``messages``
    byte-identically without calling a model again.
    """

    kind: Literal["history.compacted"] = "history.compacted"
    through_index: int = 0
    summary: str = ""
    replaced: int = 0
    error: ErrorInfo | None = None


class ModelResponded(RunEventBase):
    kind: Literal["model.responded"] = "model.responded"
    response: ModelResponse
    latency_ms: int = 0


class ModelFailed(RunEventBase):
    kind: Literal["model.failed"] = "model.failed"
    error: ErrorInfo


class ChildRunStarted(RunEventBase):
    kind: Literal["child.started"] = "child.started"
    call_id: CallId
    child_run_id: RunId
    agent: str


class ToolCompleted(RunEventBase):
    kind: Literal["tool.completed"] = "tool.completed"
    result: ToolResult


class ApprovalRequested(RunEventBase):
    kind: Literal["approval.requested"] = "approval.requested"
    call_ids: tuple[CallId, ...]
    reason: str = ""


class ApprovalResolved(RunEventBase):
    kind: Literal["approval.resolved"] = "approval.resolved"
    call_id: CallId
    granted: bool
    actor: str = "system"


class RunPaused(RunEventBase):
    kind: Literal["run.paused"] = "run.paused"
    reason: str = ""


class RunResumed(RunEventBase):
    kind: Literal["run.resumed"] = "run.resumed"


class RunCompleted(RunEventBase):
    kind: Literal["run.completed"] = "run.completed"
    result: AgentResult


class RunFailed(RunEventBase):
    kind: Literal["run.failed"] = "run.failed"
    error: ErrorInfo


class RunCancelled(RunEventBase):
    kind: Literal["run.cancelled"] = "run.cancelled"
    reason: str = ""


RunEvent = Annotated[
    RunStarted
    | PromptResolved
    | MemoryRecalled
    | MemoryConsolidated
    | HistoryCompacted
    | ModelResponded
    | ModelFailed
    | ChildRunStarted
    | ToolCompleted
    | ApprovalRequested
    | ApprovalResolved
    | RunPaused
    | RunResumed
    | RunCompleted
    | RunFailed
    | RunCancelled,
    Field(discriminator="kind"),
]

__all__ = [
    "ApprovalRequested",
    "ApprovalResolved",
    "ChildRunStarted",
    "HistoryCompacted",
    "MemoryConsolidated",
    "MemoryRecalled",
    "ModelFailed",
    "ModelResponded",
    "PromptResolved",
    "RunCancelled",
    "RunCompleted",
    "RunEvent",
    "RunEventBase",
    "RunFailed",
    "RunPaused",
    "RunResumed",
    "RunStarted",
    "ToolCompleted",
]
