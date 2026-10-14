"""What the pure policy asks the world to do next.

A ``Decision`` is an *intent*, never an action: it contains no I/O and no
results. The ``EffectExecutor`` turns an intent into events. This split is what
lets the identical agent loop run in-process and inside a Temporal workflow.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from skynet.domain.errors import ErrorInfo
from skynet.domain.memory import ConsolidationMode, MemoryQuery, MemoryScope
from skynet.domain.messages import ToolCall
from skynet.domain.models import ModelParams, ModelRequest
from skynet.domain.prompts import PromptRef
from skynet.domain.task import AgentResult


class DecisionBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class ResolvePrompt(DecisionBase):
    kind: Literal["resolve_prompt"] = "resolve_prompt"
    ref: PromptRef
    variables: dict[str, str] = {}


class RecallMemory(DecisionBase):
    """One retrieval across every scope the run may read.

    Carries the whole ``MemoryQuery`` rather than a string and a limit, so the
    scopes an executor searches are the scopes the policy authorised -- there is
    nowhere left for a second opinion about which namespace to read.
    """

    kind: Literal["recall_memory"] = "recall_memory"
    query: MemoryQuery


class ConsolidateMemory(DecisionBase):
    """Ask a model to turn a source into memory operations, then apply them.

    Extraction and reflection are one decision because they are the same shape:
    select a deterministic source, ask for ``MemoryOperation``s, apply them
    atomically, record what happened. Only the source and the prompt differ, and
    both follow from the policy -- which is where the visibility requirement is
    actually satisfied. A second variant would cost an event, an activity, a
    reducer case and a payload model to express a boolean.
    """

    kind: Literal["consolidate_memory"] = "consolidate_memory"
    mode: ConsolidationMode
    scope: MemoryScope
    from_index: int = 0
    through_index: int = 0
    """Doubles as the watermark: ``apply`` advances to it even on failure."""
    retrieval: MemoryQuery | None = None
    """Existing memories to reconcile against, so contradictions are visible."""
    params: ModelParams
    idempotency_key: str
    """Written record ids derive from this, which is what makes a retry safe."""


class CompactHistory(DecisionBase):
    """Fold the head of the conversation into one summary note.

    The request is built purely by the policy, so this needs no new activity:
    the executor calls the model it already knows how to call.
    """

    kind: Literal["compact_history"] = "compact_history"
    request: ModelRequest
    through_index: int
    keep_last: int = 0
    reason: str = ""


class CallModel(DecisionBase):
    kind: Literal["call_model"] = "call_model"
    request: ModelRequest


class InvokeTools(DecisionBase):
    """Invoke one or more capabilities.

    A tuple rather than a single call, so parallel tool execution and parallel
    sub-agents require no change to this shape -- only a fan-out in the executor.
    """

    kind: Literal["invoke_tools"] = "invoke_tools"
    calls: tuple[ToolCall, ...]


class RequestApproval(DecisionBase):
    kind: Literal["request_approval"] = "request_approval"
    calls: tuple[ToolCall, ...]
    reason: str = ""


class Complete(DecisionBase):
    kind: Literal["complete"] = "complete"
    result: AgentResult


class Fail(DecisionBase):
    kind: Literal["fail"] = "fail"
    error: ErrorInfo


Decision = Annotated[
    ResolvePrompt
    | RecallMemory
    | ConsolidateMemory
    | CompactHistory
    | CallModel
    | InvokeTools
    | RequestApproval
    | Complete
    | Fail,
    Field(discriminator="kind"),
]

__all__ = [
    "CallModel",
    "CompactHistory",
    "Complete",
    "ConsolidateMemory",
    "Decision",
    "DecisionBase",
    "Fail",
    "InvokeTools",
    "RecallMemory",
    "RequestApproval",
    "ResolvePrompt",
]
