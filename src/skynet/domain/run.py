"""Run state, the explicit state machine, and the pure reducer.

``AgentRun`` is the complete state of one agent execution. It changes in exactly
one way: ``apply(run, event)``. That single rule is what makes the runtime
durable -- state can always be rebuilt with ``replay(initial, events)``, whether
the events come from memory, a database, or a Temporal workflow history.

Everything in this module is pure: no clock, no I/O, no randomness.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from functools import reduce
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from skynet.domain.agent import AgentContext, AgentSpec, FinishMode
from skynet.domain.errors import ErrorInfo, InvalidTransition
from skynet.domain.events import (
    ApprovalRequested,
    ApprovalResolved,
    ChildRunStarted,
    HistoryCompacted,
    MemoryConsolidated,
    MemoryRecalled,
    ModelFailed,
    ModelResponded,
    PromptResolved,
    RunCancelled,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
    ToolCompleted,
)
from skynet.domain.ids import CallId, RunId
from skynet.domain.memory import ConsolidationMode, MemoryHit, merge_hits
from skynet.domain.messages import Message, ToolCall, system, tool_result
from skynet.domain.models import FinishReason, Usage
from skynet.domain.prompts import RenderedPrompt
from skynet.domain.task import AgentResult, Task
from skynet.domain.tools import ToolResult


class AgentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_suspended(self) -> bool:
        """Suspended runs exit the loop and wait for an external event."""
        return self in _SUSPENDED


_TERMINAL = frozenset({AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED})
_SUSPENDED = frozenset({AgentStatus.WAITING_APPROVAL, AgentStatus.PAUSED, AgentStatus.WAITING})

_ALLOWED: Mapping[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.CREATED: frozenset(
        {AgentStatus.RUNNING, AgentStatus.CANCELLED, AgentStatus.FAILED}
    ),
    AgentStatus.RUNNING: frozenset(
        {
            AgentStatus.WAITING_APPROVAL,
            AgentStatus.PAUSED,
            AgentStatus.WAITING,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.WAITING_APPROVAL: frozenset(
        {AgentStatus.RUNNING, AgentStatus.CANCELLED, AgentStatus.FAILED}
    ),
    AgentStatus.PAUSED: frozenset({AgentStatus.RUNNING, AgentStatus.CANCELLED}),
    AgentStatus.WAITING: frozenset(
        {AgentStatus.RUNNING, AgentStatus.CANCELLED, AgentStatus.FAILED}
    ),
    AgentStatus.COMPLETED: frozenset(),
    AgentStatus.FAILED: frozenset(),
    AgentStatus.CANCELLED: frozenset(),
}


class StepRecord(BaseModel):
    """One model turn plus the observations it produced. A reporting view."""

    model_config = ConfigDict(frozen=True)

    index: int
    assistant: Message
    tool_results: tuple[ToolResult, ...] = ()
    usage: Usage = Usage()
    finish_reason: FinishReason = FinishReason.STOP


class AgentRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: RunId
    spec: AgentSpec
    task: Task
    context: AgentContext

    finish_mode: FinishMode = FinishMode.NO_TOOL_CALLS
    status: AgentStatus = AgentStatus.CREATED
    seq: int = 0
    iteration: int = 0

    prompt: RenderedPrompt | None = None
    messages: tuple[Message, ...] = ()
    steps: tuple[StepRecord, ...] = ()
    pending_calls: tuple[ToolCall, ...] = ()
    approvals: dict[CallId, bool] = {}
    memories: tuple[MemoryHit, ...] = ()
    recalled_queries: tuple[str, ...] = ()
    """Signatures of retrievals already issued -- the recall watermark.

    A set of signatures rather than a boolean, because recall is no longer a
    once-per-run event: a second query with a different text or scope is a
    different retrieval and must be allowed to fire.
    """

    consolidated_through: int = 0
    """How far into ``messages`` consolidation has read. Strictly increasing."""
    consolidation_count: int = 0
    reflected_at_iteration: int = -1
    compactions: int = 0
    compacted_messages: int = 0

    usage: Usage = Usage()
    tool_call_count: int = 0
    model_failures: int = 0
    parse_retries: int = 0

    last_error: ErrorInfo | None = None
    result: AgentResult | None = None
    child_runs: tuple[tuple[CallId, RunId], ...] = ()


def _transition(run: AgentRun, target: AgentStatus, **updates: object) -> AgentRun:
    if target is not run.status and target not in _ALLOWED[run.status]:
        raise InvalidTransition(f"cannot move run {run.run_id} from {run.status} to {target}")
    return run.model_copy(update={"status": target, **updates})


def _require_active(run: AgentRun, event: RunEvent) -> None:
    if run.status.is_terminal:
        raise InvalidTransition(
            f"cannot apply {type(event).__name__} to run {run.run_id} in state {run.status}"
        )


def apply(run: AgentRun, event: RunEvent) -> AgentRun:
    """Fold one event into the run. Total, pure and deterministic."""
    match event:
        case RunStarted():
            return _transition(run, AgentStatus.RUNNING, seq=event.seq)

        case PromptResolved(prompt=prompt):
            _require_active(run, event)
            return run.model_copy(update={"prompt": prompt, "seq": event.seq})

        case MemoryRecalled(query=query, hits=hits):
            _require_active(run, event)
            # Accumulate rather than replace. Recall can fire more than once,
            # and overwriting would make a later, narrower query silently
            # discard everything the first one found.
            signature = query.signature()
            issued = run.recalled_queries
            return run.model_copy(
                update={
                    "memories": merge_hits(run.memories, hits),
                    "recalled_queries": (issued if signature in issued else (*issued, signature)),
                    "seq": event.seq,
                }
            )

        case MemoryConsolidated(mode=mode, through_index=through):
            _require_active(run, event)
            # The watermark advances even when nothing was written and even when
            # the pass failed -- that is the whole termination argument for
            # "consolidate before completing". It must never touch `messages`:
            # the completion check reads those, so mutating them here would flip
            # the verdict and leave the run oscillating.
            updates: dict[str, object] = {
                "consolidated_through": max(run.consolidated_through, through),
                "consolidation_count": run.consolidation_count + 1,
                "seq": event.seq,
            }
            if mode is ConsolidationMode.REFLECT:
                updates["reflected_at_iteration"] = run.iteration
            return run.model_copy(update=updates)

        case HistoryCompacted(through_index=through, summary=summary):
            _require_active(run, event)
            # A failed summarisation counts but folds nothing. Replacing real
            # history with "(summary unavailable)" would destroy context to save
            # tokens the trimmer can reclaim anyway -- and the count is what
            # guarantees this terminates, not the fold.
            replaced = max(0, min(through, len(run.messages))) if summary else 0
            return run.model_copy(
                update={
                    "messages": _fold_history(run.messages, replaced, summary),
                    "compactions": run.compactions + 1,
                    "compacted_messages": run.compacted_messages + replaced,
                    "seq": event.seq,
                }
            )

        case ModelResponded(response=response):
            _require_active(run, event)
            message = response.message
            parse_failed = response.finish_reason is FinishReason.PARSE_ERROR
            step = StepRecord(
                index=run.iteration,
                assistant=message,
                usage=response.usage,
                finish_reason=response.finish_reason,
            )
            return run.model_copy(
                update={
                    "messages": (*run.messages, message),
                    "steps": (*run.steps, step),
                    "pending_calls": message.tool_calls,
                    "iteration": run.iteration + 1,
                    "usage": run.usage + response.usage,
                    "model_failures": 0,
                    "parse_retries": run.parse_retries + 1 if parse_failed else 0,
                    "seq": event.seq,
                }
            )

        case ModelFailed(error=error):
            _require_active(run, event)
            return run.model_copy(
                update={
                    "model_failures": run.model_failures + 1,
                    "last_error": error,
                    "seq": event.seq,
                }
            )

        case ChildRunStarted(call_id=call_id, child_run_id=child_run_id):
            _require_active(run, event)
            return run.model_copy(
                update={
                    "child_runs": (*run.child_runs, (call_id, child_run_id)),
                    "seq": event.seq,
                }
            )

        case ToolCompleted(result=result):
            _require_active(run, event)
            return run.model_copy(
                update={
                    "messages": (
                        *run.messages,
                        tool_result(result.call_id, result.name, result.content),
                    ),
                    "steps": _record_tool_result(run.steps, result),
                    "pending_calls": tuple(c for c in run.pending_calls if c.id != result.call_id),
                    "tool_call_count": run.tool_call_count + 1,
                    "seq": event.seq,
                }
            )

        case ApprovalRequested():
            _require_active(run, event)
            return _transition(run, AgentStatus.WAITING_APPROVAL, seq=event.seq)

        case ApprovalResolved(call_id=call_id, granted=granted):
            _require_active(run, event)
            resolved = run.model_copy(
                update={
                    "approvals": {**run.approvals, call_id: granted},
                    "seq": event.seq,
                }
            )
            # Resolving while RUNNING is legal (pre-approval), which makes the
            # resume path idempotent.
            if resolved.status is AgentStatus.WAITING_APPROVAL:
                return _transition(resolved, AgentStatus.RUNNING)
            return resolved

        case RunPaused():
            return _transition(run, AgentStatus.PAUSED, seq=event.seq)

        case RunResumed():
            return _transition(run, AgentStatus.RUNNING, seq=event.seq)

        case RunCompleted(result=result):
            return _transition(run, AgentStatus.COMPLETED, seq=event.seq, result=result)

        case RunFailed(error=error):
            return _transition(run, AgentStatus.FAILED, seq=event.seq, last_error=error)

        case RunCancelled():
            return _transition(run, AgentStatus.CANCELLED, seq=event.seq)

        case _:  # pragma: no cover - exhaustiveness guard
            assert_never(event)


COMPACTION_NOTE_PREFIX = "Summary of the earlier conversation:"


def _fold_history(
    messages: tuple[Message, ...], through_index: int, summary: str
) -> tuple[Message, ...]:
    """Replace ``messages[:through_index]`` with one system note.

    Exactly one note ever exists: the previous one sits at index 0 and is
    therefore inside the next fold. ``steps`` is deliberately untouched --
    compaction is lossy only for the model's view of the run, never for the log.
    """
    through = max(0, min(through_index, len(messages)))
    if through == 0:
        return messages
    tail = list(messages[through:])
    # A tool result whose requesting assistant turn was just folded away is an
    # orphan; strict providers reject it outright.
    while tail and tail[0].is_tool_result:
        tail.pop(0)
    note = system(f"{COMPACTION_NOTE_PREFIX}\n{summary.strip()}")
    return (note, *tail)


def _record_tool_result(
    steps: tuple[StepRecord, ...], result: ToolResult
) -> tuple[StepRecord, ...]:
    if not steps:
        return steps
    last = steps[-1]
    updated = last.model_copy(update={"tool_results": (*last.tool_results, result)})
    return (*steps[:-1], updated)


def replay(initial: AgentRun, events: Iterable[RunEvent]) -> AgentRun:
    """Rebuild a run from its event log. The recovery path, and a test tool."""
    return reduce(apply, events, initial)


def new_run(
    *,
    spec: AgentSpec,
    task: Task,
    context: AgentContext,
    finish_mode: FinishMode = FinishMode.NO_TOOL_CALLS,
) -> AgentRun:
    return AgentRun(
        run_id=context.run_id,
        spec=spec,
        task=task,
        context=context,
        finish_mode=finish_mode,
    )


__all__ = [
    "COMPACTION_NOTE_PREFIX",
    "AgentRun",
    "AgentStatus",
    "StepRecord",
    "apply",
    "new_run",
    "replay",
]
