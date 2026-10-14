"""Turning a conversation, or a pile of memories, into memory operations.

**A pure collaborator, not a port.** A port models an I/O boundary, and
consolidation's only I/O is a model call and some store writes -- both already
ports. A ``MemoryConsolidator`` port would hide an LLM call outside the
``ModelProvider`` seam: untraced by the existing generation span, unswappable by
configuration, and needing a fake of its own. So the *logic* is extracted here,
where it is testable by calling three functions, and the *effects* stay at the
seam that already exists.

Three functions, in the order the executor uses them:

``consolidation_request``  what to ask, given a mode and a deterministic source
``parse_operations``       tolerant reading of the answer -- never raises
``plan_writes``            operations to concrete records, with derived ids
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import TypeAdapter, ValidationError

from skynet.application.history_compaction import render
from skynet.domain.ids import MemoryId, RunId
from skynet.domain.memory import (
    AddMemory,
    ConsolidationMode,
    InvalidateMemory,
    MemoryOperation,
    MemoryRecord,
    MemoryScope,
    UpdateMemory,
)
from skynet.domain.messages import Message, system, user
from skynet.domain.models import ModelParams, ModelRequest, ToolChoice

_OPERATION = TypeAdapter(MemoryOperation)

MAX_OPERATIONS = 12
"""A cap on one pass. A model that returns fifty "facts" has started narrating."""

FORMAT = """Reply with JSON and nothing else, in exactly this shape:

{"operations": [
  {"kind": "add", "content": "...", "memory_kind": "semantic",
   "tags": ["..."], "importance": 0.5},
  {"kind": "update", "target": "<id of an existing memory>", "content": "..."},
  {"kind": "invalidate", "target": "<id of an existing memory>", "reason": "..."}
]}

`memory_kind` is one of: semantic (something believed to be true), episodic
(something that happened), procedural (how something is done here).
`importance` runs from 0.0 to 1.0.
Use `update` when a new fact replaces an existing one, and `invalidate` when an
existing one stopped being true and nothing replaces it. Never invent an id:
`target` must be one of the ids listed under EXISTING MEMORIES.
Return {"operations": []} if there is nothing worth remembering. That is a
perfectly good answer and is much better than padding."""

EXTRACT_INSTRUCTION = f"""You maintain the long-term memory of an AI agent.

Read the transcript below and decide what is worth remembering *beyond this
task*. Remember durable things: facts established, conventions discovered,
credentials and locations, approaches that worked or failed and why.

Do not remember: chit-chat, restatements of the objective, intermediate
arithmetic, or anything that will be false tomorrow. A memory you write will be
shown to the agent in future tasks as though it were true, so writing something
speculative is worse than writing nothing.

Reconcile against the existing memories listed below rather than duplicating
them: if one is now wrong, update or invalidate it.

{FORMAT}"""

REFLECT_INSTRUCTION = f"""You maintain the long-term memory of an AI agent.

Below are memories the agent already holds. Look across them for something that
is not written down in any single one: a pattern, a generalisation, a
contradiction, or a conclusion that only follows from several of them together.

Add only insights that are genuinely new. Restating an existing memory in
different words makes the memory store larger and less useful, which is the
opposite of the point. If two memories contradict each other, invalidate the one
that is now wrong instead of adding a third.

{FORMAT}"""


def consolidation_request(
    mode: ConsolidationMode,
    *,
    objective: str,
    messages: Sequence[Message] = (),
    candidates: Sequence[MemoryRecord] = (),
    params: ModelParams,
) -> ModelRequest:
    """Build the consolidation call.

    The two modes share everything but their source and instruction, which is
    why they are one decision rather than two.
    """
    instruction = REFLECT_INSTRUCTION if mode is ConsolidationMode.REFLECT else EXTRACT_INSTRUCTION
    parts = [f"TASK: {objective}"] if objective else []
    parts.append(f"EXISTING MEMORIES:\n{_render_candidates(candidates)}")
    if mode is not ConsolidationMode.REFLECT:
        transcript = "\n\n".join(render(message) for message in messages)
        parts.append(f"TRANSCRIPT:\n{transcript or '(nothing new)'}")
    return ModelRequest(
        messages=(system(instruction), user("\n\n".join(parts))),
        params=params,
        tools=(),
        tool_choice=ToolChoice.NONE,
    )


def _render_candidates(candidates: Sequence[MemoryRecord]) -> str:
    if not candidates:
        return "(none)"
    return "\n".join(
        f"- [{record.id}] ({record.kind.value}) {record.content}" for record in candidates
    )


def parse_operations(
    text: str,
    *,
    known_ids: Iterable[MemoryId] = (),
    limit: int = MAX_OPERATIONS,
) -> tuple[MemoryOperation, ...]:
    """Read operations out of a model's answer. Never raises.

    Deliberately tolerant in three ways, each for a failure seen in practice:
    code fences around the JSON, prose wrapped around it, and one malformed
    entry in an otherwise good list. A single bad item drops that item, not the
    whole pass -- and a completely unreadable answer is an empty tuple, which
    the executor reports as a consolidation that achieved nothing rather than as
    an exception that would fail a run.

    Operations naming an id that was never offered are dropped: a hallucinated
    target either does nothing or, worse, invalidates something real.
    """
    payload = _extract_json(text)
    if payload is None:
        return ()
    raw = payload.get("operations") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return ()

    allowed = {str(memory_id) for memory_id in known_ids}
    operations: list[MemoryOperation] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        try:
            operation = _OPERATION.validate_python(item)
        except ValidationError:
            continue
        if isinstance(operation, UpdateMemory | InvalidateMemory) and (
            str(operation.target) not in allowed
        ):
            continue
        operations.append(operation)
    return tuple(operations)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | list | None:
    candidates = [match.strip() for match in _FENCE.findall(text)]
    stripped = text.strip()
    candidates.append(stripped)
    for opening, closing in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opening), stripped.rfind(closing)
        if 0 <= start < end:
            candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict | list):
            return parsed
    return None


@dataclass(frozen=True, slots=True)
class ConsolidationPlan:
    """Exactly what to write and what to end, with every id already decided."""

    writes: tuple[MemoryRecord, ...] = ()
    invalidations: tuple[tuple[MemoryId, MemoryId | None], ...] = ()
    """``(target, superseded_by)``. ``superseded_by`` is set only for an update."""
    rejected: int = 0
    """Operations dropped for naming a memory this run may not write to."""


def plan_writes(
    operations: Sequence[MemoryOperation],
    *,
    key: str,
    scope: MemoryScope,
    at: datetime,
    candidates: Sequence[MemoryRecord] = (),
    source_run_id: RunId | None = None,
    default_ttl_hours: float | None = None,
) -> ConsolidationPlan:
    """Turn operations into records, deriving every id from ``key``.

    Ids are ``f"{key}-{index}"``, which is what makes a retry idempotent with no
    ``IdGenerator`` and no bookkeeping: replaying the same decision writes the
    same rows over the top of themselves rather than a second copy.

    An operation may only touch a memory in ``scope``. Retrieval spans several
    scopes so the model can *reason* about shared memories, but a run that may
    not write to a scope must not be able to end a belief in it either --
    otherwise "read-only access" would be read-and-destroy.
    """
    writable = {str(record.id) for record in candidates if record.scope == scope}
    writes: list[MemoryRecord] = []
    invalidations: list[tuple[MemoryId, MemoryId | None]] = []
    rejected = 0

    for index, operation in enumerate(operations):
        if isinstance(operation, InvalidateMemory | UpdateMemory) and (
            str(operation.target) not in writable
        ):
            rejected += 1
            continue

        match operation:
            case InvalidateMemory(target=target):
                invalidations.append((target, None))
            case AddMemory() | UpdateMemory():
                memory_id = MemoryId(f"{key}-{index}")
                writes.append(
                    _record(
                        operation,
                        memory_id=memory_id,
                        scope=scope,
                        at=at,
                        source_run_id=source_run_id,
                        default_ttl_hours=default_ttl_hours,
                    )
                )
                if isinstance(operation, UpdateMemory):
                    invalidations.append((operation.target, memory_id))

    return ConsolidationPlan(
        writes=tuple(writes), invalidations=tuple(invalidations), rejected=rejected
    )


def _record(
    operation: AddMemory | UpdateMemory,
    *,
    memory_id: MemoryId,
    scope: MemoryScope,
    at: datetime,
    source_run_id: RunId | None,
    default_ttl_hours: float | None,
) -> MemoryRecord:
    ttl = operation.ttl_hours if operation.ttl_hours is not None else default_ttl_hours
    return MemoryRecord(
        id=memory_id,
        scope=scope,
        content=operation.content.strip(),
        kind=operation.memory_kind,
        tags=operation.tags,
        importance=operation.importance,
        created_at=at,
        valid_from=at,
        expires_at=at + timedelta(hours=ttl) if ttl else None,
        source_run_id=source_run_id,
    )


__all__ = [
    "EXTRACT_INSTRUCTION",
    "MAX_OPERATIONS",
    "REFLECT_INSTRUCTION",
    "ConsolidationPlan",
    "consolidation_request",
    "parse_operations",
    "plan_writes",
]
