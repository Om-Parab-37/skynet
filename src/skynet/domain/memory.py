"""Agent memory: what a memory *is*, where it lives, and when it was true.

Three ideas carry the whole design.

**A scope is a value, not a string.** ``MemoryScope(level, owner)`` renders to
``"level:owner"`` for logs and SQL, but the format is defined in exactly one
place, so nobody can spell it differently.

**Memory is bi-temporal.** ``created_at`` is transaction time -- when the agent
learned it. ``valid_from``/``valid_to`` is valid time -- when the fact actually
held. A contradicted memory is never deleted; it is stamped with ``valid_to``
and a ``superseded_by`` pointer. That is what keeps "what did the agent believe
last Tuesday?" answerable, which is the only thing that makes a memory bug
diagnosable after the fact.

**Retrieval is split in two.** A store reports *similarity* (``relevance``); the
application fuses similarity with recency, importance and scope precedence into
a ``score``. Keeping the fusion out of the adapters is what lets a keyword fake
and a pgvector table pass the same contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from skynet.domain.ids import MemoryId, RunId


class MemoryKind(StrEnum):
    """What sort of thing is remembered.

    Kept separate because a model treats them very differently: "this is true"
    is not "this happened once" is not "this is how the job is done here".
    """

    EPISODIC = "episodic"
    """Something that happened: an event, an observation, an outcome."""
    SEMANTIC = "semantic"
    """A fact believed to hold."""
    PROCEDURAL = "procedural"
    """How to do something here: a recipe, a convention, a gotcha."""


class ScopeLevel(StrEnum):
    """Namespace levels, listed most specific first."""

    RUN = "run"
    AGENT = "agent"
    PROJECT = "project"
    USER = "user"

    @property
    def precedence(self) -> int:
        """0 for the most specific level. Used as a ranking *tiebreak* only."""
        return _PRECEDENCE[self]


_PRECEDENCE = {
    ScopeLevel.RUN: 0,
    ScopeLevel.AGENT: 1,
    ScopeLevel.PROJECT: 2,
    ScopeLevel.USER: 3,
}


class MemoryScope(BaseModel):
    """The definition of a namespace -- the one place the format is decided."""

    model_config = ConfigDict(frozen=True)

    level: ScopeLevel
    owner: str

    def namespace(self) -> str:
        return f"{self.level.value}:{self.owner}"

    @classmethod
    def parse(cls, namespace: str) -> MemoryScope:
        level, separator, owner = namespace.partition(":")
        if not separator or not owner:
            raise ValueError(f"malformed memory namespace: {namespace!r}")
        return cls(level=ScopeLevel(level), owner=owner)

    @classmethod
    def run(cls, run_id: RunId | str) -> MemoryScope:
        return cls(level=ScopeLevel.RUN, owner=str(run_id))

    @classmethod
    def agent(cls, name: str) -> MemoryScope:
        return cls(level=ScopeLevel.AGENT, owner=name)

    @classmethod
    def project(cls, name: str) -> MemoryScope:
        return cls(level=ScopeLevel.PROJECT, owner=name)

    @classmethod
    def user(cls, name: str) -> MemoryScope:
        return cls(level=ScopeLevel.USER, owner=name)


class MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: MemoryId
    scope: MemoryScope
    content: str
    kind: MemoryKind = MemoryKind.SEMANTIC
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = {}
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    """0..1. Set by whoever wrote the memory; the consolidator is the main author."""

    created_at: datetime | None = None
    """Transaction time: when this was learned."""
    valid_from: datetime | None = None
    """Valid time, start: when the fact began to hold."""
    valid_to: datetime | None = None
    """Valid time, end. Set on contradiction. The row itself is never deleted."""
    superseded_by: MemoryId | None = None
    """The replacement, when this was contradicted rather than merely expired."""
    expires_at: datetime | None = None
    """TTL: never contradicted, just stale."""
    source_run_id: RunId | None = None

    def is_live(self, at: datetime) -> bool:
        """Whether this memory was believed at ``at``."""
        if self.valid_from is not None and at < self.valid_from:
            return False
        if self.valid_to is not None and at >= self.valid_to:
            return False
        return not (self.expires_at is not None and at >= self.expires_at)


class MemoryQuery(BaseModel):
    """One retrieval, across every scope the caller is allowed to read.

    Multi-scope is a property of the *query* rather than a loop in the caller:
    a store embeds the text once and issues one statement. Looping outside would
    embed the same text N times and turn one ``await`` into N.
    """

    model_config = ConfigDict(frozen=True)

    scopes: tuple[MemoryScope, ...] = ()
    text: str = ""
    kinds: tuple[MemoryKind, ...] = ()
    tags: tuple[str, ...] = ()
    limit: int = 8
    min_score: float = 0.0
    """Threshold on raw ``relevance``, applied by the store."""
    as_of: datetime | None = None
    """Read the world as it was believed at this instant.

    ``None`` means "apply supersession only": records with ``valid_to`` set are
    excluded, but TTLs are not evaluated, because a store has no clock of its
    own. Callers on the hot path always pass the run's clock reading.
    """
    include_invalidated: bool = False
    """Ask for the history too -- diagnostics, never a prompt."""

    def namespaces(self) -> tuple[str, ...]:
        return tuple(scope.namespace() for scope in self.scopes)

    def signature(self) -> str:
        """A stable identity for "this exact retrieval has already been issued".

        Purely derived, so it is the same in a live run and in its replay.
        """
        scopes = ",".join(sorted(self.namespaces()))
        kinds = ",".join(sorted(kind.value for kind in self.kinds))
        tags = ",".join(sorted(self.tags))
        return f"{scopes}|{self.text.strip().lower()}|{kinds}|{tags}|{self.limit}"


class RetrievalComponents(BaseModel):
    """Raw per-arm scores from a hybrid store, kept so ranking can re-weight.

    A fused rank is not a similarity, so thresholding on it would be
    meaningless; carrying the components is what keeps ``min_score`` honest.
    """

    model_config = ConfigDict(frozen=True)

    semantic: float | None = None
    lexical: float | None = None
    fused: float | None = None


class MemoryHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: MemoryRecord
    relevance: float = 0.0
    """Raw similarity, 0..1, as reported by the store."""
    score: float = 0.0
    """Fused score from ``application.memory_ranking.rank``. 0 until ranked."""
    components: RetrievalComponents | None = None


MAX_WORKING_MEMORIES = 24
"""How many recalled memories a run carries at once.

A cap rather than a growing set: recall fires more than once now, and an
unbounded working set would quietly become the largest thing in every prompt.
"""


def merge_hits(
    existing: Iterable[MemoryHit],
    incoming: Iterable[MemoryHit],
    *,
    limit: int = MAX_WORKING_MEMORIES,
) -> tuple[MemoryHit, ...]:
    """Accumulate two recalls into one working set, keeping the better score.

    Lives in the domain because the reducer calls it: no clock, no re-scoring,
    nothing that could differ between a live run and its replay. The id tiebreak
    is load-bearing -- without it two equal-scoring memories can swap places on
    replay, which changes the prompt and so the answer.
    """
    best: dict[MemoryId, MemoryHit] = {}
    for hit in (*existing, *incoming):
        seen = best.get(hit.record.id)
        if seen is None or hit.score > seen.score:
            best[hit.record.id] = hit
    ordered = sorted(best.values(), key=lambda hit: (-hit.score, hit.record.id))
    return tuple(ordered[:limit])


# -- consolidation ------------------------------------------------------------
#
# Deliberately no DELETE and no NOOP. Deletion would break the bi-temporal
# guarantee, so contradiction is InvalidateMemory; and a NOOP is just an absent
# operation that would bloat the log to say nothing.


class ConsolidationMode(StrEnum):
    EXTRACT = "extract"
    """Read a slice of the conversation and write down what was learned."""
    REFLECT = "reflect"
    """Read existing memories and write down higher-level insight."""


class MemoryOperationBase(BaseModel):
    model_config = ConfigDict(frozen=True)


class AddMemory(MemoryOperationBase):
    kind: Literal["add"] = "add"
    content: str
    memory_kind: MemoryKind = MemoryKind.SEMANTIC
    tags: tuple[str, ...] = ()
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_hours: float | None = None


class UpdateMemory(MemoryOperationBase):
    """A correction. Executed as invalidate-then-add, never as a mutation."""

    kind: Literal["update"] = "update"
    target: MemoryId
    content: str
    memory_kind: MemoryKind = MemoryKind.SEMANTIC
    tags: tuple[str, ...] = ()
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_hours: float | None = None


class InvalidateMemory(MemoryOperationBase):
    """The fact stopped being true and nothing replaces it."""

    kind: Literal["invalidate"] = "invalidate"
    target: MemoryId
    reason: str = ""


MemoryOperation = Annotated[
    AddMemory | UpdateMemory | InvalidateMemory,
    Field(discriminator="kind"),
]


__all__ = [
    "MAX_WORKING_MEMORIES",
    "AddMemory",
    "ConsolidationMode",
    "InvalidateMemory",
    "MemoryHit",
    "MemoryKind",
    "MemoryOperation",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScope",
    "merge_hits",
    "RetrievalComponents",
    "ScopeLevel",
    "UpdateMemory",
]
