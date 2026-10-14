"""Memory capabilities.

Scoping is the runtime's job, not the model's: the agent says what to remember,
and the tool decides where it lands. That is what stops agents reading or
corrupting each other's memory, and it is not a decision worth delegating to a
token sampler.

The model *does* choose kind and importance, because those are judgements about
the content rather than about authority -- and getting them wrong costs ranking
quality, not isolation.
"""

from __future__ import annotations

from skynet.application.function_tool import tool
from skynet.application.memory_ranking import candidate_limit, rank
from skynet.capabilities.base import BoundTool
from skynet.domain.ids import MemoryId
from skynet.domain.memory import MemoryKind, MemoryQuery, MemoryRecord, MemoryScope
from skynet.domain.permissions import Permission
from skynet.domain.tools import ToolContext
from skynet.ports.clock import Clock, IdGenerator
from skynet.ports.memory import MemoryStore

KIND_HELP = (
    "semantic for a fact that is true, procedural for how something is done "
    "here, episodic for something that happened"
)


def _scope(ctx: ToolContext) -> MemoryScope:
    return MemoryScope.agent(ctx.agent_name)


def _kind(value: str) -> MemoryKind:
    """Fall back rather than fail: an unrecognised kind is a mislabelled memory,
    and refusing to store it would lose the content over a taxonomy quibble."""
    try:
        return MemoryKind(value.strip().lower())
    except ValueError:
        return MemoryKind.SEMANTIC


class MemoryWriteTool(BoundTool):
    def __init__(self, store: MemoryStore, ids: IdGenerator, clock: Clock) -> None:
        self._store = store
        self._ids = ids
        self._clock = clock
        self._tool = tool(
            name="memory_write",
            description=(
                "Remember something for later use in this or a future task. "
                "Store durable facts, conventions and outcomes -- not chit-chat "
                "or intermediate working."
            ),
            required_permissions={Permission.MEMORY_WRITE},
            idempotent=False,
        )(self._write)

    async def _write(
        self,
        content: str,
        ctx: ToolContext,
        kind: str = "semantic",
        tags: str = "",
        importance: float = 0.5,
    ) -> str:
        """What to remember, its kind (semantic, procedural or episodic),
        optional comma-separated tags, and how important it is from 0 to 1."""
        now = self._clock.now()
        record = MemoryRecord(
            id=MemoryId(self._ids.new_id("mem")),
            scope=_scope(ctx),
            content=content,
            kind=_kind(kind),
            tags=tuple(t.strip() for t in tags.split(",") if t.strip()),
            importance=min(1.0, max(0.0, importance)),
            metadata={"run_id": str(ctx.run_id)},
            created_at=now,
            valid_from=now,
            source_run_id=ctx.run_id,
        )
        memory_id = await self._store.put(record)
        return f"Stored {record.kind.value} memory {memory_id}."


class MemorySearchTool(BoundTool):
    def __init__(self, store: MemoryStore, clock: Clock) -> None:
        self._store = store
        self._clock = clock
        self._tool = tool(
            name="memory_search",
            description="Search previously remembered facts, conventions and outcomes.",
            required_permissions={Permission.MEMORY_READ},
            idempotent=True,
        )(self._search)

    async def _search(self, query: str, ctx: ToolContext, kind: str = "", limit: int = 5) -> str:
        """What to look for, optionally narrowed to one kind of memory."""
        now = self._clock.now()
        kinds = (_kind(kind),) if kind.strip() else ()
        candidates = await self._store.search(
            MemoryQuery(
                scopes=(_scope(ctx),),
                text=query,
                kinds=kinds,
                limit=candidate_limit(limit),
                as_of=now,
            )
        )
        hits = rank(candidates, at=now, limit=limit)
        if not hits:
            return "No matching memories."
        # The kind is shown because it changes how the model should treat the
        # line: "this happened once" is not "this is true".
        return "\n".join(f"- ({hit.record.kind.value}) {hit.record.content}" for hit in hits)


__all__ = ["MemorySearchTool", "MemoryWriteTool"]
