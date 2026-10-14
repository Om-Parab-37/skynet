"""In-memory implementations of every port.

These ship inside the package rather than in ``tests/`` for two reasons: adapter
authors reuse them, and the contract suites in ``skynet.testing.contracts`` run
against both these and the real adapters.

``FixedClock`` and ``SequentialIdGenerator`` matter more than they look: they
make the event log byte-stable, which is what lets tests assert an exact
sequence of events instead of poking at final state.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

from skynet.domain.agent import AgentContext, AgentSpec, TraceContext
from skynet.domain.errors import AgentNotFound, ProviderError, ToolNotFound
from skynet.domain.events import RunEvent
from skynet.domain.ids import CallId, MemoryId, RunId
from skynet.domain.memory import MemoryHit, MemoryQuery, MemoryRecord, MemoryScope
from skynet.domain.messages import ToolCall
from skynet.domain.models import ModelRequest, ModelResponse
from skynet.domain.prompts import PromptRef, PromptTemplate
from skynet.domain.run import AgentRun, replay
from skynet.domain.storage import ObjectRef
from skynet.domain.task import AgentResult, Task
from skynet.domain.tools import ToolContext, ToolResult, ToolSpec, ToolStatus
from skynet.ports.model import ProviderCapabilities
from skynet.ports.sandbox import ExecRequest, ExecResult, ResourceProfile
from skynet.ports.tracing import SpanKind

# -- time and identity --------------------------------------------------------


class FixedClock:
    def __init__(self, start: datetime | None = None, step_ms: int = 1000) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._step = timedelta(milliseconds=step_ms)

    def now(self) -> datetime:
        current = self._now
        self._now += self._step
        return current


class FrozenClock:
    def __init__(self, at: datetime | None = None) -> None:
        self._at = at or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._at


class SequentialIdGenerator:
    def __init__(self) -> None:
        self._counters: dict[str, count[int]] = {}

    def new_id(self, prefix: str = "") -> str:
        counter = self._counters.setdefault(prefix, count(1))
        return f"{prefix}-{next(counter)}" if prefix else str(next(counter))

    def new_trace_id(self) -> str:
        counter = self._counters.setdefault("__trace__", count(1))
        return f"{next(counter):032x}"


# -- model --------------------------------------------------------------------


class ScriptedModelProvider:
    """Returns canned responses, or computes one from the request.

    Records every request, so tests can assert on what the agent actually asked
    the model -- which tools it offered, what context it built.
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse] = (),
        *,
        rule: Callable[[ModelRequest], ModelResponse] | None = None,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self._responses = list(responses)
        self._rule = rule
        self._capabilities = capabilities or ProviderCapabilities(
            native_tools=True, parallel_tool_calls=True
        )
        self.requests: list[ModelRequest] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._rule is not None:
            return self._rule(request)
        if not self._responses:
            raise ProviderError("ScriptedModelProvider ran out of responses")
        return self._responses.pop(0)


class FlakyModelProvider:
    """Fails the first ``fail_times`` calls, then delegates to ``inner``."""

    def __init__(self, inner: Any, *, fail_times: int = 1, retryable: bool = True) -> None:
        self._inner = inner
        self._remaining = fail_times
        self._retryable = retryable

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._inner.capabilities

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if self._remaining > 0:
            self._remaining -= 1
            raise ProviderError("transient provider failure", retryable=self._retryable)
        return await self._inner.generate(request)


# -- tools --------------------------------------------------------------------


def _spec(name: str, description: str = "", **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description or name,
        parameters={"type": "object", "properties": {}},
        **kwargs,
    )


class EchoTool:
    def __init__(self, name: str = "echo", **spec_kwargs: Any) -> None:
        self._spec = _spec(name, "Echo the given text back.", **spec_kwargs)

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=self._spec.name,
            status=ToolStatus.OK,
            content=str(call.arguments.get("text", "")),
        )


class FailingTool:
    def __init__(self, name: str = "boom", message: str = "tool exploded") -> None:
        self._spec = _spec(name, "Always raises.")
        self._message = message

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        raise RuntimeError(self._message)


class SlowTool:
    def __init__(self, name: str = "slow", delay: float = 0.05, **spec_kwargs: Any) -> None:
        self._spec = _spec(name, "Sleeps before answering.", **spec_kwargs)
        self._delay = delay

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult(
            call_id=call.id,
            name=self._spec.name,
            status=ToolStatus.OK,
            content=f"{self._spec.name} done",
        )


class BigOutputTool:
    def __init__(self, name: str = "big", size: int = 50_000) -> None:
        self._spec = _spec(name, "Produces a large amount of output.")
        self._size = size

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=self._spec.name,
            status=ToolStatus.OK,
            content="x" * self._size,
        )


# -- memory, storage, prompts -------------------------------------------------


def _words(text: str) -> set[str]:
    """Word tokens worth matching on. Two letters and under are noise."""
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _trigrams(text: str) -> set[str]:
    normalised = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()} "
    return {normalised[i : i + 3] for i in range(max(0, len(normalised) - 2))}


def _similarity(query: str, content: str) -> float:
    """Word overlap with character-trigram partial credit.

    The trigram arm is what makes "pangolin" find "pangolins": a token-only
    score gives a morphological variant exactly zero, and a store whose only
    near-miss behaviour is "no results" cannot stand in for a semantic one in
    tests. Containment (share of *query* trigrams present), not Jaccard, so a
    long document is not penalised for being long.
    """
    terms = _words(query)
    if not terms:
        return 1.0
    lexical = len(terms & _words(content)) / len(terms)
    grams = _trigrams(query)
    fuzzy = len(grams & _trigrams(content)) / len(grams) if grams else 0.0
    return 0.7 * lexical + 0.3 * fuzzy


class InMemoryMemoryStore:
    """Process-local store honouring scopes, kinds, tags and bi-temporality.

    Deliberately more than a dictionary: it is one of the two implementations
    the shared ``MemoryStoreContract`` runs against, so every filtering rule the
    real store must obey has to be obeyed here too, or the contract proves
    nothing. Scoring is *not* its job -- it reports raw relevance and leaves
    fusion to ``application.memory_ranking``.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[MemoryId, MemoryRecord]] = {}
        self._ids = count(1)

    async def put(self, record: MemoryRecord) -> MemoryId:
        memory_id = record.id or MemoryId(f"mem-{next(self._ids)}")
        stored = record.model_copy(update={"id": memory_id})
        self._records.setdefault(record.scope.namespace(), {})[memory_id] = stored
        return memory_id

    async def get(self, scope: MemoryScope, memory_id: MemoryId) -> MemoryRecord | None:
        return self._records.get(scope.namespace(), {}).get(memory_id)

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        hits: list[MemoryHit] = []
        for namespace in query.namespaces():
            for record in self._records.get(namespace, {}).values():
                if not self._matches(record, query):
                    continue
                relevance = _similarity(query.text, record.content)
                if relevance <= 0.0 or relevance < query.min_score:
                    continue
                hits.append(MemoryHit(record=record, relevance=relevance, score=relevance))
        hits.sort(key=lambda hit: (-hit.relevance, hit.record.id))
        return tuple(hits[: query.limit])

    @staticmethod
    def _matches(record: MemoryRecord, query: MemoryQuery) -> bool:
        if query.kinds and record.kind not in query.kinds:
            return False
        if query.tags and not set(query.tags) & set(record.tags):
            return False
        if query.include_invalidated:
            return True
        if query.as_of is not None:
            return record.is_live(query.as_of)
        # No reading given, so only supersession can be judged: a store has no
        # clock, and guessing one would make TTL expiry depend on wall time in
        # a fake that exists to be deterministic.
        return record.valid_to is None

    async def invalidate(
        self,
        scope: MemoryScope,
        memory_id: MemoryId,
        *,
        at: datetime,
        superseded_by: MemoryId | None = None,
    ) -> bool:
        records = self._records.get(scope.namespace(), {})
        record = records.get(memory_id)
        if record is None:
            return False
        records[memory_id] = record.model_copy(
            update={"valid_to": at, "superseded_by": superseded_by}
        )
        return True

    async def delete(self, scope: MemoryScope, memory_id: MemoryId) -> bool:
        return self._records.get(scope.namespace(), {}).pop(memory_id, None) is not None

    async def purge_expired(self, *, before: datetime) -> int:
        purged = 0
        for records in self._records.values():
            for memory_id, record in list(records.items()):
                ended = record.valid_to or record.expires_at
                if ended is not None and ended < before:
                    del records[memory_id]
                    purged += 1
        return purged


class InMemoryObjectStore:
    def __init__(self, bucket: str = "skynet") -> None:
        self.bucket = bucket
        self._data: dict[str, bytes] = {}
        self._refs: dict[str, ObjectRef] = {}

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectRef:
        ref = ObjectRef(
            bucket=self.bucket,
            key=key,
            size=len(data),
            content_type=content_type,
            metadata=dict(metadata or {}),
        )
        self._data[key] = data
        self._refs[key] = ref
        return ref

    async def get(self, ref: ObjectRef) -> bytes:
        return self._data[ref.key]

    async def delete(self, ref: ObjectRef) -> bool:
        self._refs.pop(ref.key, None)
        return self._data.pop(ref.key, None) is not None

    async def list(self, prefix: str) -> tuple[ObjectRef, ...]:
        return tuple(ref for key, ref in sorted(self._refs.items()) if key.startswith(prefix))


class StaticPromptProvider:
    def __init__(self, templates: Mapping[str, str] | None = None, default: str = "") -> None:
        self._templates = dict(templates or {})
        self._default = default

    async def fetch(self, ref: PromptRef) -> PromptTemplate:
        return PromptTemplate(
            ref=ref,
            template=self._templates.get(ref.name, self._default),
            version=1,
            source="static",
        )


# -- tracing and events -------------------------------------------------------


class RecordingSpan:
    def __init__(self, name: str, kind: SpanKind, parent: TraceContext | None) -> None:
        self.name = name
        self.kind = kind
        self.parent = parent
        self.attributes: dict[str, Any] = {}
        self.input: Any = None
        self.output: Any = None
        self.error: Any = None

    def set_attributes(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def set_output(self, output: Any) -> None:
        self.output = output

    def record_error(self, error: Any) -> None:
        self.error = error

    @property
    def context(self) -> TraceContext:
        return self.parent or TraceContext()


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[RecordingSpan]:
        recorded = RecordingSpan(name, kind, parent)
        recorded.input = input
        recorded.attributes.update(attributes or {})
        self.spans.append(recorded)
        yield recorded

    async def flush(self) -> None:
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.spans)


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def publish(self, event: RunEvent) -> None:
        self.events.append(event)

    def kinds(self) -> tuple[str, ...]:
        return tuple(e.kind for e in self.events)


# -- sandbox ------------------------------------------------------------------


class FakeSandbox:
    """Dict-backed filesystem plus canned responses keyed by command prefix."""

    def __init__(
        self,
        files: Mapping[str, bytes] | None = None,
        responses: Mapping[str, ExecResult] | None = None,
        default: ExecResult | None = None,
    ) -> None:
        self._files: dict[str, bytes] = dict(files or {})
        self._responses = dict(responses or {})
        self._default = default or ExecResult(exit_code=0, stdout="")
        self.commands: list[tuple[str, ...]] = []

    @property
    def id(self) -> str:
        return "fake-sandbox"

    @property
    def workdir(self) -> str:
        return "/workspace"

    async def exec(self, request: ExecRequest) -> ExecResult:
        self.commands.append(request.command)
        joined = " ".join(request.command)
        for prefix, result in self._responses.items():
            if joined.startswith(prefix):
                return result
        return self._default

    async def write_file(self, path: str, data: bytes) -> None:
        self._files[path] = data

    async def read_file(self, path: str) -> bytes:
        return self._files[path]

    async def list_files(self, path: str | None = None) -> tuple[str, ...]:
        prefix = path or self.workdir
        return tuple(sorted(p for p in self._files if p.startswith(prefix)))


class FakeSandboxProvider:
    def __init__(self, sandbox: FakeSandbox | None = None) -> None:
        self.sandbox = sandbox or FakeSandbox()
        self.released: list[str] = []

    @asynccontextmanager
    async def acquire(
        self,
        *,
        workspace_id: str,
        image: str | None = None,
        resources: ResourceProfile | None = None,
    ) -> AsyncIterator[FakeSandbox]:
        yield self.sandbox

    async def exists(self, workspace_id: str) -> bool:
        return workspace_id not in self.released

    async def release(self, workspace_id: str) -> None:
        self.released.append(workspace_id)


# -- runs, agents, approvals --------------------------------------------------


class InMemoryRunStore:
    def __init__(self) -> None:
        self._events: dict[RunId, list[RunEvent]] = {}
        self._snapshots: dict[RunId, AgentRun] = {}

    async def append(self, run_id: RunId, events: Sequence[RunEvent]) -> None:
        self._events.setdefault(run_id, []).extend(events)

    async def events(self, run_id: RunId) -> tuple[RunEvent, ...]:
        return tuple(self._events.get(run_id, ()))

    async def snapshot(self, run: AgentRun) -> None:
        self._snapshots[run.run_id] = run

    async def load(self, run_id: RunId) -> AgentRun | None:
        return self._snapshots.get(run_id)

    async def list_runs(self, limit: int = 50) -> tuple[AgentRun, ...]:
        return tuple(list(self._snapshots.values())[:limit])

    async def rebuild(self, run_id: RunId, initial: AgentRun) -> AgentRun:
        """Recovery path: state from the log rather than from the snapshot."""
        return replay(initial, await self.events(run_id))


class StaticAgentCatalog:
    def __init__(self, specs: Sequence[AgentSpec] = ()) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def register(self, spec: AgentSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> AgentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise AgentNotFound(f"unknown agent '{name}'") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def all(self) -> tuple[AgentSpec, ...]:
        return tuple(self._specs.values())


class StubSpawner:
    """Sub-agent tests without recursion: canned results keyed by agent name."""

    def __init__(self, results: Mapping[str, AgentResult] | None = None) -> None:
        self._results = dict(results or {})
        self.calls: list[tuple[str, Task, AgentContext]] = []
        self._ids = count(1)

    async def spawn(
        self, *, agent: str, task: Task, parent: AgentContext, call_id: CallId
    ) -> tuple[RunId, AgentResult]:
        self.calls.append((agent, task, parent))
        result = self._results.get(agent, AgentResult(output=f"{agent} finished"))
        return RunId(f"child-{next(self._ids)}"), result


class AutoApproveGate:
    async def request(self, run_id: RunId, call_ids: Sequence[CallId], reason: str) -> None:
        return None


class RecordingApprovalGate:
    def __init__(self) -> None:
        self.requests: list[tuple[RunId, tuple[CallId, ...], str]] = []

    async def request(self, run_id: RunId, call_ids: Sequence[CallId], reason: str) -> None:
        self.requests.append((run_id, tuple(call_ids), reason))


class RaisingTool:
    """A tool whose spec exists but which is never expected to be invoked."""

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        raise ToolNotFound(f"{self._spec.name} should not have been invoked")


__all__ = [
    "AutoApproveGate",
    "BigOutputTool",
    "EchoTool",
    "FailingTool",
    "FakeSandbox",
    "FakeSandboxProvider",
    "FixedClock",
    "FlakyModelProvider",
    "FrozenClock",
    "InMemoryMemoryStore",
    "InMemoryObjectStore",
    "InMemoryRunStore",
    "RaisingTool",
    "RecordingApprovalGate",
    "RecordingEventSink",
    "RecordingSpan",
    "RecordingTracer",
    "ScriptedModelProvider",
    "SequentialIdGenerator",
    "SlowTool",
    "StaticAgentCatalog",
    "StaticPromptProvider",
    "StubSpawner",
]
