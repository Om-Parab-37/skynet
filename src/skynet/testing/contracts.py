"""Port contract suites.

Subclass one of these, point ``instance`` at an implementation, and the same
behavioural expectations run against it. The in-memory fakes pass them and so
must every real adapter -- which is how "MinIO can be swapped out" stops being a
claim and becomes a test.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from skynet.domain.errors import SandboxError
from skynet.domain.ids import MemoryId
from skynet.domain.memory import MemoryKind, MemoryQuery, MemoryRecord, MemoryScope
from skynet.ports.sandbox import ExecRequest

# How the shipped sandboxes report a file that is not there.
_MISSING_FILE = (FileNotFoundError, KeyError, SandboxError, OSError)

DAY = timedelta(days=1)


class ObjectStoreContract:
    @pytest.fixture
    @abstractmethod
    def store(self) -> Any: ...

    async def test_put_then_get_round_trips(self, store):
        ref = await store.put("a/b.txt", b"payload", content_type="text/plain")
        assert ref.key == "a/b.txt"
        assert ref.size == 7
        assert await store.get(ref) == b"payload"

    async def test_list_filters_by_prefix(self, store):
        await store.put("runs/1/x.txt", b"x")
        await store.put("runs/2/y.txt", b"y")
        keys = {r.key for r in await store.list("runs/1/")}
        assert keys == {"runs/1/x.txt"}

    async def test_delete_removes_the_object(self, store):
        ref = await store.put("gone.txt", b"z")
        assert await store.delete(ref) is True
        assert await store.delete(ref) is False

    async def test_binary_payloads_survive(self, store):
        payload = bytes(range(256))
        ref = await store.put("bin.dat", payload)
        assert await store.get(ref) == payload


NS = MemoryScope.agent("ns")
OTHER = MemoryScope.agent("other")
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _record(
    memory_id: str, content: str, *, scope: MemoryScope = NS, **fields: Any
) -> MemoryRecord:
    return MemoryRecord(id=MemoryId(memory_id), scope=scope, content=content, **fields)


class MemoryStoreContract:
    """What every memory store must do.

    The line this suite holds: **it fixes filtering, which is correctness, and
    never ranking, which is quality.** An earlier version asserted an exact
    result list for a keyword search -- an assertion no store with a semantic
    arm can pass, because "volcanoes erupt" has a small non-zero similarity to
    "pangolins". That is a design failure wearing a test failure's clothes: it
    would have forced every correct hybrid implementation to drop real hits in
    order to look right. So relevance is asserted only as *ordering*, and
    thresholds are asserted only as *filters that are applied*.
    """

    @pytest.fixture
    @abstractmethod
    def memory(self) -> Any: ...

    async def test_put_then_get_round_trips(self, memory):
        memory_id = await memory.put(_record("m1", "the sky is blue"))
        fetched = await memory.get(NS, memory_id)
        assert fetched is not None and fetched.content == "the sky is blue"

    async def test_put_is_idempotent_by_id(self, memory):
        await memory.put(_record("m1", "first"))
        await memory.put(_record("m1", "second"))
        hits = await memory.search(MemoryQuery(scopes=(NS,)))
        assert [h.record.content for h in hits] == ["second"]

    async def test_search_ranks_the_relevant_record_first(self, memory):
        await memory.put(_record("m1", "pangolins eat ants"))
        await memory.put(_record("m2", "volcanoes erupt"))
        hits = await memory.search(MemoryQuery(scopes=(NS,), text="pangolins", limit=5))
        assert hits and hits[0].record.id == MemoryId("m1")

    async def test_search_reports_relevance_in_the_unit_interval(self, memory):
        await memory.put(_record("m1", "pangolins eat ants"))
        hit = (await memory.search(MemoryQuery(scopes=(NS,), text="pangolins")))[0]
        assert 0.0 <= hit.relevance <= 1.0

    async def test_search_honours_min_score(self, memory):
        await memory.put(_record("m1", "pangolins eat ants"))
        query = MemoryQuery(scopes=(NS,), text="pangolins")
        assert await memory.search(query.model_copy(update={"min_score": 1.01})) == ()
        assert await memory.search(query) != ()

    async def test_search_is_scoped(self, memory):
        await memory.put(_record("m1", "secret", scope=NS))
        assert await memory.search(MemoryQuery(scopes=(OTHER,), text="secret")) == ()

    async def test_search_spans_every_requested_scope(self, memory):
        await memory.put(_record("m1", "shared knowledge", scope=NS))
        await memory.put(_record("m2", "shared knowledge", scope=OTHER))
        hits = await memory.search(MemoryQuery(scopes=(NS, OTHER), text="shared knowledge"))
        assert {h.record.id for h in hits} == {MemoryId("m1"), MemoryId("m2")}

    async def test_search_filters_by_kind(self, memory):
        await memory.put(_record("m1", "a fact", kind=MemoryKind.SEMANTIC))
        await memory.put(_record("m2", "a fact", kind=MemoryKind.EPISODIC))
        hits = await memory.search(
            MemoryQuery(scopes=(NS,), text="fact", kinds=(MemoryKind.EPISODIC,))
        )
        assert [h.record.id for h in hits] == [MemoryId("m2")]

    async def test_search_filters_by_tag(self, memory):
        await memory.put(_record("m1", "deploy notes", tags=("ops",)))
        await memory.put(_record("m2", "deploy notes", tags=("docs",)))
        hits = await memory.search(MemoryQuery(scopes=(NS,), text="deploy", tags=("ops",)))
        assert [h.record.id for h in hits] == [MemoryId("m1")]

    async def test_search_respects_the_limit(self, memory):
        for index in range(5):
            await memory.put(_record(f"m{index}", "shared keyword here"))
        assert len(await memory.search(MemoryQuery(scopes=(NS,), text="keyword", limit=2))) == 2

    async def test_empty_text_returns_everything_in_scope(self, memory):
        await memory.put(_record("m1", "pangolins eat ants"))
        await memory.put(_record("m2", "volcanoes erupt"))
        hits = await memory.search(MemoryQuery(scopes=(NS,), text=""))
        assert {h.record.id for h in hits} == {MemoryId("m1"), MemoryId("m2")}

    # -- bi-temporality: invalidate, never delete -----------------------------

    async def test_invalidate_keeps_the_record_and_stamps_it(self, memory):
        await memory.put(_record("m1", "the office is in Berlin", valid_from=T0))
        await memory.put(_record("m2", "the office is in Munich", valid_from=T0 + DAY))
        assert (
            await memory.invalidate(NS, MemoryId("m1"), at=T0 + DAY, superseded_by=MemoryId("m2"))
            is True
        )
        old = await memory.get(NS, MemoryId("m1"))
        assert old is not None
        assert old.valid_to == T0 + DAY
        assert old.superseded_by == MemoryId("m2")

    async def test_invalidated_records_leave_the_results(self, memory):
        await memory.put(_record("m1", "the office is in Berlin", valid_from=T0))
        await memory.invalidate(NS, MemoryId("m1"), at=T0 + DAY)
        query = MemoryQuery(scopes=(NS,), text="office", as_of=T0 + 2 * DAY)
        assert await memory.search(query) == ()

    async def test_as_of_still_returns_the_earlier_belief(self, memory):
        await memory.put(_record("m1", "the office is in Berlin", valid_from=T0))
        await memory.invalidate(NS, MemoryId("m1"), at=T0 + DAY)
        hits = await memory.search(
            MemoryQuery(scopes=(NS,), text="office", as_of=T0 + timedelta(hours=1))
        )
        assert [h.record.id for h in hits] == [MemoryId("m1")]

    async def test_include_invalidated_returns_the_history(self, memory):
        await memory.put(_record("m1", "the office is in Berlin", valid_from=T0))
        await memory.invalidate(NS, MemoryId("m1"), at=T0 + DAY)
        hits = await memory.search(
            MemoryQuery(scopes=(NS,), text="office", as_of=T0 + 2 * DAY, include_invalidated=True)
        )
        assert [h.record.id for h in hits] == [MemoryId("m1")]

    async def test_expired_records_are_not_returned(self, memory):
        await memory.put(_record("m1", "the build token", expires_at=T0 + DAY))
        assert await memory.search(MemoryQuery(scopes=(NS,), text="build", as_of=T0)) != ()
        assert (
            await memory.search(MemoryQuery(scopes=(NS,), text="build", as_of=T0 + 2 * DAY)) == ()
        )

    async def test_records_not_yet_valid_are_not_returned(self, memory):
        await memory.put(_record("m1", "the new policy", valid_from=T0 + DAY))
        assert await memory.search(MemoryQuery(scopes=(NS,), text="policy", as_of=T0)) == ()

    async def test_invalidate_reports_a_missing_record(self, memory):
        assert await memory.invalidate(NS, MemoryId("nope"), at=T0) is False

    async def test_delete_removes_the_record(self, memory):
        await memory.put(_record("m1", "x"))
        assert await memory.delete(NS, MemoryId("m1")) is True
        assert await memory.get(NS, MemoryId("m1")) is None
        assert await memory.delete(NS, MemoryId("m1")) is False


class SandboxContract:
    @pytest.fixture
    @abstractmethod
    def provider(self) -> Any: ...

    async def test_command_output_and_exit_code(self, provider):
        async with provider.acquire(workspace_id="contract") as sandbox:
            result = await sandbox.exec(
                ExecRequest(command=("/bin/sh", "-c", "echo hello"), timeout_s=30)
            )
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_failure_exit_code_and_stderr(self, provider):
        async with provider.acquire(workspace_id="contract") as sandbox:
            result = await sandbox.exec(
                ExecRequest(command=("/bin/sh", "-c", "echo bad >&2; exit 3"), timeout_s=30)
            )
        assert result.exit_code == 3
        assert "bad" in result.stderr

    async def test_timeout_is_reported_not_raised(self, provider):
        async with provider.acquire(workspace_id="contract") as sandbox:
            result = await sandbox.exec(
                ExecRequest(command=("/bin/sh", "-c", "sleep 5"), timeout_s=0.5)
            )
        assert result.timed_out is True

    async def test_file_write_then_read_round_trips(self, provider):
        async with provider.acquire(workspace_id="contract") as sandbox:
            path = f"{sandbox.workdir}/note.txt"
            await sandbox.write_file(path, b"contents")
            assert await sandbox.read_file(path) == b"contents"

    async def test_workspace_persists_across_acquisitions(self, provider):
        """A workspace must survive between tool calls.

        Each tool acquires the sandbox independently, so an implementation that
        creates fresh isolation per acquisition silently loses every file the
        previous tool wrote.
        """
        async with provider.acquire(workspace_id="persistent") as first:
            await first.write_file(f"{first.workdir}/kept.txt", b"still here")

        async with provider.acquire(workspace_id="persistent") as second:
            assert await second.read_file(f"{second.workdir}/kept.txt") == b"still here"

    async def test_workspaces_are_isolated_from_each_other(self, provider):
        async with provider.acquire(workspace_id="ws-a") as sandbox:
            await sandbox.write_file(f"{sandbox.workdir}/secret.txt", b"a")

        async with provider.acquire(workspace_id="ws-b") as sandbox:
            with pytest.raises(_MISSING_FILE):
                await sandbox.read_file(f"{sandbox.workdir}/secret.txt")

    async def test_files_can_be_enumerated(self, provider):
        """Without enumeration a workspace cannot be harvested, and anything the
        agent produced dies with the sandbox."""
        async with provider.acquire(workspace_id="listing") as sandbox:
            await sandbox.write_file(f"{sandbox.workdir}/a.txt", b"a")
            await sandbox.write_file(f"{sandbox.workdir}/nested/b.txt", b"b")
            found = await sandbox.list_files()
        names = {p.rsplit("/", 1)[-1] for p in found}
        assert {"a.txt", "b.txt"} <= names
        await provider.release("listing")

    async def test_release_discards_the_workspace(self, provider):
        """A workspace must be endable, or every finished run leaks one."""
        async with provider.acquire(workspace_id="disposable") as sandbox:
            await sandbox.write_file(f"{sandbox.workdir}/tmp.txt", b"transient")

        await provider.release("disposable")

        async with provider.acquire(workspace_id="disposable") as sandbox:
            with pytest.raises(_MISSING_FILE):
                await sandbox.read_file(f"{sandbox.workdir}/tmp.txt")
        await provider.release("disposable")

    async def test_written_files_are_visible_to_commands(self, provider):
        async with provider.acquire(workspace_id="contract") as sandbox:
            path = f"{sandbox.workdir}/script.py"
            await sandbox.write_file(path, b"print(6*7)")
            result = await sandbox.exec(ExecRequest(command=("python3", path), timeout_s=60))
        assert result.stdout.strip() == "42"


__all__ = ["MemoryStoreContract", "ObjectStoreContract", "SandboxContract"]


class RestartableSandboxContract(SandboxContract):
    """For providers whose workspaces should survive the provider itself.

    A durable runtime can restart mid-run; if the sandbox forgets everything on
    restart, the agent silently loses work it believes it has done.
    """

    @abstractmethod
    def fresh_provider(self) -> Any:
        """A brand-new provider instance sharing the same backend."""

    async def test_workspace_survives_a_new_provider_instance(self, provider):
        async with provider.acquire(workspace_id="restart-ws") as sandbox:
            await sandbox.write_file(f"{sandbox.workdir}/kept.txt", b"survived")

        replacement = self.fresh_provider()
        async with replacement.acquire(workspace_id="restart-ws") as sandbox:
            assert await sandbox.read_file(f"{sandbox.workdir}/kept.txt") == b"survived"
