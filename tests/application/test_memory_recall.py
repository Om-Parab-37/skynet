"""Recall through the real executor: scopes, temporal filtering, ranking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from skynet.application.runner import InProcessAgentRunner
from skynet.domain.agent import MemoryConfig
from skynet.domain.events import MemoryRecalled
from skynet.domain.ids import MemoryId
from skynet.domain.memory import MemoryRecord, MemoryScope
from skynet.ports.agents import RunRequest
from skynet.testing.builders import build_services, make_context, make_spec, make_task, says
from skynet.testing.fakes import InMemoryMemoryStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def remembered(memory_id: str, content: str, scope: MemoryScope, **fields) -> MemoryRecord:
    fields.setdefault("created_at", T0)
    return MemoryRecord(id=MemoryId(memory_id), scope=scope, content=content, **fields)


async def recall(memory: MemoryConfig, records=(), **variables: str):
    store = InMemoryMemoryStore()
    for record in records:
        await store.put(record)
    services = build_services(responses=[says("done")], memory=store)
    run = await InProcessAgentRunner(services).run(
        RunRequest(
            spec=make_spec(memory=memory),
            task=make_task(objective="what do we know about pangolins?"),
            context=make_context(variables=dict(variables)),
        )
    )
    recalls = [e for e in services.sink.events if isinstance(e, MemoryRecalled)]
    return run, recalls


async def test_recall_reads_the_agents_own_memories() -> None:
    run, recalls = await recall(
        MemoryConfig(auto_recall=True),
        [remembered("m1", "pangolins eat ants", MemoryScope.agent("demo"))],
    )
    assert len(recalls) == 1
    assert [hit.record.id for hit in run.memories] == [MemoryId("m1")]


async def test_recall_does_not_reach_into_another_agents_memories() -> None:
    _, recalls = await recall(
        MemoryConfig(auto_recall=True),
        [remembered("m1", "pangolins eat ants", MemoryScope.agent("other"))],
    )
    assert recalls[0].hits == ()


async def test_recall_reads_a_shared_scope_once_granted() -> None:
    shared = remembered("m1", "pangolins eat ants", MemoryScope.project("acme"))
    without_grant, _ = await recall(MemoryConfig(auto_recall=True), [shared], project="acme")
    with_grant, _ = await recall(
        MemoryConfig(auto_recall=True, share_project=True), [shared], project="acme"
    )
    assert without_grant.memories == ()
    assert [hit.record.id for hit in with_grant.memories] == [MemoryId("m1")]


async def test_recall_skips_memories_that_have_expired() -> None:
    """The executor reads the clock and hands it to the store as ``as_of``;
    without that wiring a TTL would be decoration."""
    scope = MemoryScope.agent("demo")
    _, recalls = await recall(
        MemoryConfig(auto_recall=True),
        [
            remembered("m1", "pangolins eat ants", scope, expires_at=T0 - DAY),
            remembered("m2", "pangolins have scales", scope),
        ],
    )
    assert [hit.record.id for hit in recalls[0].hits] == [MemoryId("m2")]


async def test_recall_skips_memories_that_were_contradicted() -> None:
    scope = MemoryScope.agent("demo")
    _, recalls = await recall(
        MemoryConfig(auto_recall=True),
        [
            remembered("m1", "pangolins eat ants", scope, valid_to=T0 - DAY),
            remembered("m2", "pangolins eat termites", scope),
        ],
    )
    assert [hit.record.id for hit in recalls[0].hits] == [MemoryId("m2")]


async def test_recalled_hits_arrive_ranked() -> None:
    """The executor ranks, so the reducer and the prompt see a fused score."""
    scope = MemoryScope.agent("demo")
    _, recalls = await recall(
        MemoryConfig(auto_recall=True),
        [
            remembered("m1", "pangolins eat ants", scope, importance=0.1),
            remembered("m2", "pangolins eat ants", scope, importance=0.9),
        ],
    )
    hits = recalls[0].hits
    assert [hit.record.id for hit in hits] == [MemoryId("m2"), MemoryId("m1")]
    assert all(hit.score > 0.0 for hit in hits)


async def test_recall_honours_the_configured_limit() -> None:
    scope = MemoryScope.agent("demo")
    records = [remembered(f"m{i}", "pangolins eat ants", scope) for i in range(10)]
    _, recalls = await recall(MemoryConfig(auto_recall=True, recall_limit=3), records)
    assert len(recalls[0].hits) == 3


async def test_no_recall_without_the_policy() -> None:
    _, recalls = await recall(
        MemoryConfig(), [remembered("m1", "pangolins eat ants", MemoryScope.agent("demo"))]
    )
    assert recalls == []
