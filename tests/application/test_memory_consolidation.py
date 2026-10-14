"""Consolidation: extraction, supersession, reflection, and not oscillating."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from skynet.application import memory_consolidation as mc
from skynet.application.context import DefaultContextBuilder
from skynet.application.policy import AgentPolicy
from skynet.application.registry import InMemoryToolRegistry
from skynet.application.runner import InProcessAgentRunner
from skynet.domain.agent import MemoryConfig, ResourceLimits
from skynet.domain.decisions import Complete, ConsolidateMemory
from skynet.domain.events import MemoryConsolidated
from skynet.domain.ids import MemoryId, RunId
from skynet.domain.memory import (
    AddMemory,
    ConsolidationMode,
    InvalidateMemory,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    UpdateMemory,
)
from skynet.domain.messages import assistant, user
from skynet.domain.models import ModelParams
from skynet.domain.run import AgentStatus
from skynet.ports.agents import RunRequest
from skynet.testing.builders import build_services, make_context, make_run, make_spec, make_task
from skynet.testing.fakes import InMemoryMemoryStore, ScriptedModelProvider

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = MemoryScope.agent("demo")
PARAMS = ModelParams(model="test")


def existing(memory_id: str, content: str, scope: MemoryScope = SCOPE) -> MemoryRecord:
    return MemoryRecord(id=MemoryId(memory_id), scope=scope, content=content, created_at=T0)


# -- parsing: tolerant, and never a source of exceptions ----------------------


def operations_json(*operations: dict) -> str:
    return json.dumps({"operations": list(operations)})


def test_operations_are_read_from_plain_json() -> None:
    parsed = mc.parse_operations(operations_json({"kind": "add", "content": "a fact"}))
    assert parsed == (AddMemory(content="a fact"),)


def test_operations_are_read_out_of_a_code_fence() -> None:
    """Models fence JSON constantly; refusing it would be refusing most answers."""
    fenced = f"Sure!\n```json\n{operations_json({'kind': 'add', 'content': 'a fact'})}\n```\n"
    assert mc.parse_operations(fenced) == (AddMemory(content="a fact"),)


def test_operations_are_read_out_of_surrounding_prose() -> None:
    noisy = f"Here is what I found: {operations_json({'kind': 'add', 'content': 'a fact'})} Done."
    assert mc.parse_operations(noisy) == (AddMemory(content="a fact"),)


def test_a_bare_list_is_accepted() -> None:
    assert mc.parse_operations('[{"kind": "add", "content": "a fact"}]') != ()


def test_one_bad_entry_does_not_discard_the_good_ones() -> None:
    parsed = mc.parse_operations(
        operations_json({"kind": "nonsense"}, {"kind": "add", "content": "a fact"})
    )
    assert parsed == (AddMemory(content="a fact"),)


def test_unreadable_output_is_no_operations_rather_than_an_error() -> None:
    """A model that rambles must degrade to "learned nothing", never to a crash."""
    for text in ("", "I am not going to do that", "{", "null", "```json\n{{{\n```"):
        assert mc.parse_operations(text) == ()


def test_an_invented_target_is_dropped() -> None:
    """A hallucinated id either does nothing or invalidates something real."""
    parsed = mc.parse_operations(
        operations_json({"kind": "invalidate", "target": "ghost"}), known_ids=[MemoryId("m1")]
    )
    assert parsed == ()


def test_a_known_target_survives() -> None:
    parsed = mc.parse_operations(
        operations_json({"kind": "invalidate", "target": "m1"}), known_ids=[MemoryId("m1")]
    )
    assert parsed == (InvalidateMemory(target=MemoryId("m1")),)


def test_a_flood_of_operations_is_capped() -> None:
    flood = operations_json(*({"kind": "add", "content": f"f{i}"} for i in range(50)))
    assert len(mc.parse_operations(flood)) == mc.MAX_OPERATIONS


# -- planning: derived ids are what make a retry safe -------------------------


def plan(*operations, candidates=(), key="k", **kwargs):
    return mc.plan_writes(operations, key=key, scope=SCOPE, at=T0, candidates=candidates, **kwargs)


def test_written_ids_derive_from_the_idempotency_key() -> None:
    written = plan(AddMemory(content="a"), AddMemory(content="b")).writes
    assert [record.id for record in written] == [MemoryId("k-0"), MemoryId("k-1")]


def test_replanning_the_same_pass_produces_the_same_ids() -> None:
    """This, and not bookkeeping, is what makes a consolidation retry idempotent."""
    first = plan(AddMemory(content="a")).writes
    second = plan(AddMemory(content="a")).writes
    assert [r.id for r in first] == [r.id for r in second]


def test_an_update_is_an_invalidate_then_an_add() -> None:
    old = existing("m1", "the office is in Berlin")
    result = plan(UpdateMemory(target=MemoryId("m1"), content="Munich"), candidates=(old,))
    assert [r.content for r in result.writes] == ["Munich"]
    assert result.invalidations == ((MemoryId("m1"), MemoryId("k-0")),)


def test_an_invalidation_names_no_replacement() -> None:
    old = existing("m1", "gone")
    result = plan(InvalidateMemory(target=MemoryId("m1")), candidates=(old,))
    assert result.invalidations == ((MemoryId("m1"), None),)
    assert result.writes == ()


def test_a_memory_in_a_scope_this_run_cannot_write_is_untouchable() -> None:
    """Retrieval spans several scopes so the model can reason about shared
    memories. Read-only access must not become read-and-destroy."""
    shared = existing("p1", "a team convention", scope=MemoryScope.project("acme"))
    result = plan(InvalidateMemory(target=MemoryId("p1")), candidates=(shared,))
    assert result.invalidations == ()
    assert result.rejected == 1


def test_a_ttl_becomes_an_expiry() -> None:
    written = plan(AddMemory(content="a", ttl_hours=24)).writes[0]
    assert written.expires_at == T0 + timedelta(hours=24)


def test_the_agent_default_ttl_applies_when_the_model_gives_none() -> None:
    written = plan(AddMemory(content="a"), default_ttl_hours=6).writes[0]
    assert written.expires_at == T0 + timedelta(hours=6)


def test_a_written_record_records_where_it_came_from() -> None:
    written = plan(AddMemory(content="a"), source_run_id=RunId("run-1")).writes[0]
    assert written.source_run_id == RunId("run-1")
    assert written.created_at == T0 and written.valid_from == T0


# -- the request --------------------------------------------------------------


def test_extraction_shows_the_transcript_and_the_existing_memories() -> None:
    request = mc.consolidation_request(
        ConsolidationMode.EXTRACT,
        objective="find the port",
        messages=(user("what port?"), assistant("5433")),
        candidates=(existing("m1", "the office is in Berlin"),),
        params=PARAMS,
    )
    body = request.messages[1].content
    assert "5433" in body and "m1" in body and "find the port" in body
    assert request.tools == ()


def test_reflection_shows_memories_but_no_transcript() -> None:
    """Its source is what the agent already believes, not what it just said."""
    request = mc.consolidation_request(
        ConsolidationMode.REFLECT,
        objective="",
        messages=(user("some chatter"),),
        candidates=(existing("m1", "a remembered fact"),),
        params=PARAMS,
    )
    body = request.messages[1].content
    assert "a remembered fact" in body
    assert "some chatter" not in body


# -- the policy gates ---------------------------------------------------------


def policy() -> AgentPolicy:
    return AgentPolicy(InMemoryToolRegistry(()), DefaultContextBuilder())


def finished_run(**memory_fields):
    spec = make_spec(memory=MemoryConfig(**memory_fields))
    return make_run(spec=spec).model_copy(
        update={"prompt": None, "messages": (user("q"), assistant("the answer"))}
    )


def ready(run):
    from skynet.domain.prompts import PromptRef, RenderedPrompt

    return run.model_copy(update={"prompt": RenderedPrompt(ref=PromptRef(name="p"), text="system")})


def test_a_finished_run_consolidates_before_completing() -> None:
    decision = policy().decide(ready(finished_run(consolidate=True)))
    assert isinstance(decision, ConsolidateMemory)
    assert decision.mode is ConsolidationMode.EXTRACT
    assert decision.through_index == 2


def test_a_finished_run_completes_once_it_has() -> None:
    run = ready(finished_run(consolidate=True)).model_copy(update={"consolidated_through": 2})
    assert isinstance(policy().decide(run), Complete)


def test_consolidation_is_off_by_default() -> None:
    assert isinstance(policy().decide(ready(finished_run())), Complete)


def test_reflection_fires_on_its_interval_and_only_once_per_turn() -> None:
    spec = make_spec(memory=MemoryConfig(reflect_every=2))
    base = ready(make_run(spec=spec)).model_copy(update={"messages": (user("q"),)})

    assert policy().decide(base.model_copy(update={"iteration": 1})) is not None
    assert not isinstance(
        policy().decide(base.model_copy(update={"iteration": 1})), ConsolidateMemory
    )
    at_two = base.model_copy(update={"iteration": 2})
    assert isinstance(policy().decide(at_two), ConsolidateMemory)
    assert not isinstance(
        policy().decide(at_two.model_copy(update={"reflected_at_iteration": 2})),
        ConsolidateMemory,
    )


def test_reflection_leaves_the_message_watermark_alone() -> None:
    """Advancing it would let reflection starve the extraction that matters."""
    spec = make_spec(memory=MemoryConfig(reflect_every=1))
    run = ready(make_run(spec=spec)).model_copy(
        update={
            "iteration": 1,
            "messages": (user("q"), assistant("a"), user("go on")),
            "consolidated_through": 1,
        }
    )
    decision = policy().decide(run)
    assert isinstance(decision, ConsolidateMemory)
    assert decision.mode is ConsolidationMode.REFLECT
    assert decision.through_index == 1, "reflection moved the extraction watermark"
    assert decision.through_index != len(run.messages)


def test_reflection_respects_the_consolidation_budget() -> None:
    spec = make_spec(
        memory=MemoryConfig(reflect_every=1), limits=ResourceLimits(max_consolidations=2)
    )
    run = ready(make_run(spec=spec)).model_copy(
        update={"iteration": 1, "messages": (user("q"),), "consolidation_count": 2}
    )
    assert not isinstance(policy().decide(run), ConsolidateMemory)


# -- end to end, through the real executor ------------------------------------


async def consolidating_run(model_output: str, *, seeded=(), **memory_fields):
    store = InMemoryMemoryStore()
    for record in seeded:
        await store.put(record)
    memory_fields.setdefault("consolidate", True)

    def rule(request):
        from skynet.testing.builders import says

        if request.messages and "long-term memory" in request.messages[0].content:
            return says(model_output)
        return says("the answer is 5433")

    services = build_services(model=ScriptedModelProvider(rule=rule), memory=store)
    spec = make_spec(memory=MemoryConfig(**memory_fields))
    run = await InProcessAgentRunner(services).run(
        RunRequest(spec=spec, task=make_task(), context=make_context())
    )
    events = [e for e in services.sink.events if isinstance(e, MemoryConsolidated)]
    return run, store, events


async def test_extraction_writes_what_the_model_asked_for() -> None:
    run, store, events = await consolidating_run(
        operations_json({"kind": "add", "content": "the port is 5433", "memory_kind": "semantic"})
    )
    assert run.status is AgentStatus.COMPLETED
    assert len(events) == 1 and len(events[0].written) == 1
    stored = await store.search(MemoryQuery(scopes=(SCOPE,), text=""))
    assert [hit.record.content for hit in stored] == ["the port is 5433"]
    assert stored[0].record.kind is MemoryKind.SEMANTIC


async def test_an_update_leaves_the_old_belief_readable() -> None:
    """The Zep guarantee: contradiction stamps a record, it never deletes one."""
    old = existing("m1", "the office is in Berlin")
    run, store, events = await consolidating_run(
        operations_json({"kind": "update", "target": "m1", "content": "the office is in Munich"}),
        seeded=(old,),
    )
    superseded = await store.get(SCOPE, MemoryId("m1"))
    assert superseded is not None, "the old record was deleted, not invalidated"
    assert superseded.valid_to is not None
    assert superseded.superseded_by == events[0].written[0]

    live = await store.search(
        MemoryQuery(scopes=(SCOPE,), text="office", as_of=run.context.deadline)
    )
    assert "Munich" in " ".join(hit.record.content for hit in live)


async def test_the_earlier_belief_is_still_retrievable_as_of_before() -> None:
    old = existing("m1", "the office is in Berlin")
    _, store, _ = await consolidating_run(
        operations_json({"kind": "update", "target": "m1", "content": "Munich"}), seeded=(old,)
    )
    before = await store.search(
        MemoryQuery(scopes=(SCOPE,), text="office", as_of=T0 + timedelta(seconds=1))
    )
    assert [hit.record.id for hit in before] == [MemoryId("m1")]


async def test_a_run_consolidates_once_and_then_completes() -> None:
    run, _, events = await consolidating_run(operations_json({"kind": "add", "content": "a"}))
    assert run.status is AgentStatus.COMPLETED
    assert len(events) == 1, "the completion gate re-opened"


async def test_zero_operations_still_completes_the_run() -> None:
    run, _, events = await consolidating_run(operations_json())
    assert run.status is AgentStatus.COMPLETED
    assert len(events) == 1 and events[0].operations == ()
    assert run.consolidated_through >= len(run.messages) - 1


async def test_unreadable_consolidation_output_still_completes_the_run() -> None:
    run, _, events = await consolidating_run("I would rather not")
    assert run.status is AgentStatus.COMPLETED
    assert len(events) == 1 and events[0].operations == ()


async def test_a_consolidation_that_raises_still_completes_the_run() -> None:
    """Memory must never be able to fail a run."""
    store = InMemoryMemoryStore()

    def rule(request):
        from skynet.testing.builders import says

        if request.messages and "long-term memory" in request.messages[0].content:
            raise RuntimeError("the memory model exploded")
        return says("the answer")

    services = build_services(model=ScriptedModelProvider(rule=rule), memory=store)
    spec = make_spec(memory=MemoryConfig(consolidate=True))
    run = await InProcessAgentRunner(services).run(
        RunRequest(spec=spec, task=make_task(), context=make_context())
    )
    events = [e for e in services.sink.events if isinstance(e, MemoryConsolidated)]
    assert run.status is AgentStatus.COMPLETED
    assert len(events) == 1 and events[0].error is not None
    assert run.model_failures == 0


async def test_running_the_same_pass_twice_writes_one_record() -> None:
    """Idempotency by derived id, so a retry cannot duplicate a memory."""
    from skynet.application.consolidator import Consolidator
    from skynet.testing.builders import says

    store = InMemoryMemoryStore()
    model = ScriptedModelProvider(
        rule=lambda _: says(operations_json({"kind": "add", "content": "a fact"}))
    )
    services = build_services(model=model, memory=store)
    decision = ConsolidateMemory(
        mode=ConsolidationMode.EXTRACT,
        scope=SCOPE,
        through_index=1,
        params=PARAMS,
        idempotency_key="run-1:extract:0:1",
    )
    consolidator = Consolidator(store, model, services.clock)
    for _ in range(2):
        await consolidator.run(decision, run_id=RunId("run-1"))
    stored = await store.search(MemoryQuery(scopes=(SCOPE,), text="", limit=50))
    assert len(stored) == 1, "the retry wrote a second copy"
