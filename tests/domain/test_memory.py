"""The memory domain model: scopes, bi-temporality, and the operation union."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from skynet.domain.ids import MemoryId, RunId
from skynet.domain.memory import (
    MAX_WORKING_MEMORIES,
    AddMemory,
    InvalidateMemory,
    MemoryHit,
    MemoryKind,
    MemoryOperation,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    ScopeLevel,
    UpdateMemory,
    merge_hits,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def record(**fields) -> MemoryRecord:
    fields.setdefault("id", MemoryId("m1"))
    fields.setdefault("scope", MemoryScope.agent("demo"))
    fields.setdefault("content", "a fact")
    return MemoryRecord(**fields)


# -- scopes -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "rendered"),
    [
        (MemoryScope.run(RunId("run-1")), "run:run-1"),
        (MemoryScope.agent("demo"), "agent:demo"),
        (MemoryScope.project("skynet"), "project:skynet"),
        (MemoryScope.user("ada"), "user:ada"),
    ],
)
def test_namespaces_render_as_level_colon_owner(scope: MemoryScope, rendered: str) -> None:
    assert scope.namespace() == rendered


def test_parse_round_trips_a_namespace() -> None:
    scope = MemoryScope.project("skynet")
    assert MemoryScope.parse(scope.namespace()) == scope


def test_parse_keeps_colons_in_the_owner() -> None:
    """Run ids and user handles contain colons; only the first one delimits."""
    assert MemoryScope.parse("user:acme:ada").owner == "acme:ada"


@pytest.mark.parametrize("bad", ["agent", "agent:", ":demo", "nonsense:demo"])
def test_parse_rejects_malformed_namespaces(bad: str) -> None:
    with pytest.raises(ValueError):
        MemoryScope.parse(bad)


def test_scope_precedence_runs_most_specific_first() -> None:
    levels = [ScopeLevel.RUN, ScopeLevel.AGENT, ScopeLevel.PROJECT, ScopeLevel.USER]
    assert [level.precedence for level in levels] == sorted(level.precedence for level in levels)


def test_scopes_are_frozen_and_hashable() -> None:
    scope = MemoryScope.agent("demo")
    assert scope in {MemoryScope.agent("demo")}
    with pytest.raises(ValidationError):
        scope.owner = "other"


# -- bi-temporality -----------------------------------------------------------


def test_a_plain_record_is_always_live() -> None:
    assert record().is_live(T0) is True


def test_a_record_is_not_live_before_it_became_true() -> None:
    fact = record(valid_from=T0)
    assert fact.is_live(T0 - DAY) is False
    assert fact.is_live(T0) is True


def test_a_record_stops_being_live_at_valid_to() -> None:
    fact = record(valid_from=T0, valid_to=T0 + DAY)
    assert fact.is_live(T0 + DAY - timedelta(seconds=1)) is True
    assert fact.is_live(T0 + DAY) is False


def test_a_record_stops_being_live_at_its_ttl() -> None:
    fact = record(expires_at=T0 + DAY)
    assert fact.is_live(T0) is True
    assert fact.is_live(T0 + DAY) is False


def test_importance_is_bounded() -> None:
    with pytest.raises(ValidationError):
        record(importance=1.5)


# -- queries ------------------------------------------------------------------


def test_a_query_renders_every_namespace_it_covers() -> None:
    query = MemoryQuery(scopes=(MemoryScope.agent("demo"), MemoryScope.project("skynet")))
    assert query.namespaces() == ("agent:demo", "project:skynet")


def test_the_query_signature_ignores_scope_order_and_text_casing() -> None:
    """It identifies a retrieval, so cosmetic differences must not fork it."""
    agent, project = MemoryScope.agent("demo"), MemoryScope.project("skynet")
    first = MemoryQuery(scopes=(agent, project), text="Pangolins ")
    second = MemoryQuery(scopes=(project, agent), text="pangolins")
    assert first.signature() == second.signature()


def test_the_query_signature_separates_different_retrievals() -> None:
    scopes = (MemoryScope.agent("demo"),)
    assert (
        MemoryQuery(scopes=scopes, text="a").signature()
        != MemoryQuery(scopes=scopes, text="b").signature()
    )
    assert (
        MemoryQuery(scopes=scopes, text="a").signature()
        != MemoryQuery(scopes=scopes, text="a", kinds=(MemoryKind.EPISODIC,)).signature()
    )


# -- consolidation operations -------------------------------------------------


def test_operations_round_trip_through_the_discriminated_union() -> None:
    adapter = TypeAdapter(MemoryOperation)
    for operation in (
        AddMemory(content="a fact"),
        UpdateMemory(target=MemoryId("m1"), content="a better fact"),
        InvalidateMemory(target=MemoryId("m1"), reason="contradicted"),
    ):
        assert adapter.validate_python(operation.model_dump()) == operation


def test_there_is_no_delete_operation() -> None:
    """Deleting would break the bi-temporal guarantee: contradiction invalidates."""
    with pytest.raises(ValidationError):
        TypeAdapter(MemoryOperation).validate_python({"kind": "delete", "target": "m1"})


# -- merging recalls ----------------------------------------------------------


def scored(memory_id: str, score: float) -> MemoryHit:
    return MemoryHit(record=record(id=MemoryId(memory_id)), relevance=score, score=score)


def test_merging_accumulates_two_recalls() -> None:
    """Recall fires more than once now; the second must not clobber the first."""
    merged = merge_hits([scored("m1", 0.5)], [scored("m2", 0.4)])
    assert [h.record.id for h in merged] == [MemoryId("m1"), MemoryId("m2")]


def test_merging_dedupes_keeping_the_better_score() -> None:
    merged = merge_hits([scored("m1", 0.1)], [scored("m1", 0.9)])
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merging_orders_equal_scores_by_id() -> None:
    """Replay stability: the prompt must not depend on arrival order."""
    forwards = merge_hits([scored("m1", 0.5)], [scored("m2", 0.5)])
    backwards = merge_hits([scored("m2", 0.5)], [scored("m1", 0.5)])
    assert [h.record.id for h in forwards] == [h.record.id for h in backwards]


def test_merging_caps_the_working_set() -> None:
    hits = [scored(f"m{index:02d}", index / 100) for index in range(40)]
    assert len(merge_hits(hits, [])) == MAX_WORKING_MEMORIES
