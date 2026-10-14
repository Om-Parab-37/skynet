"""Ranking: the fusion the stores are deliberately not allowed to do."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from skynet.application.memory_ranking import (
    DEFAULT_HALF_LIFE_HOURS,
    RankingWeights,
    candidate_limit,
    rank,
    recency_decay,
    score_of,
)
from skynet.domain.ids import MemoryId
from skynet.domain.memory import MemoryHit, MemoryRecord, MemoryScope

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HALF_LIFE = timedelta(hours=DEFAULT_HALF_LIFE_HOURS)


def hit(memory_id: str, relevance: float = 0.5, **fields) -> MemoryHit:
    fields.setdefault("scope", MemoryScope.agent("demo"))
    fields.setdefault("content", "a fact")
    fields.setdefault("created_at", T0)
    record = MemoryRecord(id=MemoryId(memory_id), **fields)
    return MemoryHit(record=record, relevance=relevance)


# -- recency ------------------------------------------------------------------


def test_a_brand_new_memory_has_full_recency() -> None:
    assert recency_decay(T0, T0) == 1.0


def test_one_half_life_is_exactly_half() -> None:
    assert recency_decay(T0, T0 + HALF_LIFE) == 0.5


def test_two_half_lives_are_a_quarter() -> None:
    assert recency_decay(T0, T0 + 2 * HALF_LIFE) == 0.25


def test_an_undated_memory_sits_in_the_middle() -> None:
    """Not knowing when it was learned is not evidence either way."""
    assert recency_decay(None, T0) == 0.5


def test_a_future_memory_is_not_scored_above_a_new_one() -> None:
    """Clock skew must not manufacture a relevance bonus."""
    assert recency_decay(T0, T0 - HALF_LIFE) == 1.0


# -- monotonicity per component -----------------------------------------------


def test_score_rises_with_relevance() -> None:
    assert score_of(hit("m", relevance=0.9), T0) > score_of(hit("m", relevance=0.1), T0)


def test_score_rises_with_recency() -> None:
    older = hit("m", created_at=T0 - 4 * HALF_LIFE)
    newer = hit("m", created_at=T0)
    assert score_of(newer, T0) > score_of(older, T0)


def test_score_rises_with_importance() -> None:
    assert score_of(hit("m", importance=0.9), T0) > score_of(hit("m", importance=0.1), T0)


def test_a_zero_component_does_not_annihilate_the_score() -> None:
    """The reason this is a weighted sum and not a product.

    A memory written seconds ago whose similarity is merely mediocre must still
    be retrievable; under a product it would score exactly zero and vanish.
    """
    fresh_but_unmatched = hit("m", relevance=0.0, created_at=T0, importance=0.9)
    assert score_of(fresh_but_unmatched, T0) > 0.0


# -- scope precedence ---------------------------------------------------------


def test_scope_precedence_breaks_a_tie() -> None:
    private = hit("m1", relevance=0.5, scope=MemoryScope.agent("demo"))
    shared = hit("m2", relevance=0.5, scope=MemoryScope.user("ada"))
    assert score_of(private, T0) > score_of(shared, T0)


def test_scope_precedence_does_not_override_relevance() -> None:
    """A tiebreak, not a multiplier: an excellent shared memory still wins."""
    mediocre_private = hit("m1", relevance=0.2, scope=MemoryScope.agent("demo"))
    excellent_shared = hit("m2", relevance=0.95, scope=MemoryScope.user("ada"))
    assert score_of(excellent_shared, T0) > score_of(mediocre_private, T0)


# -- rank ---------------------------------------------------------------------


def test_rank_orders_by_score_and_truncates() -> None:
    ranked = rank([hit("m1", 0.1), hit("m2", 0.9), hit("m3", 0.5)], at=T0, limit=2)
    assert [h.record.id for h in ranked] == [MemoryId("m2"), MemoryId("m3")]


def test_rank_writes_the_score_onto_the_hit() -> None:
    ranked = rank([hit("m1", 0.7)], at=T0)
    assert ranked[0].score == score_of(hit("m1", 0.7), T0)
    assert ranked[0].relevance == 0.7


def test_equal_scores_order_by_id() -> None:
    """Replay stability. Without this two equal memories can swap between a
    live run and its replay, which changes the prompt and so the answer."""
    forwards = rank([hit("m1", 0.5), hit("m2", 0.5)], at=T0)
    backwards = rank([hit("m2", 0.5), hit("m1", 0.5)], at=T0)
    assert [h.record.id for h in forwards] == [MemoryId("m1"), MemoryId("m2")]
    assert [h.record.id for h in backwards] == [h.record.id for h in forwards]


def test_rank_drops_memories_no_longer_believed() -> None:
    """Re-checked even though the store filters: a store that forgets is a bug,
    and it must not be one that puts a retracted fact in front of the model."""
    live = hit("m1", 0.5)
    retracted = hit("m2", 0.9, valid_to=T0 - timedelta(hours=1))
    assert [h.record.id for h in rank([live, retracted], at=T0)] == [MemoryId("m1")]


def test_rank_can_be_asked_for_the_history() -> None:
    retracted = hit("m2", 0.9, valid_to=T0 - timedelta(hours=1))
    assert rank([retracted], at=T0, include_invalidated=True) != ()


def test_custom_weights_change_what_wins() -> None:
    relevant_but_old = hit("m1", 0.9, created_at=T0 - 8 * HALF_LIFE)
    fresh_but_vague = hit("m2", 0.3, created_at=T0)
    by_relevance = rank(
        [relevant_but_old, fresh_but_vague],
        at=T0,
        weights=RankingWeights(relevance=1.0, recency=0.0, importance=0.0),
    )
    by_recency = rank(
        [relevant_but_old, fresh_but_vague],
        at=T0,
        weights=RankingWeights(relevance=0.0, recency=1.0, importance=0.0),
    )
    assert by_relevance[0].record.id == MemoryId("m1")
    assert by_recency[0].record.id == MemoryId("m2")


def test_candidates_are_over_fetched() -> None:
    """Fusion reorders, so asking for exactly ``limit`` would drop the winner."""
    assert candidate_limit(8) > 8
    assert candidate_limit(0) >= 1
