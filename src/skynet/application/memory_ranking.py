"""How retrieved memories are ordered. Pure, so it is identical everywhere.

The store does similarity; this module does ranking. Keeping fusion here rather
than in each adapter means the keyword fake and a pgvector table order results
the same way, which is precisely what lets the shared contract stop asserting
scoring behaviour at all.

    score = (w_r*relevance + w_t*recency + w_i*importance) / total  +  scope_bonus

A **weighted sum, not a product.** The product form zeroes a memory the moment
any one component is zero, which silently buries a brand-new memory whose
similarity happens to be mediocre -- the exact memory that just got written.

Scope precedence is a **tiebreak, not a multiplier**: an excellent project
memory should still beat a mediocre private one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from skynet.domain.memory import MemoryHit, ScopeLevel

DEFAULT_HALF_LIFE_HOURS = 72.0

SCOPE_BONUS: dict[ScopeLevel, float] = {
    ScopeLevel.RUN: 0.06,
    ScopeLevel.AGENT: 0.04,
    ScopeLevel.PROJECT: 0.02,
    ScopeLevel.USER: 0.0,
}

CANDIDATE_MULTIPLIER = 8
"""How far a store over-fetches before ranking truncates.

Fusion can reorder arbitrarily, so asking the store for exactly ``limit`` rows
would let a memory that ranks first be dropped before it is ever scored.
"""


def candidate_limit(limit: int) -> int:
    return max(1, limit) * CANDIDATE_MULTIPLIER


@dataclass(frozen=True, slots=True)
class RankingWeights:
    relevance: float = 1.0
    recency: float = 0.35
    importance: float = 0.25

    @property
    def total(self) -> float:
        return self.relevance + self.recency + self.importance


DEFAULT_WEIGHTS = RankingWeights()


def recency_decay(
    created_at: datetime | None,
    at: datetime,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
) -> float:
    """1.0 for a brand-new memory, exactly 0.5 at one half-life.

    An undated memory scores 0.5 rather than 0 or 1: not knowing when something
    was learned is not evidence that it is stale, nor that it is fresh.
    """
    if created_at is None or half_life_hours <= 0:
        return 0.5
    age_hours = max(0.0, (at - created_at).total_seconds() / 3600.0)
    return 0.5 ** (age_hours / half_life_hours)


def score_of(
    hit: MemoryHit,
    at: datetime,
    weights: RankingWeights = DEFAULT_WEIGHTS,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
) -> float:
    record = hit.record
    weighted = (
        weights.relevance * hit.relevance
        + weights.recency * recency_decay(record.created_at, at, half_life_hours)
        + weights.importance * record.importance
    )
    return weighted / weights.total + SCOPE_BONUS.get(record.scope.level, 0.0)


def rank(
    hits: Iterable[MemoryHit],
    *,
    at: datetime,
    limit: int | None = None,
    weights: RankingWeights = DEFAULT_WEIGHTS,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    include_invalidated: bool = False,
) -> tuple[MemoryHit, ...]:
    """Fuse, drop anything no longer believed, order, truncate.

    Liveness is re-checked even though the store filters too. A store that
    forgets to is a bug, and a memory the agent has stopped believing must never
    reach a prompt because of one.

    Ties break on record id. That is not cosmetic: without it two equal-scoring
    memories can swap order between a live run and its replay, which changes the
    prompt and therefore the model's answer.
    """
    scored = [
        hit.model_copy(update={"score": score_of(hit, at, weights, half_life_hours)})
        for hit in hits
        if include_invalidated or hit.record.is_live(at)
    ]
    scored.sort(key=lambda hit: (-hit.score, hit.record.id))
    return tuple(scored if limit is None else scored[:limit])


__all__ = [
    "CANDIDATE_MULTIPLIER",
    "DEFAULT_HALF_LIFE_HOURS",
    "DEFAULT_WEIGHTS",
    "SCOPE_BONUS",
    "RankingWeights",
    "candidate_limit",
    "rank",
    "recency_decay",
    "score_of",
]
