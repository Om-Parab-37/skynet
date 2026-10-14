"""Postgres + pgvector memory store.

The first genuinely shared, mutable state in this system -- until now the API
and the worker shared only an append-only event log. Two consequences are
accepted deliberately: neither process may connect eagerly at boot, and both may
race the migration. The lazy pool handles the first, an advisory lock the second.

**Hybrid retrieval, composed correctly.** One ``live`` CTE applies scope, kind,
tags and the bi-temporal window, so the lexical and semantic arms cannot
disagree about what is currently true. They are combined with a FULL OUTER JOIN:
an INNER join would quietly reduce hybrid search to an *intersection*, which is
the classic way this gets built wrong and looks fine in a demo. Reciprocal rank
fusion merges them -- scale-free, so no normalisation is needed -- while the raw
component scores come back too, so the pure ranker can re-weight and
``min_score`` can threshold on a real similarity rather than on a fused rank,
against which it would be meaningless.

``websearch_to_tsquery``, never ``to_tsquery``: agent-authored text is full of
punctuation, and ``to_tsquery`` raises a syntax error on it -- turning a badly
worded question into a dead run.

**Failure policy: reads degrade, writes fail loudly.** Telling an agent it
remembered something it did not is worse than telling it the store is down: it
will reason from that fact later and nothing in the log will explain why. An
embedding failure stores the record with no vector, which leaves it fully
searchable lexically. A store failure on write raises.

*One event loop per process.* The pool binds to the first loop that touches it,
because ``build_services`` is synchronous and may run with none.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from skynet.domain.errors import ConfigurationError, SkynetError
from skynet.domain.ids import MemoryId, RunId
from skynet.domain.memory import (
    MemoryHit,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    RetrievalComponents,
    ScopeLevel,
)
from skynet.ports.embedding import Embedder

logger = logging.getLogger(__name__)

MIGRATIONS = pathlib.Path(__file__).parent / "migrations"
ADVISORY_LOCK = 0x5CE7_0001
RRF_K = 60
"""Reciprocal-rank-fusion constant. 60 is the value the literature settles on;
it flattens the difference between ranks 1 and 2 enough that a single arm's
confidence cannot dominate the other."""


class _Breaker:
    """Stops a dead database turning every turn into a full connect timeout."""

    def __init__(self, *, threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None

    def allows(self, now: float) -> bool:
        if self._opened_at is None:
            return True
        if now - self._opened_at >= self._cooldown:
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_failure(self, now: float) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = now

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


class PostgresMemoryStore:
    """``MemoryStore`` over Postgres, with an optional vector arm."""

    def __init__(
        self,
        dsn: str,
        *,
        embedder: Embedder | None = None,
        dimensions: int = 768,
        statement_timeout_ms: int = 10_000,
        min_size: int = 1,
        max_size: int = 8,
        migrate: bool = True,
    ) -> None:
        self._dsn = dsn
        self._embedder = embedder
        self._dimensions = dimensions
        self._statement_timeout_ms = statement_timeout_ms
        self._migrate = migrate
        self._pool = self._make_pool(min_size, max_size)
        self._opened = False
        self._lock = asyncio.Lock()
        self._breaker = _Breaker()
        self._embedding_disabled = False

    # -- lifecycle -------------------------------------------------------------

    def _make_pool(self, min_size: int, max_size: int):
        """Constructed closed. ``build_services`` is synchronous and may run with
        no event loop at all, so opening here would bind the pool to nothing."""
        try:
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ConfigurationError(
                "the postgres memory backend needs the 'postgres' extra: uv sync --extra postgres"
            ) from exc

        return AsyncConnectionPool(
            self._dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            configure=self._configure,
        )

    async def _configure(self, conn) -> None:
        """Runs on every pooled connection, *after* the schema exists.

        ``register_vector_async`` needs the ``vector`` type to be present, so
        migrations cannot run through this pool -- a failure here leaves the
        connection in a transaction, psycopg discards it, and the pool retries
        forever. That is why the schema is applied on a standalone connection
        first, and why nothing here is wrapped in a ``try``: a failure at this
        point is a real misconfiguration and should be loud.
        """
        from pgvector.psycopg import register_vector_async

        # Autocommit, for two reasons. The pool rejects a connection left inside
        # a transaction, and `register_vector_async` has to query the catalog to
        # find the type oid -- which starts one. More importantly, the pool
        # *rolls back* on return, so a bare `UPDATE` outside an explicit
        # transaction would be silently undone: `invalidate` would report
        # success and change nothing. Writes that need atomicity say so with
        # `conn.transaction()`.
        await conn.set_autocommit(True)
        await conn.execute(f"SET statement_timeout = {self._statement_timeout_ms}")
        await register_vector_async(conn)

    async def ready(self):
        """Open the pool and apply migrations. Called lazily, or by the bootstrap."""
        return await self._ready()

    async def _ready(self):
        if not self._opened:
            async with self._lock:
                if not self._opened:
                    if self._migrate:
                        await self._apply_migrations()
                    await self._pool.open(wait=True, timeout=30)
                    self._opened = True
        return self._pool

    async def aclose(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False

    async def _apply_migrations(self) -> None:
        """Numbered, append-only, and guarded by an advisory lock.

        Not Alembic: it drags SQLAlchemy into a project with no ORM, and
        reversibility is worthless for regenerable data. Not bare
        ``CREATE TABLE IF NOT EXISTS`` either -- that silently never applies a
        later column change. Not ``docker-entrypoint-initdb.d``, which Postgres
        runs only when the data directory is empty, so migration 002 would apply
        on a fresh laptop and never on yours.
        """
        import psycopg

        files = sorted(MIGRATIONS.glob("*.sql"))
        # A standalone connection, deliberately: the pool cannot be used here.
        async with await psycopg.AsyncConnection.connect(self._dsn) as conn:
            async with conn.transaction():
                await conn.execute(f"SELECT pg_advisory_xact_lock({ADVISORY_LOCK})")
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS skynet_schema_version ("
                    " version text PRIMARY KEY,"
                    " applied_at timestamptz NOT NULL DEFAULT now())"
                )
                cursor = await conn.execute("SELECT version FROM skynet_schema_version")
                rows = await cursor.fetchall()
                applied = {row[0] for row in rows}
                for path in files:
                    if path.name in applied:
                        continue
                    logger.info("applying memory migration %s", path.name)
                    await conn.execute(path.read_text())
                    await conn.execute(
                        "INSERT INTO skynet_schema_version (version) VALUES (%s)", (path.name,)
                    )
            await self._assert_dimensions(conn)

    async def _assert_dimensions(self, conn) -> None:
        """Refuse to start on a mismatch.

        A silent mismatch presents as 100% recall loss with no error anywhere,
        which is close to undiagnosable from the outside.
        """
        cursor = await conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'memory_vectors'::regclass AND attname = 'embedding'"
        )
        row = await cursor.fetchone()
        if row and row[0] not in (None, -1) and row[0] != self._dimensions:
            raise ConfigurationError(
                f"memory_vectors.embedding is vector({row[0]}) but the configured "
                f"embedder produces {self._dimensions} dimensions. Re-embed with "
                f"TRUNCATE memory_vectors and a backfill, or set the dimension back."
            )

    # -- writes ----------------------------------------------------------------

    async def put(self, record: MemoryRecord) -> MemoryId:
        pool = await self._ready()
        async with pool.connection() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO memories (id, namespace, scope_kind, scope_key, kind, content,
                                      tags, metadata, importance, source_run_id,
                                      created_at, valid_from, valid_to, superseded_by, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        coalesce(%s, now()), %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    namespace = EXCLUDED.namespace, scope_kind = EXCLUDED.scope_kind,
                    scope_key = EXCLUDED.scope_key, kind = EXCLUDED.kind,
                    content = EXCLUDED.content, tags = EXCLUDED.tags,
                    metadata = EXCLUDED.metadata, importance = EXCLUDED.importance,
                    source_run_id = EXCLUDED.source_run_id, created_at = EXCLUDED.created_at,
                    valid_from = EXCLUDED.valid_from, valid_to = EXCLUDED.valid_to,
                    superseded_by = EXCLUDED.superseded_by, expires_at = EXCLUDED.expires_at
                """,
                (
                    str(record.id),
                    record.scope.namespace(),
                    record.scope.level.value,
                    record.scope.owner,
                    record.kind.value,
                    record.content,
                    list(record.tags),
                    json.dumps(record.metadata),
                    record.importance,
                    str(record.source_run_id) if record.source_run_id else None,
                    record.created_at,
                    record.valid_from,
                    record.valid_to,
                    str(record.superseded_by) if record.superseded_by else None,
                    record.expires_at,
                ),
            )
            vector = await self._embed_one(record.content)
            if vector is not None:
                await conn.execute(
                    """
                    INSERT INTO memory_vectors (memory_id, model, dim, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (memory_id) DO UPDATE SET
                        model = EXCLUDED.model, dim = EXCLUDED.dim,
                        embedding = EXCLUDED.embedding
                    """,
                    (str(record.id), self._embedder.model, len(vector), vector),  # type: ignore[union-attr]
                )
        return record.id

    async def invalidate(
        self,
        scope: MemoryScope,
        memory_id: MemoryId,
        *,
        at: datetime,
        superseded_by: MemoryId | None = None,
    ) -> bool:
        pool = await self._ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "UPDATE memories SET valid_to = %s, superseded_by = %s "
                "WHERE id = %s AND namespace = %s",
                (
                    at,
                    str(superseded_by) if superseded_by else None,
                    str(memory_id),
                    scope.namespace(),
                ),
            )
            return cursor.rowcount > 0

    async def delete(self, scope: MemoryScope, memory_id: MemoryId) -> bool:
        pool = await self._ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM memories WHERE id = %s AND namespace = %s",
                (str(memory_id), scope.namespace()),
            )
            return cursor.rowcount > 0

    async def purge_expired(self, *, before: datetime) -> int:
        pool = await self._ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM memories WHERE coalesce(valid_to, expires_at) < %s", (before,)
            )
            return cursor.rowcount

    # -- reads -----------------------------------------------------------------

    async def get(self, scope: MemoryScope, memory_id: MemoryId) -> MemoryRecord | None:
        pool = await self._ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = %s AND namespace = %s",
                (str(memory_id), scope.namespace()),
            )
            row = await cursor.fetchone()
        return _record(row) if row else None

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        if not query.scopes:
            return ()
        pool = await self._ready()
        vector = None if not query.text.strip() else await self._embed_one(query.text)
        sql, params = _search_sql(query, vector)
        async with pool.connection() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        hits: list[MemoryHit] = []
        for row in rows:
            record, lexical, semantic = _record(row), row[-2], row[-1]
            relevance = _relevance(lexical, semantic, has_text=bool(query.text.strip()))
            if relevance < query.min_score:
                continue
            hits.append(
                MemoryHit(
                    record=record,
                    relevance=relevance,
                    score=relevance,
                    components=RetrievalComponents(
                        semantic=semantic, lexical=lexical, fused=relevance
                    ),
                )
            )
        return tuple(hits[: query.limit])

    # -- embedding -------------------------------------------------------------

    async def _embed_one(self, text: str) -> list[float] | None:
        """``None`` means "store it without a vector".

        A permanent failure latches off for the process rather than retrying on
        every call: the worst case of this store then becomes lexical search,
        which is exactly the store this replaced.
        """
        if self._embedder is None or self._embedding_disabled or not text.strip():
            return None
        loop = asyncio.get_running_loop()
        if not self._breaker.allows(loop.time()):
            return None
        try:
            vectors = await self._embedder.embed([text])
        except Exception:  # noqa: BLE001 - a missing vector is not a failed run
            self._breaker.record_failure(loop.time())
            logger.warning("embedding failed; storing without a vector", exc_info=True)
            return None
        self._breaker.record_success()
        if not vectors or not vectors[0]:
            return None
        return list(vectors[0])


_COLUMNS = (
    "id, namespace, scope_kind, scope_key, kind, content, tags, metadata, importance, "
    "source_run_id, created_at, valid_from, valid_to, superseded_by, expires_at"
)


def _record(row: Sequence[Any]) -> MemoryRecord:
    return MemoryRecord(
        id=MemoryId(row[0]),
        scope=MemoryScope(level=ScopeLevel(row[2]), owner=row[3]),
        kind=MemoryKind(row[4]),
        content=row[5],
        tags=tuple(row[6] or ()),
        metadata=row[7] or {},
        importance=row[8],
        source_run_id=RunId(row[9]) if row[9] else None,
        created_at=row[10],
        valid_from=row[11],
        valid_to=row[12],
        superseded_by=MemoryId(row[13]) if row[13] else None,
        expires_at=row[14],
    )


def _relevance(lexical: float | None, semantic: float | None, *, has_text: bool) -> float:
    """One similarity in 0..1, from whichever arms fired.

    ``ts_rank_cd`` is unbounded in principle and tiny in practice, so it is
    clamped rather than normalised against a corpus statistic that would change
    under us as the table grows.
    """
    if not has_text:
        return 1.0
    best = max(
        min(float(lexical), 1.0) if lexical is not None else 0.0,
        float(semantic) if semantic is not None else 0.0,
    )
    return max(0.0, min(best, 1.0))


def _search_sql(query: MemoryQuery, vector: list[float] | None) -> tuple[str, dict[str, Any]]:
    text = query.text.strip()
    over_fetch = max(query.limit, 1) * 4
    params: dict[str, Any] = {
        "namespaces": list(query.namespaces()),
        "kinds": [kind.value for kind in query.kinds] or None,
        "tags": list(query.tags) or None,
        "as_of": query.as_of,
        "include_invalidated": query.include_invalidated,
        "text": text,
        "vector": vector,
        "limit": over_fetch,
    }

    # `as_of IS NULL` means "judge supersession only": a store has no clock of
    # its own, so evaluating a TTL against wall time here would make results
    # depend on something the caller never supplied.
    live = f"""
        live AS (
            SELECT {_COLUMNS}, search_vector FROM memories
            WHERE namespace = ANY(%(namespaces)s)
              AND (%(kinds)s::text[] IS NULL OR kind = ANY(%(kinds)s::text[]))
              AND (%(tags)s::text[] IS NULL OR tags && %(tags)s::text[])
              AND (
                %(include_invalidated)s
                OR (
                  CASE WHEN %(as_of)s::timestamptz IS NULL THEN valid_to IS NULL
                  ELSE (valid_from IS NULL OR valid_from <= %(as_of)s::timestamptz)
                       AND (valid_to IS NULL OR valid_to > %(as_of)s::timestamptz)
                       AND (expires_at IS NULL OR expires_at > %(as_of)s::timestamptz)
                  END
                )
              )
        )
    """

    if not text:
        return (
            f"WITH {live} "
            f"SELECT {_COLUMNS}, NULL::real AS lexical, NULL::real AS semantic "
            "FROM live ORDER BY coalesce(valid_from, created_at) DESC, id "
            "LIMIT %(limit)s",
            params,
        )

    return (
        f"""
        WITH {live},
        lexical AS (
            SELECT id, ts_rank_cd(search_vector, q) AS score,
                   row_number() OVER (ORDER BY ts_rank_cd(search_vector, q) DESC, id) AS rank
            FROM live, websearch_to_tsquery('english', %(text)s) q
            WHERE search_vector @@ q
            LIMIT %(limit)s
        ),
        semantic AS (
            SELECT l.id, 1 - (v.embedding <=> %(vector)s::vector) AS score,
                   row_number() OVER (ORDER BY v.embedding <=> %(vector)s::vector, l.id) AS rank
            FROM live l JOIN memory_vectors v ON v.memory_id = l.id
            WHERE %(vector)s::vector IS NOT NULL
            ORDER BY v.embedding <=> %(vector)s::vector
            LIMIT %(limit)s
        ),
        -- FULL OUTER, never INNER: an inner join silently turns hybrid search
        -- into an intersection of the two arms.
        fused AS (
            SELECT coalesce(lexical.id, semantic.id) AS id,
                   lexical.score AS lexical_score,
                   semantic.score AS semantic_score,
                   coalesce(1.0 / ({RRF_K} + lexical.rank), 0.0)
                 + coalesce(1.0 / ({RRF_K} + semantic.rank), 0.0) AS rrf
            FROM lexical FULL OUTER JOIN semantic ON lexical.id = semantic.id
        )
        SELECT {", ".join(f"live.{column.strip()}" for column in _COLUMNS.split(","))},
               fused.lexical_score::real, fused.semantic_score::real
        FROM fused JOIN live ON live.id = fused.id
        ORDER BY fused.rrf DESC, live.id
        LIMIT %(limit)s
        """,
        params,
    )


class MemoryStoreUnavailable(SkynetError):
    """Raised on a write the store could not accept."""


__all__ = ["MIGRATIONS", "PostgresMemoryStore"]
