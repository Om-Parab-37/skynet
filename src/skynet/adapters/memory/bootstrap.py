"""Apply the memory schema, then exit.

Runs as a one-shot container from the application image, so the DDL lives in
exactly one place and cannot drift from the code that queries it. Safe to run
concurrently with anything: the migration takes an advisory lock, and it is
idempotent.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from skynet.adapters.memory.postgres import PostgresMemoryStore
from skynet.config.settings import Settings

logger = logging.getLogger("skynet.memory.bootstrap")


async def bootstrap(settings: Settings | None = None) -> int:
    settings = settings or Settings()
    if settings.memory.backend != "postgres":
        logger.info("memory backend is '%s'; nothing to migrate", settings.memory.backend)
        return 0

    store = PostgresMemoryStore(
        settings.memory.dsn,
        dimensions=settings.embedding.dimensions,
        statement_timeout_ms=settings.memory.statement_timeout_ms,
        min_size=1,
        max_size=1,
    )
    try:
        await store.ready()
        logger.info("memory schema is up to date at %s", _safe(settings.memory.dsn))
        return 0
    except Exception as exc:  # noqa: BLE001 - report clearly and fail the container
        logger.error("memory bootstrap failed: %s", exc)
        return 1
    finally:
        await store.aclose()


def _safe(dsn: str) -> str:
    """The DSN without its password -- these logs are read in shared terminals."""
    if "@" not in dsn:
        return dsn
    prefix, _, host = dsn.rpartition("@")
    scheme, _, credentials = prefix.partition("://")
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname).1s %(name)s: %(message)s")
    sys.exit(asyncio.run(bootstrap()))


if __name__ == "__main__":
    main()
