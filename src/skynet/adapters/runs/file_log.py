"""A run event log shared between processes.

Under the in-process engine the API and the runtime share memory, so an
in-memory sink is enough. Under Temporal they are different containers: the
worker produces events and the API serves them. Something durable has to sit
between them.

This is the smallest implementation that is actually correct: one
newline-delimited JSON file per run on a shared volume, appended by the worker
and tailed by the API. It is deliberately unglamorous -- the point is that
``EventSink`` and ``EventStream`` are the seam, so replacing this with Redis
Streams, Postgres LISTEN/NOTIFY or Kafka changes nothing above it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import TypeAdapter

from skynet.domain.events import RunEvent
from skynet.domain.ids import RunId

logger = logging.getLogger(__name__)

_EVENTS = TypeAdapter(RunEvent)
_TERMINAL_KINDS = frozenset({"run.completed", "run.failed", "run.cancelled"})


class FileEventLog:
    """Implements ``EventSink`` (write) and ``EventStream`` (read)."""

    def __init__(self, root: str | Path, poll_interval_s: float = 0.4) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._poll = poll_interval_s
        self._lock = asyncio.Lock()

    def _path(self, run_id: RunId) -> Path:
        return self._root / f"{run_id}.jsonl"

    async def publish(self, event: RunEvent) -> None:
        line = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
        async with self._lock:
            await asyncio.to_thread(self._append, self._path(event.run_id), line)

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def history(self, run_id: RunId) -> tuple[RunEvent, ...]:
        path = self._path(run_id)
        if not path.is_file():
            return ()
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(_EVENTS.validate_python(json.loads(line)))
            except Exception as exc:  # noqa: BLE001 - skip a torn trailing write
                logger.debug("skipping malformed event line: %s", exc)
        return tuple(events)

    async def subscribe(
        self, run_id: RunId, *, timeout_s: float = 3600.0
    ) -> AsyncIterator[RunEvent]:
        """Yield events as they appear, replaying anything already written.

        Replaying first means a subscriber that connects late still sees the
        whole run, which is what makes the stream useful for a UI.
        """
        seen = 0
        waited = 0.0
        while waited < timeout_s:
            events = self.history(run_id)
            for event in events[seen:]:
                yield event
                if event.kind in _TERMINAL_KINDS:
                    return
            if len(events) > seen:
                seen = len(events)
                waited = 0.0
                continue
            await asyncio.sleep(self._poll)
            waited += self._poll


__all__ = ["FileEventLog"]
