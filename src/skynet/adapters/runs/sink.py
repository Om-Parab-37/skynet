"""Event sinks.

Runtime observability, as distinct from AI tracing. A broadcasting sink is what
the API's streaming endpoint subscribes to; composition means a deployment can
log, stream and persist the same events without any of them knowing about the
others.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence

from skynet.domain.events import RunEvent
from skynet.domain.ids import RunId
from skynet.ports.events import EventSink

logger = logging.getLogger("skynet.events")

_TERMINAL_KINDS = frozenset({"run.completed", "run.failed", "run.cancelled"})


class LoggingEventSink:
    async def publish(self, event: RunEvent) -> None:
        logger.info("%s %s seq=%s", event.run_id, event.kind, event.seq)


class CompositeEventSink:
    def __init__(self, sinks: Sequence[EventSink]) -> None:
        self._sinks = tuple(sinks)

    async def publish(self, event: RunEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.publish(event)
            except Exception as exc:  # noqa: BLE001 - a sink must not break a run
                logger.warning("event sink %s failed: %s", type(sink).__name__, exc)


class BroadcastEventSink:
    """Fan-out to live subscribers, with a replay buffer for late joiners."""

    def __init__(self, buffer_size: int = 1000) -> None:
        self._buffers: dict[RunId, list[RunEvent]] = {}
        self._subscribers: dict[RunId, list[asyncio.Queue[RunEvent | None]]] = {}
        self._buffer_size = buffer_size

    async def publish(self, event: RunEvent) -> None:
        buffer = self._buffers.setdefault(event.run_id, [])
        buffer.append(event)
        del buffer[: max(0, len(buffer) - self._buffer_size)]

        for queue in self._subscribers.get(event.run_id, []):
            queue.put_nowait(event)
        if event.kind in _TERMINAL_KINDS:
            for queue in self._subscribers.get(event.run_id, []):
                queue.put_nowait(None)

    def history(self, run_id: RunId) -> tuple[RunEvent, ...]:
        return tuple(self._buffers.get(run_id, ()))

    async def subscribe(self, run_id: RunId) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        # Replay what already happened so a subscriber never misses the start.
        for event in self.history(run_id):
            queue.put_nowait(event)
            if event.kind in _TERMINAL_KINDS:
                queue.put_nowait(None)

        self._subscribers.setdefault(run_id, []).append(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self._subscribers.get(run_id, []).remove(queue)


__all__ = ["BroadcastEventSink", "CompositeEventSink", "LoggingEventSink"]
