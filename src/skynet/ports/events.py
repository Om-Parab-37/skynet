"""Event publication port.

Where runtime observability (as distinct from AI tracing) plugs in: logging,
SSE streaming, metrics, or a message bus are all implementations of this.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from skynet.domain.events import RunEvent
from skynet.domain.ids import RunId


@runtime_checkable
class EventSink(Protocol):
    async def publish(self, event: RunEvent) -> None: ...


@runtime_checkable
class EventStream(Protocol):
    """Subscription side, used by the API's streaming endpoint."""

    def subscribe(self, run_id: RunId) -> AsyncIterator[RunEvent]: ...


__all__ = ["EventSink", "EventStream"]
