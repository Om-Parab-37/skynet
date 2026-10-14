"""Human-in-the-loop approval port.

The prototype ships auto-approve and manual gates. A real deployment implements
this against a UI, a chat app, or a ticketing system without touching the loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from skynet.domain.ids import CallId, RunId


@runtime_checkable
class ApprovalGate(Protocol):
    async def request(self, run_id: RunId, call_ids: Sequence[CallId], reason: str) -> None: ...


__all__ = ["ApprovalGate"]
