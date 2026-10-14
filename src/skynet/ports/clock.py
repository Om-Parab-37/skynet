"""Time and identity ports.

These exist so the core never calls ``datetime.now()`` or ``uuid4()`` directly.
Under Temporal the workflow supplies ``workflow.now()`` and ``workflow.uuid4()``,
which is what makes replay deterministic; in tests they supply fixed values,
which is what makes event logs byte-stable and assertable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class IdGenerator(Protocol):
    def new_id(self, prefix: str = "") -> str: ...

    def new_trace_id(self) -> str:
        """A 32-character hex trace id.

        Separate from ``new_id`` because tracing backends inherit OpenTelemetry's
        id format and reject anything else, whereas ``new_id`` deliberately
        produces readable identifiers.
        """
        ...


__all__ = ["Clock", "IdGenerator"]
