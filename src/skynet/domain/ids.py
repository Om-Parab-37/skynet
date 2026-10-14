"""Identifier newtypes.

Distinct string aliases so that mixing a run id with a call id is a type error
rather than a silent bug. No generation logic lives here -- creating an id is a
non-deterministic side effect and belongs behind the ``IdGenerator`` port.
"""

from typing import NewType

RunId = NewType("RunId", str)
CallId = NewType("CallId", str)
TaskId = NewType("TaskId", str)
MemoryId = NewType("MemoryId", str)
SpanId = NewType("SpanId", str)
TraceId = NewType("TraceId", str)

__all__ = ["CallId", "MemoryId", "RunId", "SpanId", "TaskId", "TraceId"]
