"""Tracing port.

The core opens spans; it never learns which backend records them. Span nesting
(agent -> model -> tool -> sub-agent) is expressed through ``TraceContext``,
which is a domain type, so the hierarchy survives swapping the backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from skynet.domain.agent import TraceContext
from skynet.domain.errors import ErrorInfo


class SpanKind(StrEnum):
    AGENT = "agent"
    GENERATION = "generation"
    TOOL = "tool"
    INTERNAL = "internal"


@runtime_checkable
class Span(Protocol):
    def set_attributes(self, **attributes: Any) -> None: ...

    def set_output(self, output: Any) -> None: ...

    def record_error(self, error: BaseException | ErrorInfo) -> None: ...

    @property
    def context(self) -> TraceContext: ...


@runtime_checkable
class Tracer(Protocol):
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[Span]:
        """Open a span.

        ``input`` is the operation's actual payload -- the messages sent to a
        model, the arguments passed to a tool -- and pairs with
        ``Span.set_output``. ``attributes`` is descriptive metadata about the
        operation. Keeping them separate is what lets a backend show a
        request/response pair rather than a bag of properties.
        """
        ...

    async def flush(self) -> None: ...


__all__ = ["Span", "SpanKind", "Tracer"]
