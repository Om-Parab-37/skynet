"""Tracing that does nothing, and tracing that prints.

Having more than one trivial implementation is not redundancy: it is the
cheapest possible proof that the core genuinely does not depend on a tracing
backend.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from skynet.domain.agent import TraceContext
from skynet.ports.tracing import SpanKind

logger = logging.getLogger("skynet.trace")


class _NullSpan:
    def __init__(self, parent: TraceContext | None = None) -> None:
        self._parent = parent or TraceContext()

    def set_attributes(self, **attributes: Any) -> None:
        return None

    def set_output(self, output: Any) -> None:
        return None

    def record_error(self, error: Any) -> None:
        return None

    @property
    def context(self) -> TraceContext:
        return self._parent


class NoopTracer:
    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[_NullSpan]:
        yield _NullSpan(parent)

    async def flush(self) -> None:
        return None


class _LoggingSpan(_NullSpan):
    def __init__(self, name: str, kind: SpanKind, parent: TraceContext | None) -> None:
        super().__init__(parent)
        self._name = name
        self._kind = kind

    def set_output(self, output: Any) -> None:
        text = str(output)
        logger.info("  %s -> %s", self._name, text[:200].replace("\n", " "))

    def record_error(self, error: Any) -> None:
        logger.warning("  %s !! %s", self._name, error)


class ConsoleTracer:
    """Logs span boundaries. Useful during development, cheap in production."""

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[_LoggingSpan]:
        logger.info("[%s] %s", kind, name)
        yield _LoggingSpan(name, kind, parent)

    async def flush(self) -> None:
        return None


__all__ = ["ConsoleTracer", "NoopTracer"]
