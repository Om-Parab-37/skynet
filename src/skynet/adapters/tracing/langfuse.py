"""AI tracing backed by Langfuse.

Langfuse v4 is OpenTelemetry-based and models observations by *type*, which maps
cleanly onto the harness's own span kinds: an agent run becomes an ``agent``
observation, a model call a ``generation``, a capability a ``tool``. The result
is the hierarchy the runtime already describes -- agent -> model -> tool ->
sub-agent -- rather than a flat list.

Nesting works two ways, because the runtime runs two ways:

* **In-process**, ``start_as_current_observation`` puts the span on the OTel
  context, so anything started inside it nests automatically -- including work
  fanned out with ``asyncio.gather``, which inherits the context at task
  creation.
* **Across processes** (a Temporal activity is not in the workflow's context),
  the domain's own ``TraceContext`` is passed explicitly, so a run's spans still
  land in one trace.

Two properties are enforced throughout: the SDK is optional, and a tracing
failure never fails a run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import ExitStack, asynccontextmanager
from typing import Any

from skynet.adapters.tracing.noop import _NullSpan
from skynet.domain.agent import TraceContext
from skynet.domain.errors import ConfigurationError
from skynet.domain.ids import SpanId, TraceId
from skynet.ports.tracing import SpanKind

logger = logging.getLogger(__name__)

# The harness's span kinds expressed in Langfuse's observation vocabulary.
_AS_TYPE: Mapping[SpanKind, str] = {
    SpanKind.AGENT: "agent",
    SpanKind.GENERATION: "generation",
    SpanKind.TOOL: "tool",
    SpanKind.INTERNAL: "span",
}

# Langfuse inherits OTel's id formats; anything else is rejected outright.
_TRACE_ID_LENGTH = 32
_SPAN_ID_LENGTH = 16


def _is_otel_id(value: str | None, length: int) -> bool:
    return bool(value) and len(value) == length and all(c in "0123456789abcdef" for c in value)


class _LangfuseSpan:
    def __init__(self, handle: Any, parent: TraceContext) -> None:
        self._handle = handle
        self._parent = parent
        self._metadata: dict[str, Any] = {}

    def set_attributes(self, **attributes: Any) -> None:
        """Record attributes, promoting the ones Langfuse models natively.

        The port stays deliberately generic -- a tracer receives plain
        attributes -- but token counts and model names are first-class in
        Langfuse and drive its cost and latency views, so the *adapter*
        recognises them. Everything else stays metadata.
        """
        self._metadata.update(attributes)
        update: dict[str, Any] = {"metadata": self._metadata}

        if model := attributes.get("model"):
            update["model"] = str(model)

        usage = {
            key: int(attributes[source])
            for key, source in (("input", "input_tokens"), ("output", "output_tokens"))
            if isinstance(attributes.get(source), int)
        }
        if usage:
            usage["total"] = sum(usage.values())
            update["usage_details"] = usage

        _safe(lambda: self._handle.update(**update))

    def set_output(self, output: Any) -> None:
        _safe(lambda: self._handle.update(output=output))

    def record_error(self, error: Any) -> None:
        _safe(lambda: self._handle.update(level="ERROR", status_message=str(error)[:2000]))

    @property
    def context(self) -> TraceContext:
        """Linkage a child span (or a child agent) should inherit."""
        trace_id = _safe(lambda: self._handle.trace_id) or self._parent.trace_id
        span_id = _safe(lambda: self._handle.id) or self._parent.parent_span_id
        return TraceContext(
            trace_id=TraceId(trace_id) if trace_id else None,
            parent_span_id=SpanId(span_id) if span_id else None,
            baggage=self._parent.baggage,
        )


def _safe(action: Any, default: Any = None) -> Any:
    """Observability must never take a run down with it."""
    try:
        return action()
    except Exception as exc:  # noqa: BLE001
        logger.debug("langfuse call failed: %s", exc)
        return default


class LangfuseTracer:
    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def create(cls, *, public_key: str, secret_key: str, host: str) -> LangfuseTracer:
        try:
            from langfuse import Langfuse
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigurationError(
                "langfuse is not installed; install the 'langfuse' extra"
            ) from exc
        return cls(Langfuse(public_key=public_key, secret_key=secret_key, host=host))

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        input: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        parent_context = parent or TraceContext()
        manager = _safe(
            lambda: self._client.start_as_current_observation(
                name=name,
                as_type=_AS_TYPE.get(kind, "span"),
                input=input,
                metadata={"kind": str(kind), **dict(attributes or {})},
                trace_context=_trace_context(parent_context),
                end_on_exit=True,
            )
        )
        if manager is None:
            # Degrade to a no-op span rather than failing the run.
            yield _NullSpan(parent_context)
            return

        with ExitStack() as stack:
            handle = stack.enter_context(manager)
            # Trace-level attributes ride on the domain's baggage, which is
            # already the slot for metadata that spans a whole run. Applying it
            # inside every span means a run reads correctly even when its spans
            # are produced by different processes.
            if attributes_cm := _propagate(parent_context.baggage):
                _safe(lambda: stack.enter_context(attributes_cm))

            span = _LangfuseSpan(handle, parent_context)
            try:
                yield span
            except Exception as exc:
                span.record_error(exc)
                raise

    async def flush(self) -> None:
        _safe(self._client.flush)


# Baggage keys the adapter understands. Anything else stays plain metadata.
_TRACE_NAME = "trace_name"
_SESSION_ID = "session_id"
_USER_ID = "user_id"


def _propagate(baggage: Mapping[str, str]) -> Any:
    """Turn domain baggage into Langfuse trace-level attributes."""
    if not baggage:
        return None
    try:
        from langfuse import propagate_attributes
    except Exception:  # noqa: BLE001 - older SDKs simply lack it
        return None

    known = {
        "trace_name": baggage.get(_TRACE_NAME),
        "session_id": baggage.get(_SESSION_ID),
        "user_id": baggage.get(_USER_ID),
    }
    extra = {k: v for k, v in baggage.items() if k not in {_TRACE_NAME, _SESSION_ID, _USER_ID}}
    return _safe(
        lambda: propagate_attributes(
            **{k: v for k, v in known.items() if v},
            metadata=extra or None,
        )
    )


def _has_ambient_span() -> bool:
    """Is an OTel span already active in this process?"""
    try:
        from opentelemetry import trace as otel_trace

        return otel_trace.get_current_span().get_span_context().is_valid
    except Exception:  # noqa: BLE001
        return False


def _trace_context(parent: TraceContext) -> dict[str, str] | None:
    """Explicit linkage -- used only when ambient context cannot do the job.

    Two regimes, and getting the precedence wrong flattens the tree:

    * In-process, an enclosing span is already on the OTel context and gives
      true parent/child nesting. Passing an explicit trace context here would
      *override* that and reparent the span to the trace root.
    * Across processes (a Temporal activity), there is no ambient span, so the
      domain's own ``TraceContext`` is the only thing tying the run together.

    Ambient therefore wins whenever it exists.
    """
    if _has_ambient_span():
        return None
    if not _is_otel_id(parent.trace_id, _TRACE_ID_LENGTH):
        return None
    context: dict[str, str] = {"trace_id": str(parent.trace_id)}
    if _is_otel_id(parent.parent_span_id, _SPAN_ID_LENGTH):
        context["parent_span_id"] = str(parent.parent_span_id)
    return context


__all__ = ["LangfuseTracer"]
