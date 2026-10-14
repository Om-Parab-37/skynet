"""API wiring.

The API depends on the ``AgentRunner`` port and nothing else about execution, so
the same routes serve both the in-process engine and Temporal. Which one is
active is a settings lookup made once at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from skynet.adapters.runs.sink import BroadcastEventSink, CompositeEventSink, LoggingEventSink
from skynet.application.runner import InProcessAgentRunner
from skynet.application.services import RuntimeServices
from skynet.config.container import build_event_log, build_services
from skynet.config.settings import Settings
from skynet.domain.ids import RunId
from skynet.ports.agents import RunRequest

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Everything the HTTP layer needs, assembled once at startup."""

    settings: Settings
    services: RuntimeServices
    runner: Any
    events: BroadcastEventSink
    requests: dict[RunId, RunRequest] = field(default_factory=dict)
    handles: dict[RunId, Any] = field(default_factory=dict)

    async def close(self) -> None:
        await self.services.tracer.flush()
        if hasattr(self.services.memory, "aclose"):
            await self.services.memory.aclose()


async def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings()
    broadcast = BroadcastEventSink()

    services = build_services(settings)
    # When the worker runs elsewhere, the shared log -- not this process's
    # memory -- is the source of truth for events.
    shared = build_event_log(settings)
    # Compose the broadcast sink in so the streaming endpoint sees every event
    # without the runtime knowing an HTTP API exists.
    services = RuntimeServices(
        **{
            **{f: getattr(services, f) for f in services.__dataclass_fields__},
            "sink": CompositeEventSink([LoggingEventSink(), broadcast]),
        }
    )

    if settings.execution.engine == "temporal":
        from skynet.adapters.temporal.runner import TemporalAgentRunner
        from skynet.adapters.temporal.worker import connect

        client = await connect(settings)
        runner: Any = TemporalAgentRunner(
            client,
            task_queue=settings.temporal.task_queue,
            execution_timeout_s=settings.temporal.workflow_timeout_s,
        )
        logger.info("execution engine: temporal (%s)", settings.temporal.host)
    else:
        runner = InProcessAgentRunner(services)
        logger.info("execution engine: in-process")

    return Runtime(
        settings=settings,
        services=services,
        runner=runner,
        events=shared if shared is not None else broadcast,
    )


__all__ = ["Runtime", "build_runtime"]
