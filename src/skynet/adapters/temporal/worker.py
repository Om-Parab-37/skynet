"""Temporal worker bootstrap.

Two things here are load-bearing:

* ``pydantic_data_converter`` on both client and worker, because every payload
  crossing the boundary is a pydantic model;
* the workflow sandbox is told to pass through ``pydantic`` and ``skynet`` --
  re-importing them inside the sandbox is both slow and, for pydantic's compiled
  core, unsafe. Our own code is deterministic by construction, so passing it
  through is sound.
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from skynet.adapters.temporal.activities import AgentActivities
from skynet.adapters.temporal.converter import DATA_CONVERTER
from skynet.adapters.temporal.workflow import AgentWorkflow
from skynet.config.container import build_services
from skynet.config.settings import Settings

logger = logging.getLogger(__name__)

PASSTHROUGH_MODULES = ("pydantic", "pydantic_core", "skynet", "yaml", "httpx")


async def connect(settings: Settings) -> Client:
    return await Client.connect(
        settings.temporal.host,
        namespace=settings.temporal.namespace,
        data_converter=DATA_CONVERTER,
    )


def sandbox_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(*PASSTHROUGH_MODULES)
    )


async def run_worker(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname).1s %(name)s: %(message)s"
    )

    client = await connect(settings)
    services = build_services(settings)
    activities = AgentActivities(services)

    # A crashed worker cannot clean up after itself, so the next one does it.
    sandboxes = services.sandboxes
    if sandboxes is not None and hasattr(sandboxes, "reap_abandoned"):
        reaped = await sandboxes.reap_abandoned()
        if reaped:
            logger.info("reaped %d abandoned sandbox container(s)", reaped)

    logger.info(
        "worker starting: queue=%s host=%s model=%s/%s sandbox=%s",
        settings.temporal.task_queue,
        settings.temporal.host,
        settings.model.provider,
        settings.model.name,
        settings.sandbox.backend,
    )

    worker = Worker(
        client,
        task_queue=settings.temporal.task_queue,
        workflows=[AgentWorkflow],
        activities=activities.all(),
        workflow_runner=sandbox_runner(),
    )
    try:
        await worker.run()
    finally:
        # Duck-typed, matching the `reap_abandoned` idiom above: a store that
        # holds no connections needs no closing, and the port stays five methods
        # wide rather than growing a lifecycle it mostly does not have.
        if hasattr(services.memory, "aclose"):
            await services.memory.aclose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
