"""Command-line interface.

A thin front end over the same ``AgentRunner`` port the HTTP API uses -- which
is the point: interfaces are interchangeable because none of them contains
agent logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer

from skynet.application.runner import InProcessAgentRunner
from skynet.config.container import build_services
from skynet.config.settings import Settings
from skynet.domain.agent import AgentContext, TraceContext
from skynet.domain.ids import RunId, TaskId, TraceId
from skynet.domain.run import AgentRun, AgentStatus
from skynet.domain.task import Task
from skynet.ports.agents import RunRequest

app = typer.Typer(add_completion=False, help="Skynet agent harness")


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(levelname).1s %(name)s: %(message)s")


@app.command()
def run(
    objective: Annotated[str, typer.Argument(help="What the agent should accomplish")],
    agent: Annotated[str, typer.Option(help="Agent name from agents.yaml")] = "assistant",
    model: Annotated[str, typer.Option(help="Override the model")] = "",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run an agent to completion in-process."""
    settings = Settings()
    if model:
        settings.model.name = model
    if verbose:
        settings.tracing.backend = "console"
    _configure_logging("DEBUG" if verbose else settings.log_level)

    services = build_services(settings)
    spec = services.agents.get(agent)
    if model:
        spec = spec.model_copy(update={"model": spec.model.model_copy(update={"model": model})})

    ids = services.ids
    run_id = RunId(ids.new_id("run"))
    request = RunRequest(
        spec=spec,
        task=Task(id=TaskId(ids.new_id("task")), objective=objective),
        context=AgentContext(
            run_id=run_id,
            agent_name=agent,
            permissions=spec.default_permissions,
            trace=TraceContext(
                trace_id=TraceId(ids.new_trace_id()),
                baggage={"trace_name": f"agent.{agent}", "session_id": str(run_id)},
            ),
        ),
    )

    async def _execute():
        try:
            return await InProcessAgentRunner(services).run(request)
        finally:
            # Tracing backends batch in the background; without this the process
            # exits before anything is sent.
            await services.tracer.flush()

    result = asyncio.run(_execute())
    _report(result)
    raise typer.Exit(0 if result.status is AgentStatus.COMPLETED else 1)


@app.command()
def agents() -> None:
    """List the configured agents."""
    services = build_services(Settings())
    for spec in services.agents.all():
        typer.echo(f"{spec.name:14} {spec.description}")
        typer.echo(f"{'':14} tools: {', '.join(spec.tools) or 'none'}")
        if spec.sub_agents:
            typer.echo(f"{'':14} delegates to: {', '.join(spec.sub_agents)}")


@app.command()
def tools() -> None:
    """List the registered capabilities."""
    services = build_services(Settings())
    for name in services.registry.names():
        spec = services.registry.spec(name)
        assert spec is not None
        perms = ", ".join(sorted(spec.required_permissions)) or "none"
        typer.echo(f"{name:16} [{spec.kind}] requires: {perms}")
        typer.echo(f"{'':16} {spec.description.splitlines()[0]}")


def _report(run: AgentRun) -> None:
    typer.echo("")
    typer.secho(f"── {run.status.upper()} ".ljust(72, "─"), bold=True)
    for step in run.steps:
        if step.assistant.content.strip():
            typer.echo(f"  think  {step.assistant.content.strip()[:160]}")
        for called, result in zip(step.assistant.tool_calls, step.tool_results, strict=False):
            typer.echo(f"  tool   {called.name}({_brief(called.arguments)}) -> {result.status}")
            typer.echo(f"         {result.content.strip()[:160]}")
    typer.echo("")
    if run.result:
        typer.secho(run.result.output, fg=typer.colors.GREEN)
    elif run.last_error:
        typer.secho(f"{run.last_error.code}: {run.last_error.message}", fg=typer.colors.RED)
    typer.echo("")
    typer.echo(
        f"iterations={run.iteration} tools={run.tool_call_count} "
        f"tokens={run.usage.total} children={len(run.child_runs)}"
    )


def _brief(arguments: dict) -> str:
    parts = []
    for key, value in arguments.items():
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={text[:40]}{'...' if len(text) > 40 else ''}")
    return ", ".join(parts)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
