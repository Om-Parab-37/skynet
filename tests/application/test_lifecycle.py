"""Run lifecycle: resources acquired during a run must be released after it."""

from __future__ import annotations

from skynet.application.runner import InProcessAgentRunner
from skynet.capabilities.python_exec import CodeExecTool
from skynet.domain.run import AgentStatus
from skynet.ports.agents import RunRequest
from skynet.ports.sandbox import ExecResult
from skynet.testing.builders import (
    build_services,
    call,
    calls,
    make_context,
    make_spec,
    make_task,
    says,
)
from skynet.testing.fakes import FakeSandbox, FakeSandboxProvider, ScriptedModelProvider


def sandbox_services(responses, **overrides):
    sandboxes = FakeSandboxProvider(
        FakeSandbox(responses={"python3": ExecResult(exit_code=0, stdout="ok")})
    )
    services = build_services(
        responses=responses,
        tools=[CodeExecTool(sandboxes)],
        sandboxes=sandboxes,
        **overrides,
    )
    return services, sandboxes


async def run_once(services, spec=None):
    return await InProcessAgentRunner(services).run(
        RunRequest(
            spec=spec or make_spec(tools=("python_exec",)),
            task=make_task(),
            context=make_context(),
        )
    )


async def test_workspace_is_released_when_a_run_completes():
    services, sandboxes = sandbox_services(
        [calls(call("python_exec", code="print(1)")), says("done")]
    )
    run = await run_once(services)

    assert run.status is AgentStatus.COMPLETED
    assert sandboxes.released == ["run-1"], "the run's workspace was never released"


async def test_workspace_is_released_when_a_run_fails():
    """A failed run leaks just as badly as a successful one."""
    services, sandboxes = sandbox_services(
        [], model=ScriptedModelProvider(rule=lambda _: calls(call("python_exec", code="x")))
    )
    spec = make_spec(tools=("python_exec",))
    spec = spec.model_copy(update={"limits": spec.limits.model_copy(update={"max_iterations": 2})})
    run = await run_once(services, spec=spec)

    assert run.status is AgentStatus.FAILED
    assert sandboxes.released == ["run-1"]


async def test_a_run_without_a_sandbox_still_completes():
    """The sandbox is optional; its absence must not break the lifecycle."""
    services = build_services(responses=[says("no tools needed")])
    assert services.sandboxes is None
    run = await run_once(services, spec=make_spec())
    assert run.status is AgentStatus.COMPLETED
