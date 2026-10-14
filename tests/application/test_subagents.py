"""Delegation: hierarchy, authority narrowing, and failure containment."""

from __future__ import annotations

from skynet.application.runner import InProcessAgentRunner
from skynet.capabilities.delegate import DelegateTool
from skynet.domain.agent import ResourceLimits
from skynet.domain.permissions import PermissionSet
from skynet.domain.run import AgentStatus
from skynet.domain.tools import ToolStatus
from skynet.ports.agents import RunRequest
from skynet.testing.builders import (
    build_services,
    call,
    calls,
    make_context,
    make_spec,
    make_task,
    says,
)
from skynet.testing.fakes import EchoTool, ScriptedModelProvider


def delegating_spec(**kwargs):
    kwargs.setdefault("tools", ("delegate",))
    kwargs.setdefault("sub_agents", ("researcher",))
    return make_spec("supervisor", **kwargs)


async def run_with(services, spec=None, context=None):
    runner = InProcessAgentRunner(services)
    return await runner.run(
        RunRequest(
            spec=spec or delegating_spec(),
            task=make_task(),
            context=context or make_context(),
        )
    )


async def test_parent_delegates_and_receives_the_child_result():
    """One model provider serves both agents; each gets its own turn."""
    script = [
        calls(call("delegate", "c1", agent="researcher", objective="find the answer")),
        says("the researcher says: 42"),  # child agent's turn
        says("final: 42"),  # parent resumes with the child's result
    ]
    # The child runs nested inside the parent's delegate call, so it consumes
    # the second scripted response.
    services = build_services(
        model=ScriptedModelProvider([script[0], script[1], script[2]]),
        tools=[DelegateTool()],
        agents=[make_spec("researcher")],
    )
    run = await run_with(services)

    assert run.status is AgentStatus.COMPLETED
    assert run.result.output == "final: 42"

    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.OK
    assert result.content == "the researcher says: 42"

    assert len(run.child_runs) == 1
    assert "child.started" in services.sink.kinds()


async def test_child_permissions_can_only_narrow():
    captured = {}

    class Recorder(EchoTool):
        async def invoke(self, call, ctx):
            captured["child"] = ctx.permissions
            return await super().invoke(call, ctx)

    child_spec = make_spec(
        "researcher",
        tools=("echo",),
        default_permissions=PermissionSet.of("terminal", "root"),
    )
    services = build_services(
        model=ScriptedModelProvider(
            [
                calls(call("delegate", "c1", agent="researcher", objective="look")),
                calls(call("echo", "c2", text="child ran")),
                says("child done"),
                says("parent done"),
            ]
        ),
        tools=[DelegateTool(), Recorder()],
        agents=[child_spec],
    )
    parent_context = make_context(permissions=PermissionSet.of("terminal", "spawn_agent"))
    run = await run_with(services, context=parent_context)

    assert run.status is AgentStatus.COMPLETED
    # "root" was in the child's spec but never in the parent's grants, so the
    # intersection must not contain it.
    assert captured["child"].grants == frozenset({"terminal"})


async def test_delegation_to_an_unlisted_agent_is_denied():
    services = build_services(
        responses=[
            calls(call("delegate", "c1", agent="hacker", objective="exfiltrate")),
            says("not allowed, finishing"),
        ],
        tools=[DelegateTool()],
        agents=[make_spec("researcher")],
    )
    run = await run_with(services)

    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.DENIED
    assert "not delegable" in result.content
    assert run.status is AgentStatus.COMPLETED


async def test_delegation_depth_is_bounded():
    services = build_services(
        responses=[
            calls(call("delegate", "c1", agent="researcher", objective="deeper")),
            says("finishing"),
        ],
        tools=[DelegateTool()],
        agents=[make_spec("researcher")],
    )
    spec = delegating_spec(limits=ResourceLimits(max_depth=1))
    run = await run_with(services, spec=spec, context=make_context(depth=1))

    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.DENIED
    assert "depth" in result.content


async def test_child_failure_is_contained_as_an_observation():
    services = build_services(
        model=ScriptedModelProvider(
            [
                calls(call("delegate", "c1", agent="researcher", objective="fail please")),
                # The child's model runs out of responses -> the child fails.
            ]
            + []
        ),
        tools=[DelegateTool()],
        agents=[make_spec("researcher")],
    )
    run = await run_with(services)

    # The parent survives; it is told the sub-agent failed rather than crashing.
    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.ERROR
    assert "researcher" in result.content
