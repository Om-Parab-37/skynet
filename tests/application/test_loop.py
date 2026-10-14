"""End-to-end runs against fakes.

Assertions are made on the event log, because that log is what the runtime
actually guarantees: it is the audit trail, the API's stream, and the thing
Temporal replays.
"""

from __future__ import annotations

from skynet.application.runner import InProcessAgentRunner
from skynet.domain.agent import ResourceLimits
from skynet.domain.ids import CallId
from skynet.domain.messages import Role
from skynet.domain.run import AgentStatus, new_run, replay
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
from skynet.testing.fakes import (
    BigOutputTool,
    EchoTool,
    FailingTool,
    FlakyModelProvider,
    ScriptedModelProvider,
    SlowTool,
)


async def execute(services, spec=None, context=None, task=None):
    runner = InProcessAgentRunner(services)
    return await runner.run(
        RunRequest(
            spec=spec or make_spec(),
            task=task or make_task(),
            context=context or make_context(),
        )
    )


def kinds(services):
    return services.sink.kinds()


async def test_single_turn_answer_completes():
    services = build_services(responses=[says("the answer is 42")])
    run = await execute(services)

    assert run.status is AgentStatus.COMPLETED
    assert run.result.output == "the answer is 42"
    assert kinds(services) == (
        "run.started",
        "prompt.resolved",
        "model.responded",
        "run.completed",
    )


async def test_tool_call_then_answer():
    services = build_services(
        responses=[calls(call("echo", text="hello")), says("done: hello")],
        tools=[EchoTool()],
    )
    run = await execute(services, spec=make_spec(tools=("echo",)))

    assert run.status is AgentStatus.COMPLETED
    assert run.result.output == "done: hello"
    assert kinds(services) == (
        "run.started",
        "prompt.resolved",
        "model.responded",
        "tool.completed",
        "model.responded",
        "run.completed",
    )
    assert run.messages[-2].role is Role.TOOL
    assert run.messages[-2].content == "hello"


async def test_the_model_is_shown_the_tool_result():
    model = ScriptedModelProvider([calls(call("echo", text="observed")), says("ok")])
    services = build_services(model=model, tools=[EchoTool()])
    await execute(services, spec=make_spec(tools=("echo",)))

    second_request = model.requests[1]
    tool_messages = [m for m in second_request.messages if m.role is Role.TOOL]
    assert tool_messages and tool_messages[0].content == "observed"


async def test_parallel_tools_fold_in_call_order_not_completion_order():
    """Determinism under fan-out: the slowest call is first and must stay first."""
    services = build_services(
        responses=[
            calls(
                call("slow_a", "c1"),
                call("slow_b", "c2"),
                call("slow_c", "c3"),
            ),
            says("all three done"),
        ],
        tools=[
            SlowTool("slow_a", delay=0.05),
            SlowTool("slow_b", delay=0.02),
            SlowTool("slow_c", delay=0.001),
        ],
    )
    run = await execute(services, spec=make_spec(tools=("slow_a", "slow_b", "slow_c")))

    assert run.status is AgentStatus.COMPLETED
    results = run.steps[0].tool_results
    assert [r.call_id for r in results] == [CallId("c1"), CallId("c2"), CallId("c3")]


async def test_a_crashing_tool_becomes_an_observation_and_the_run_recovers():
    services = build_services(
        responses=[calls(call("boom", "c1")), says("recovered")],
        tools=[FailingTool(message="disk on fire")],
    )
    run = await execute(services, spec=make_spec(tools=("boom",)))

    assert run.status is AgentStatus.COMPLETED
    assert run.result.output == "recovered"
    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.ERROR
    assert "disk on fire" in result.content


async def test_unknown_tool_name_is_answered_with_the_valid_names():
    services = build_services(
        responses=[calls(call("teleport", "c1")), says("used echo instead")],
        tools=[EchoTool()],
    )
    run = await execute(services, spec=make_spec(tools=("echo",)))

    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.NOT_FOUND
    assert "echo" in result.content
    assert run.status is AgentStatus.COMPLETED


async def test_invalid_arguments_are_reported_back_to_the_model():
    from skynet.application.function_tool import tool

    @tool(name="divide")
    async def divide(numerator: int, denominator: int) -> str:
        """Divide two integers."""
        return str(numerator // denominator)

    services = build_services(
        responses=[calls(call("divide", "c1", numerator="ten", denominator=2)), says("fixed")],
        tools=[divide],
    )
    run = await execute(services, spec=make_spec(tools=("divide",)))

    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.ERROR
    assert "invalid arguments" in result.content


async def test_tool_lacking_permission_is_denied_not_executed():
    from skynet.domain.permissions import PermissionSet

    services = build_services(
        responses=[calls(call("echo", "c1", text="hi")), says("denied, moving on")],
        tools=[EchoTool(required_permissions=frozenset({"root"}))],
    )
    run = await execute(
        services,
        spec=make_spec(tools=("echo",)),
        context=make_context(permissions=PermissionSet.of("terminal")),
    )
    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.DENIED
    assert "root" in result.content


async def test_oversized_tool_output_is_truncated_and_stored_as_an_artifact():
    services = build_services(
        responses=[calls(call("big", "c1")), says("summarised")],
        tools=[BigOutputTool(size=50_000)],
    )
    run = await execute(services, spec=make_spec(tools=("big",)))

    result = run.steps[0].tool_results[0]
    assert result.truncated is True
    assert len(result.content) < 50_000
    assert result.artifacts and result.artifacts[0].size == 50_000
    stored = await services.objects.get(result.artifacts[0])
    assert len(stored) == 50_000


async def test_retryable_model_failure_is_retried_then_succeeds():
    inner = ScriptedModelProvider([says("second time lucky")])
    services = build_services(model=FlakyModelProvider(inner, fail_times=1, retryable=True))
    run = await execute(services)

    assert run.status is AgentStatus.COMPLETED
    assert kinds(services) == (
        "run.started",
        "prompt.resolved",
        "model.failed",
        "model.responded",
        "run.completed",
    )


async def test_non_retryable_model_failure_fails_the_run():
    inner = ScriptedModelProvider([says("never reached")])
    services = build_services(model=FlakyModelProvider(inner, fail_times=1, retryable=False))
    run = await execute(services)

    assert run.status is AgentStatus.FAILED
    assert run.last_error is not None


async def test_iteration_budget_ends_the_run_as_failed():
    """A model that will not stop must still be bounded."""
    services = build_services(
        model=ScriptedModelProvider(rule=lambda _: calls(call("echo", "c1", text="again"))),
        tools=[EchoTool()],
    )
    spec = make_spec(tools=("echo",), limits=ResourceLimits(max_iterations=3))
    run = await execute(services, spec=spec)

    assert run.status is AgentStatus.FAILED
    assert "max_iterations" in run.last_error.code
    assert run.iteration == 3


async def test_tool_call_budget_ends_the_run_as_failed():
    services = build_services(
        model=ScriptedModelProvider(rule=lambda _: calls(call("echo", "c1", text="again"))),
        tools=[EchoTool()],
    )
    spec = make_spec(tools=("echo",), limits=ResourceLimits(max_tool_calls=2, max_iterations=50))
    run = await execute(services, spec=spec)

    assert run.status is AgentStatus.FAILED
    assert "max_tool_calls" in run.last_error.code


async def test_run_survives_reconstruction_from_its_event_log():
    """The Temporal durability contract, verified without Temporal.

    Replaying the log into a fresh run must produce exactly the state the live
    run reached. If this holds, a worker can crash and resume mid-flight.
    """
    services = build_services(
        responses=[calls(call("echo", "c1", text="observed")), says("finished")],
        tools=[EchoTool()],
    )
    spec = make_spec(tools=("echo",))
    context = make_context()
    task = make_task()
    live = await execute(services, spec=spec, context=context, task=task)

    rebuilt = replay(
        new_run(spec=spec, task=task, context=context),
        services.sink.events,
    )

    assert rebuilt.status == live.status
    assert rebuilt.result == live.result
    assert rebuilt.messages == live.messages
    assert rebuilt.usage == live.usage
    assert rebuilt == live


async def test_partial_replay_then_continue_reaches_the_same_result():
    """Recovery from a crash halfway through: rebuild from a prefix, carry on."""
    services = build_services(
        responses=[calls(call("echo", "c1", text="x")), says("finished")],
        tools=[EchoTool()],
    )
    spec = make_spec(tools=("echo",))
    context = make_context()
    task = make_task()
    live = await execute(services, spec=spec, context=context, task=task)

    events = list(services.sink.events)
    for cut in range(1, len(events)):
        partial = replay(new_run(spec=spec, task=task, context=context), events[:cut])
        assert replay(partial, events[cut:]) == live
