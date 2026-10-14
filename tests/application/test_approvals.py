"""Human-in-the-loop: a run parks, an approval arrives, the run continues."""

from __future__ import annotations

import asyncio

from skynet.application.runner import InProcessAgentRunner
from skynet.domain.ids import CallId
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
from skynet.testing.fakes import EchoTool, RecordingApprovalGate


def dangerous_services(**overrides):
    return build_services(
        responses=[calls(call("deploy", "c1", text="to production")), says("deployed")],
        tools=[EchoTool(name="deploy", requires_approval=True)],
        approvals=RecordingApprovalGate(),
        **overrides,
    )


async def start(services):
    runner = InProcessAgentRunner(services)
    handle = await runner.start(
        RunRequest(spec=make_spec(tools=("deploy",)), task=make_task(), context=make_context())
    )
    return runner, handle


async def test_run_suspends_awaiting_approval_then_completes_when_granted():
    services = dangerous_services()
    runner, handle = await start(services)

    # Let the run reach the approval point.
    for _ in range(50):
        await asyncio.sleep(0)
        if services.approvals.requests:
            break
    assert services.approvals.requests, "approval was never requested"
    assert "approval.requested" in services.sink.kinds()

    await handle.approve(CallId("c1"), granted=True)
    run = await asyncio.wait_for(handle.result(), timeout=2)

    assert run.status is AgentStatus.COMPLETED
    assert run.steps[0].tool_results[0].status is ToolStatus.OK
    # The tool ran exactly once -- resuming must not duplicate work.
    assert services.sink.kinds().count("tool.completed") == 1


async def test_denied_approval_produces_a_denied_result_not_an_execution():
    services = dangerous_services()
    runner, handle = await start(services)

    for _ in range(50):
        await asyncio.sleep(0)
        if services.approvals.requests:
            break

    await handle.approve(CallId("c1"), granted=False)
    run = await asyncio.wait_for(handle.result(), timeout=2)

    assert run.status is AgentStatus.COMPLETED
    result = run.steps[0].tool_results[0]
    assert result.status is ToolStatus.DENIED
    assert "denied by operator" in result.content


async def test_approval_suspension_is_visible_in_the_event_log():
    services = dangerous_services()
    runner, handle = await start(services)
    for _ in range(50):
        await asyncio.sleep(0)
        if services.approvals.requests:
            break
    await handle.approve(CallId("c1"), granted=True)
    await asyncio.wait_for(handle.result(), timeout=2)

    kinds = services.sink.kinds()
    assert kinds.index("approval.requested") < kinds.index("tool.completed")
