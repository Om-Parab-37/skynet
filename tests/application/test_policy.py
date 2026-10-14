"""The policy is a pure function, so these tests construct state and call it.

No mocks, no async, no infrastructure -- which is the point of separating the
decision from its execution.
"""

from __future__ import annotations

from skynet.application.context import DefaultContextBuilder
from skynet.application.policy import AgentPolicy
from skynet.application.registry import InMemoryToolRegistry
from skynet.domain.agent import FinishMode, MemoryConfig, ResourceLimits
from skynet.domain.decisions import (
    CallModel,
    Complete,
    Fail,
    InvokeTools,
    RecallMemory,
    RequestApproval,
    ResolvePrompt,
)
from skynet.domain.errors import ErrorInfo
from skynet.domain.ids import CallId
from skynet.domain.memory import MemoryScope
from skynet.domain.messages import Message, Role, assistant
from skynet.domain.permissions import PermissionSet
from skynet.domain.prompts import PromptRef, RenderedPrompt
from skynet.domain.tools import ToolKind, ToolSpec
from skynet.testing.builders import call, make_context, make_run, make_spec
from skynet.testing.fakes import EchoTool, RaisingTool

PROMPT = RenderedPrompt(ref=PromptRef(name="skynet/agent"), text="system text")


def policy(*tools) -> AgentPolicy:
    return AgentPolicy(InMemoryToolRegistry(tools), DefaultContextBuilder())


def ready(**kwargs):
    """A run that has already resolved its prompt."""
    run = make_run(**kwargs)
    return run.model_copy(update={"prompt": PROMPT})


def test_first_decision_is_to_resolve_the_prompt():
    decision = policy().decide(make_run())
    assert isinstance(decision, ResolvePrompt)
    assert decision.variables["agent_name"] == "demo"
    assert decision.variables["objective"] == "do the thing"


def recalling(**kwargs):
    kwargs.setdefault("memory", MemoryConfig(auto_recall=True))
    return ready(spec=make_spec(**kwargs))


def test_auto_recall_runs_before_the_first_model_call():
    decision = policy().decide(recalling())
    assert isinstance(decision, RecallMemory)
    assert decision.query.text == "do the thing"


def test_recall_asks_for_every_scope_the_run_may_read():
    """The decision carries the authority; the executor never widens it."""
    memory = MemoryConfig(auto_recall=True, share_project=True, project="acme")
    decision = policy().decide(recalling(memory=memory))
    assert MemoryScope.project("acme") in decision.query.scopes


def test_the_same_retrieval_is_not_issued_twice():
    run = recalling()
    issued = policy().decide(run).query.signature()
    assert isinstance(policy().decide(_recalled(run, issued)), CallModel)


def test_a_different_retrieval_is_still_allowed():
    """The watermark is the query, not the run: a boolean would forbid this."""
    run = recalling()
    assert isinstance(policy().decide(_recalled(run, "some:other|query||")), RecallMemory)


def _recalled(run, signature: str):
    return run.model_copy(update={"recalled_queries": (signature,)})


def test_model_is_offered_only_allowed_and_permitted_tools():
    spec = make_spec(tools=("echo", "secret"))
    run = ready(spec=spec)
    decision = policy(
        EchoTool(),
        RaisingTool(
            ToolSpec(
                name="secret",
                description="restricted",
                parameters={"type": "object", "properties": {}},
                required_permissions=frozenset({"root"}),
            )
        ),
    ).decide(run)

    assert isinstance(decision, CallModel)
    assert [t.name for t in decision.request.tools] == ["echo"]


def test_pending_tool_calls_are_drained_before_anything_else():
    run = ready().model_copy(update={"pending_calls": (call("echo", text="hi"),)})
    decision = policy(EchoTool()).decide(run)
    assert isinstance(decision, InvokeTools)
    assert decision.calls[0].name == "echo"


def test_tool_requiring_approval_suspends_the_run():
    tool = EchoTool(name="deploy", requires_approval=True)
    run = ready().model_copy(update={"pending_calls": (call("deploy", "c1"),)})
    decision = policy(tool).decide(run)
    assert isinstance(decision, RequestApproval)
    assert decision.calls[0].id == CallId("c1")


def test_granted_approval_lets_the_tool_run():
    tool = EchoTool(name="deploy", requires_approval=True)
    run = ready().model_copy(
        update={"pending_calls": (call("deploy", "c1"),), "approvals": {CallId("c1"): True}}
    )
    assert isinstance(policy(tool).decide(run), InvokeTools)


def test_denied_approval_still_invokes_so_the_executor_can_report_the_denial():
    tool = EchoTool(name="deploy", requires_approval=True)
    run = ready().model_copy(
        update={"pending_calls": (call("deploy", "c1"),), "approvals": {CallId("c1"): False}}
    )
    assert isinstance(policy(tool).decide(run), InvokeTools)


def test_assistant_turn_without_tool_calls_completes_under_native_mode():
    run = ready().model_copy(update={"messages": (assistant("the answer"),), "iteration": 1})
    decision = policy().decide(run)
    assert isinstance(decision, Complete)
    assert decision.result.output == "the answer"


def test_plain_chatter_does_not_complete_under_emulated_mode():
    run = ready(finish_mode=FinishMode.FINAL_ANSWER_TOOL).model_copy(
        update={"messages": (assistant("thinking out loud..."),), "iteration": 1}
    )
    assert isinstance(policy().decide(run), CallModel)


def test_final_answer_tool_completes_under_emulated_mode():
    final = RaisingTool(
        ToolSpec(
            name="final_answer",
            description="finish",
            parameters={"type": "object", "properties": {}},
            kind=ToolKind.CONTROL,
        )
    )
    message = assistant("", tool_calls=(call("final_answer", answer="42"),))
    run = ready(finish_mode=FinishMode.FINAL_ANSWER_TOOL).model_copy(
        update={"messages": (message,), "iteration": 1}
    )
    decision = policy(final).decide(run)
    assert isinstance(decision, Complete)
    assert decision.result.output == "42"


def test_retryable_model_failure_is_retried():
    run = ready().model_copy(
        update={"model_failures": 1, "last_error": ErrorInfo.of("net", "boom", retryable=True)}
    )
    assert isinstance(policy().decide(run), CallModel)


def test_non_retryable_model_failure_fails_the_run():
    run = ready().model_copy(
        update={"model_failures": 1, "last_error": ErrorInfo.of("bad", "nope")}
    )
    assert isinstance(policy().decide(run), Fail)


def test_model_failures_beyond_budget_fail_the_run():
    spec = make_spec(limits=ResourceLimits(max_model_failures=2))
    run = ready(spec=spec).model_copy(
        update={"model_failures": 3, "last_error": ErrorInfo.of("net", "boom", retryable=True)}
    )
    assert isinstance(policy().decide(run), Fail)


def test_parse_retry_triggers_one_corrective_call():
    run = ready().model_copy(update={"parse_retries": 1})
    assert isinstance(policy().decide(run), CallModel)


def test_parse_retries_beyond_budget_fail_the_run():
    spec = make_spec(limits=ResourceLimits(max_parse_retries=1))
    run = ready(spec=spec).model_copy(update={"parse_retries": 2})
    decision = policy().decide(run)
    assert isinstance(decision, Fail)
    assert decision.error.code == "tool_format"


def test_iteration_budget_fails_the_run():
    spec = make_spec(limits=ResourceLimits(max_iterations=3))
    run = ready(spec=spec).model_copy(update={"iteration": 3})
    decision = policy().decide(run)
    assert isinstance(decision, Fail)
    assert "max_iterations" in decision.error.code


def test_token_budget_fails_the_run():
    from skynet.domain.models import Usage

    spec = make_spec(limits=ResourceLimits(max_total_tokens=100))
    run = ready(spec=spec).model_copy(update={"usage": Usage(input_tokens=100)})
    decision = policy().decide(run)
    assert isinstance(decision, Fail)
    assert "max_total_tokens" in decision.error.code


def test_tool_call_budget_fails_the_run():
    spec = make_spec(limits=ResourceLimits(max_tool_calls=2))
    run = ready(spec=spec).model_copy(
        update={"pending_calls": (call("echo", "c1"),), "tool_call_count": 2}
    )
    decision = policy(EchoTool()).decide(run)
    assert isinstance(decision, Fail)
    assert "max_tool_calls" in decision.error.code


def test_narrowed_permissions_hide_tools_from_the_model():
    spec = make_spec(tools=("echo",))
    context = make_context(permissions=PermissionSet.of())
    run = ready(spec=spec, context=context)
    tool = EchoTool(required_permissions=frozenset({"terminal"}))
    decision = policy(tool).decide(run)
    assert isinstance(decision, CallModel)
    assert decision.request.tools == ()


# -- reasoning models ---------------------------------------------------------


def test_completion_falls_back_to_the_reasoning_channel():
    """Thinking models routinely finish with an empty `content`.

    Treating that as the answer silently loses the result, which is exactly what
    happened before `Message.text` existed.
    """
    thinking = Message(role=Role.ASSISTANT, content="", reasoning="The answer is 479001600.")
    run = ready().model_copy(update={"messages": (thinking,), "iteration": 1})

    decision = policy().decide(run)
    assert isinstance(decision, Complete)
    assert decision.result.output == "The answer is 479001600."


def test_content_wins_over_reasoning_when_both_are_present():
    both = Message(role=Role.ASSISTANT, content="Final: 42", reasoning="thinking out loud")
    run = ready().model_copy(update={"messages": (both,), "iteration": 1})

    decision = policy().decide(run)
    assert isinstance(decision, Complete)
    assert decision.result.output == "Final: 42"
