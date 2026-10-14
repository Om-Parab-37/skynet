"""Compaction: what gets folded, when, and why it terminates."""

from __future__ import annotations

from skynet.application import history_compaction as compaction
from skynet.application.context import DefaultContextBuilder
from skynet.application.policy import AgentPolicy
from skynet.application.registry import InMemoryToolRegistry
from skynet.application.runner import InProcessAgentRunner
from skynet.domain.agent import ResourceLimits
from skynet.domain.decisions import CallModel, CompactHistory
from skynet.domain.events import HistoryCompacted
from skynet.domain.ids import CallId
from skynet.domain.messages import ToolCall, assistant, tool_result, user
from skynet.domain.prompts import PromptRef, RenderedPrompt
from skynet.domain.run import COMPACTION_NOTE_PREFIX, AgentStatus
from skynet.ports.agents import RunRequest
from skynet.testing.builders import (
    build_services,
    call,
    calls,
    make_context,
    make_run,
    make_spec,
    make_task,
    says,
)
from skynet.testing.fakes import EchoTool, ScriptedModelProvider

PROMPT = RenderedPrompt(ref=PromptRef(name="skynet/agent"), text="system text")


# -- the pure helpers ---------------------------------------------------------


def test_size_counts_framing_as_well_as_content() -> None:
    """Underestimating framing is how a request that "fits" gets rejected."""
    assert compaction.estimated_chars((user("abc"),)) > 3


def test_the_slice_always_leaves_the_recent_turns() -> None:
    messages = tuple(user(str(index)) for index in range(10))
    assert compaction.compaction_slice(messages, keep_last=4) == 6


def test_nothing_is_folded_when_there_is_only_a_tail() -> None:
    messages = tuple(user(str(index)) for index in range(3))
    assert compaction.compaction_slice(messages, keep_last=6) == 0


def test_head_share_measures_what_folding_would_save() -> None:
    messages = (user("x" * 900), user("y"))
    assert compaction.head_share(messages, 1) > 0.9


def test_the_transcript_includes_tool_calls() -> None:
    """Tool calls are most of what an agent did; prose alone reads as idleness."""
    message = assistant("", tool_calls=(ToolCall(id=CallId("c1"), name="echo", arguments={}),))
    assert "echo" in compaction.render(message)


def test_the_transcript_labels_tool_results_by_tool() -> None:
    assert "tool:echo" in compaction.render(tool_result(CallId("c1"), "echo", "hi"))


def test_the_request_carries_no_tools_and_asks_for_prose() -> None:
    messages = (user("do it"), assistant("done"))
    request = compaction.compaction_request(messages, through_index=2, params=make_spec().model)
    assert request.tools == ()
    assert request.messages[0].content == compaction.INSTRUCTION


# -- the policy stage ---------------------------------------------------------


def policy(*tools) -> AgentPolicy:
    return AgentPolicy(InMemoryToolRegistry(tools), DefaultContextBuilder())


def bloated(limit: int = 2_000, *, turns: int = 20, size: int = 400, **run_fields):
    """A run mid-conversation, oversized, and not finished: every turn asks for
    a tool, so the completion check cannot short-circuit the stage under test."""
    spec = make_spec(limits=ResourceLimits(max_context_chars=limit))
    messages = tuple(
        assistant("x" * size, tool_calls=(call("echo", f"c{index}"),)) for index in range(turns)
    )
    return make_run(spec=spec).model_copy(
        update={"prompt": PROMPT, "messages": messages, **run_fields}
    )


def test_a_conversation_within_budget_is_not_compacted() -> None:
    assert isinstance(policy().decide(bloated(limit=100_000)), CallModel)


def test_an_oversized_conversation_is_compacted() -> None:
    decision = policy().decide(bloated())
    assert isinstance(decision, CompactHistory)
    assert decision.through_index >= 1


def test_compaction_fires_before_the_trimmer_would_truncate() -> None:
    """At 0.75 of budget, so summarisation always gets its chance first."""
    run = bloated(limit=10_000, turns=20, size=400)
    assert compaction.estimated_chars(run.messages) < run.spec.limits.max_context_chars
    assert isinstance(policy().decide(run), CompactHistory)


def test_the_count_budget_stops_compaction() -> None:
    exhausted = bloated(compactions=ResourceLimits().max_compactions)
    assert isinstance(policy().decide(exhausted), CallModel)


def test_a_fold_that_would_bring_the_request_under_budget_is_taken() -> None:
    """The direct test of "will this help?", which a share threshold only
    approximates -- and approximates conservatively enough to refuse folds that
    would have worked."""
    run = bloated(limit=6_000, turns=10, size=500)
    through = compaction.compaction_slice(run.messages)
    assert compaction.head_share(run.messages, through) < compaction.MIN_HEAD_SHARE
    assert isinstance(policy().decide(run), CompactHistory)


def test_folding_stops_once_it_can_no_longer_help() -> None:
    """Termination for the branch above: after a fold that fits, the trigger
    cannot fire again, so there is no second wasted call."""
    run = bloated(limit=6_000, turns=10, size=500)
    decision = policy().decide(run)
    assert isinstance(decision, CompactHistory)
    folded = apply_compaction(run, decision.through_index, "a short summary")
    assert not isinstance(policy().decide(folded), CompactHistory)


def apply_compaction(run, through_index: int, summary: str):
    from skynet.domain.events import HistoryCompacted
    from skynet.domain.run import AgentStatus, apply

    started = run.model_copy(update={"status": AgentStatus.RUNNING})
    return apply(
        started,
        HistoryCompacted(
            run_id=started.run_id, seq=99, through_index=through_index, summary=summary
        ),
    )


def test_a_conversation_that_is_all_tail_is_left_to_the_trimmer() -> None:
    """Without this guard a run whose recent turns alone blow the budget would
    summarise forever without ever shrinking."""
    spec = make_spec(limits=ResourceLimits(max_context_chars=2_000))
    tail = tuple(
        assistant("x" * 2_000, tool_calls=(call("echo", f"c{index}"),))
        for index in range(compaction.KEEP_LAST)
    )
    run = make_run(spec=spec).model_copy(
        update={"prompt": PROMPT, "messages": (user("tiny"), *tail)}
    )
    assert compaction.estimated_chars(run.messages) > spec.limits.max_context_chars
    assert isinstance(policy().decide(run), CallModel)


def test_a_finished_run_is_not_compacted() -> None:
    """Shrinking a context that will never be sent is a wasted model call."""
    run = bloated()
    finished = run.model_copy(update={"messages": (*run.messages, assistant("done"))})
    assert not isinstance(policy().decide(finished), CompactHistory)


# -- end to end ---------------------------------------------------------------


class Rambler:
    """A model that pads every turn, so the context outgrows the budget."""

    def __init__(self, turns: int, padding: int = 800) -> None:
        self._turns = turns
        self._padding = padding
        self.summaries = 0
        self.spoken = 0

    def __call__(self, request):
        if request.messages and request.messages[0].content == compaction.INSTRUCTION:
            self.summaries += 1
            return says("the agent echoed a lot of padding")
        self.spoken += 1
        if self.spoken >= self._turns:
            return says("done")
        return calls(call("echo", f"c{self.spoken}", text="x" * self._padding))


async def run_rambling(turns: int, budget: int):
    model = Rambler(turns)
    services = build_services(
        model=ScriptedModelProvider(rule=model),
        tools=[EchoTool()],
    )
    spec = make_spec(
        tools=("echo",),
        limits=ResourceLimits(max_context_chars=budget, max_iterations=20, max_tool_calls=40),
    )
    run = await InProcessAgentRunner(services).run(
        RunRequest(spec=spec, task=make_task(), context=make_context())
    )
    compactions = [e for e in services.sink.events if isinstance(e, HistoryCompacted)]
    return run, model, compactions, services


async def test_a_long_run_compacts_and_still_completes() -> None:
    run, model, compactions, _ = await run_rambling(turns=8, budget=3_000)
    assert run.status is AgentStatus.COMPLETED
    assert compactions, "the run never compacted despite exceeding its budget"
    assert model.summaries == len(compactions)


async def test_compaction_leaves_exactly_one_note() -> None:
    run, _, compactions, _ = await run_rambling(turns=8, budget=3_000)
    assert len(compactions) >= 1
    notes = [m for m in run.messages if m.content.startswith(COMPACTION_NOTE_PREFIX)]
    assert len(notes) <= 1


async def test_compaction_does_not_loop() -> None:
    """Three guards, and the budget is the last of them."""
    run, _, compactions, _ = await run_rambling(turns=8, budget=3_000)
    assert len(compactions) <= run.spec.limits.max_compactions


async def test_the_run_survives_a_summariser_that_fails() -> None:
    """Memory work must never be able to fail a run."""

    def rule(request):
        if request.messages and request.messages[0].content == compaction.INSTRUCTION:
            raise RuntimeError("summariser exploded")
        rule.spoken = getattr(rule, "spoken", 0) + 1
        if rule.spoken >= 8:
            return says("done")
        return calls(call("echo", f"c{rule.spoken}", text="x" * 800))

    services = build_services(model=ScriptedModelProvider(rule=rule), tools=[EchoTool()])
    spec = make_spec(
        tools=("echo",),
        limits=ResourceLimits(max_context_chars=3_000, max_iterations=20, max_tool_calls=40),
    )
    run = await InProcessAgentRunner(services).run(
        RunRequest(spec=spec, task=make_task(), context=make_context())
    )
    failures = [
        e for e in services.sink.events if isinstance(e, HistoryCompacted) and e.error is not None
    ]
    assert run.status is AgentStatus.COMPLETED
    assert failures, "the failing summariser should have been recorded"
    assert run.model_failures == 0, "housekeeping must not spend the model-failure budget"
