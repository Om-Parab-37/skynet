"""The reducer is the foundation of durability -- test it exhaustively."""

from __future__ import annotations

import pytest

from skynet.domain.errors import ErrorInfo, InvalidTransition
from skynet.domain.events import (
    ApprovalRequested,
    ApprovalResolved,
    ChildRunStarted,
    HistoryCompacted,
    MemoryConsolidated,
    MemoryRecalled,
    ModelFailed,
    ModelResponded,
    PromptResolved,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
    ToolCompleted,
)
from skynet.domain.ids import CallId, MemoryId, RunId
from skynet.domain.memory import (
    MAX_WORKING_MEMORIES,
    ConsolidationMode,
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
)
from skynet.domain.messages import Role
from skynet.domain.models import FinishReason, ModelResponse
from skynet.domain.prompts import PromptRef, RenderedPrompt
from skynet.domain.run import COMPACTION_NOTE_PREFIX, AgentStatus, apply, replay
from skynet.domain.task import AgentResult
from skynet.domain.tools import ToolResult, ToolStatus
from skynet.testing.builders import call, calls, make_run, says


def started(run):
    return RunStarted(run_id=run.run_id, seq=1, spec=run.spec, task=run.task, context=run.context)


def test_run_starts_in_created_and_transitions_to_running():
    run = make_run()
    assert run.status is AgentStatus.CREATED
    run = apply(run, started(run))
    assert run.status is AgentStatus.RUNNING
    assert run.seq == 1


def test_prompt_resolved_sets_prompt():
    run = apply(make_run(), started(make_run()))
    prompt = RenderedPrompt(ref=PromptRef(name="p"), text="hello")
    run = apply(run, PromptResolved(run_id=run.run_id, seq=2, prompt=prompt))
    assert run.prompt is not None and run.prompt.text == "hello"


# -- memory -------------------------------------------------------------------

SCOPE = MemoryScope.agent("demo")


def hit(memory_id: str, score: float = 1.0) -> MemoryHit:
    record = MemoryRecord(id=MemoryId(memory_id), scope=SCOPE, content=f"fact {memory_id}")
    return MemoryHit(record=record, relevance=score, score=score)


def query(text: str = "q") -> MemoryQuery:
    return MemoryQuery(scopes=(SCOPE,), text=text)


def running():
    return apply(make_run(), started(make_run()))


def test_memory_recalled_records_the_query_and_the_hits():
    run = apply(running(), MemoryRecalled(run_id="run-1", seq=2, query=query(), hits=(hit("m1"),)))
    assert [h.record.id for h in run.memories] == [MemoryId("m1")]
    assert run.recalled_queries == (query().signature(),)


def test_a_second_recall_accumulates_rather_than_clobbering():
    """Recall fires more than once now; replacing would silently lose the first."""
    run = running()
    run = apply(run, MemoryRecalled(run_id="run-1", seq=2, query=query("a"), hits=(hit("m1"),)))
    run = apply(run, MemoryRecalled(run_id="run-1", seq=3, query=query("b"), hits=(hit("m2"),)))
    assert {h.record.id for h in run.memories} == {MemoryId("m1"), MemoryId("m2")}
    assert len(run.recalled_queries) == 2


def test_recalling_the_same_query_twice_records_one_signature():
    run = running()
    for seq in (2, 3):
        run = apply(run, MemoryRecalled(run_id="run-1", seq=seq, query=query(), hits=(hit("m1"),)))
    assert run.recalled_queries == (query().signature(),)
    assert len(run.memories) == 1


def test_the_working_set_is_capped():
    hits = tuple(hit(f"m{index:03d}", index / 100) for index in range(40))
    run = apply(running(), MemoryRecalled(run_id="run-1", seq=2, query=query(), hits=hits))
    assert len(run.memories) == MAX_WORKING_MEMORIES


def consolidated(seq: int, through: int, **fields):
    fields.setdefault("mode", ConsolidationMode.EXTRACT)
    return MemoryConsolidated(run_id="run-1", seq=seq, scope=SCOPE, through_index=through, **fields)


def test_consolidation_advances_the_watermark():
    run = apply(running(), consolidated(2, 5))
    assert run.consolidated_through == 5
    assert run.consolidation_count == 1


def test_consolidation_advances_even_when_it_achieved_nothing():
    """The whole termination argument for "consolidate before completing"."""
    empty = apply(running(), consolidated(2, 5, operations=()))
    failed = apply(running(), consolidated(2, 5, error=ErrorInfo.of("memory", "store down")))
    assert empty.consolidated_through == 5
    assert failed.consolidated_through == 5


def test_the_watermark_never_moves_backwards():
    run = apply(apply(running(), consolidated(2, 9)), consolidated(3, 4))
    assert run.consolidated_through == 9


def test_consolidation_never_touches_the_conversation():
    """`_completion` reads `messages`; mutating them here would oscillate the run."""
    run = apply(running(), ModelResponded(run_id="run-1", seq=2, response=says("hello")))
    before = run.messages
    assert apply(run, consolidated(3, 1)).messages == before


def test_reflection_watermarks_on_the_iteration():
    """Reflection does not advance `iteration`, so it needs its own watermark."""
    run = apply(running(), ModelResponded(run_id="run-1", seq=2, response=says("hello")))
    reflected = apply(run, consolidated(3, 0, mode=ConsolidationMode.REFLECT))
    assert reflected.reflected_at_iteration == run.iteration
    assert reflected.consolidated_through == 0


# -- compaction ---------------------------------------------------------------


def talking(*texts: str):
    run = running()
    for index, text in enumerate(texts):
        run = apply(run, ModelResponded(run_id="run-1", seq=index + 2, response=says(text)))
    return run


def test_compaction_folds_the_head_into_one_note():
    run = talking("one", "two", "three")
    compacted = apply(
        run, HistoryCompacted(run_id="run-1", seq=9, through_index=2, summary="they talked")
    )
    assert len(compacted.messages) == 2
    assert compacted.messages[0].role is Role.SYSTEM
    assert "they talked" in compacted.messages[0].content
    assert compacted.messages[1].content == "three"
    assert compacted.compactions == 1
    assert compacted.compacted_messages == 2


def test_compaction_never_stacks_notes():
    """The previous note sits at index 0 and so is inside the next fold."""
    run = talking("one", "two", "three", "four")
    first = apply(run, HistoryCompacted(run_id="run-1", seq=9, through_index=2, summary="A"))
    second = apply(first, HistoryCompacted(run_id="run-1", seq=10, through_index=2, summary="B"))
    notes = [m for m in second.messages if m.content.startswith(COMPACTION_NOTE_PREFIX)]
    assert len(notes) == 1
    assert "B" in notes[0].content


def test_compaction_drops_a_tool_result_orphaned_by_the_fold():
    """Strict providers reject a tool result with no matching assistant turn."""
    run = talking("one")
    run = apply(
        run,
        ModelResponded(run_id="run-1", seq=3, response=calls(call("echo", text="hi"))),
    )
    run = apply(
        run,
        ToolCompleted(
            run_id="run-1",
            seq=4,
            result=ToolResult(
                call_id=CallId("c1"), name="echo", status=ToolStatus.OK, content="hi"
            ),
        ),
    )
    compacted = apply(run, HistoryCompacted(run_id="run-1", seq=5, through_index=2, summary="s"))
    assert not any(m.is_tool_result for m in compacted.messages)


def test_compaction_leaves_the_step_log_intact():
    """Compaction is lossy for the model's view only; the log stays complete."""
    run = talking("one", "two", "three")
    compacted = apply(run, HistoryCompacted(run_id="run-1", seq=9, through_index=2, summary="s"))
    assert len(compacted.steps) == len(run.steps)


def test_a_failed_compaction_counts_but_folds_nothing():
    """Destroying real history to save tokens the trimmer can reclaim anyway
    would make a transient model failure permanently lossy."""
    run = talking("one", "two", "three")
    failed = apply(
        run,
        HistoryCompacted(
            run_id="run-1",
            seq=9,
            through_index=2,
            error=ErrorInfo.of("compaction", "model unavailable"),
        ),
    )
    assert failed.messages == run.messages
    assert failed.compactions == 1, "the budget must still advance, or this loops"


def test_replay_reproduces_a_compacted_conversation_exactly():
    """`summary` travels in the event, so no model is needed to rebuild state."""
    events = [
        started(make_run()),
        ModelResponded(run_id="run-1", seq=2, response=says("one")),
        ModelResponded(run_id="run-1", seq=3, response=says("two")),
        ModelResponded(run_id="run-1", seq=4, response=says("three")),
        HistoryCompacted(run_id="run-1", seq=5, through_index=2, summary="they talked"),
    ]
    live = replay(make_run(), events)
    again = replay(make_run(), events)
    assert live.messages == again.messages
    assert live.model_dump_json() == again.model_dump_json()


def test_model_responded_accumulates_history_usage_and_pending_calls():
    run = apply(make_run(), started(make_run()))
    response = calls(call("echo", text="hi"))
    run = apply(run, ModelResponded(run_id=run.run_id, seq=2, response=response))

    assert run.iteration == 1
    assert len(run.messages) == 1
    assert len(run.steps) == 1
    assert run.pending_calls == response.message.tool_calls
    assert run.usage.total == 15


def test_model_responded_resets_failure_counter():
    run = apply(make_run(), started(make_run()))
    run = apply(run, ModelFailed(run_id=run.run_id, seq=2, error=ErrorInfo.of("x", "y")))
    assert run.model_failures == 1
    run = apply(run, ModelResponded(run_id=run.run_id, seq=3, response=says("done")))
    assert run.model_failures == 0


def test_parse_error_increments_parse_retries():
    run = apply(make_run(), started(make_run()))
    bad = ModelResponse(message=says("junk").message, finish_reason=FinishReason.PARSE_ERROR)
    run = apply(run, ModelResponded(run_id=run.run_id, seq=2, response=bad))
    assert run.parse_retries == 1
    run = apply(run, ModelResponded(run_id=run.run_id, seq=3, response=bad))
    assert run.parse_retries == 2


def test_tool_completed_appends_message_and_clears_that_pending_call():
    run = apply(make_run(), started(make_run()))
    run = apply(
        run,
        ModelResponded(
            run_id=run.run_id,
            seq=2,
            response=calls(call("echo", "c1", text="a"), call("echo", "c2", text="b")),
        ),
    )
    assert len(run.pending_calls) == 2

    result = ToolResult(call_id=CallId("c1"), name="echo", status=ToolStatus.OK, content="a")
    run = apply(run, ToolCompleted(run_id=run.run_id, seq=3, result=result))

    assert [c.id for c in run.pending_calls] == ["c2"]
    assert run.tool_call_count == 1
    assert run.messages[-1].is_tool_result
    assert run.steps[-1].tool_results[0].content == "a"


def test_approval_request_suspends_and_resolution_resumes():
    run = apply(make_run(), started(make_run()))
    run = apply(run, ApprovalRequested(run_id=run.run_id, seq=2, call_ids=(CallId("c1"),)))
    assert run.status is AgentStatus.WAITING_APPROVAL
    assert run.status.is_suspended

    run = apply(run, ApprovalResolved(run_id=run.run_id, seq=3, call_id=CallId("c1"), granted=True))
    assert run.status is AgentStatus.RUNNING
    assert run.approvals[CallId("c1")] is True


def test_pre_approval_while_running_is_idempotent():
    """Resolving an approval that was never requested must not break the run."""
    run = apply(make_run(), started(make_run()))
    run = apply(run, ApprovalResolved(run_id=run.run_id, seq=2, call_id=CallId("c1"), granted=True))
    assert run.status is AgentStatus.RUNNING


def test_pause_and_resume():
    run = apply(make_run(), started(make_run()))
    run = apply(run, RunPaused(run_id=run.run_id, seq=2, reason="operator"))
    assert run.status is AgentStatus.PAUSED and run.status.is_suspended
    run = apply(run, RunResumed(run_id=run.run_id, seq=3))
    assert run.status is AgentStatus.RUNNING


def test_child_run_recorded():
    run = apply(make_run(), started(make_run()))
    run = apply(
        run,
        ChildRunStarted(
            run_id=run.run_id,
            seq=2,
            call_id=CallId("c1"),
            child_run_id=RunId("child-1"),
            agent="researcher",
        ),
    )
    assert run.child_runs == ((CallId("c1"), RunId("child-1")),)


@pytest.mark.parametrize(
    ("event_factory", "expected"),
    [
        (
            lambda r: RunCompleted(run_id=r.run_id, seq=9, result=AgentResult(output="x")),
            AgentStatus.COMPLETED,
        ),
        (
            lambda r: RunFailed(run_id=r.run_id, seq=9, error=ErrorInfo.of("e", "m")),
            AgentStatus.FAILED,
        ),
        (lambda r: RunCancelled(run_id=r.run_id, seq=9, reason="stop"), AgentStatus.CANCELLED),
    ],
)
def test_terminal_transitions(event_factory, expected):
    run = apply(make_run(), started(make_run()))
    run = apply(run, event_factory(run))
    assert run.status is expected
    assert run.status.is_terminal


def test_applying_to_a_terminal_run_raises():
    run = apply(make_run(), started(make_run()))
    run = apply(run, RunCompleted(run_id=run.run_id, seq=2, result=AgentResult(output="x")))
    with pytest.raises(InvalidTransition):
        apply(run, ModelResponded(run_id=run.run_id, seq=3, response=says("more")))


def test_completed_cannot_be_cancelled():
    run = apply(make_run(), started(make_run()))
    run = apply(run, RunCompleted(run_id=run.run_id, seq=2, result=AgentResult(output="x")))
    with pytest.raises(InvalidTransition):
        apply(run, RunCancelled(run_id=run.run_id, seq=3))


def test_replay_is_deterministic_and_prefix_associative():
    """The property Temporal's replay depends on."""
    run = make_run()
    events = [
        started(run),
        PromptResolved(
            run_id=run.run_id, seq=2, prompt=RenderedPrompt(ref=PromptRef(name="p"), text="s")
        ),
        ModelResponded(run_id=run.run_id, seq=3, response=calls(call("echo", text="x"))),
        ToolCompleted(
            run_id=run.run_id,
            seq=4,
            result=ToolResult(
                call_id=CallId("call-1"), name="echo", status=ToolStatus.OK, content="x"
            ),
        ),
        ModelResponded(run_id=run.run_id, seq=5, response=says("all done")),
        RunCompleted(run_id=run.run_id, seq=6, result=AgentResult(output="all done")),
    ]

    once = replay(run, events)
    twice = replay(run, events)
    assert once == twice

    for split in range(len(events) + 1):
        partial = replay(run, events[:split])
        assert replay(partial, events[split:]) == once
