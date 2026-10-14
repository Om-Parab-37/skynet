"""Context assembly: what the model is actually shown, and what gets cut."""

from __future__ import annotations

from skynet.application.context import (
    MEMORY_HEADINGS,
    DefaultContextBuilder,
    MemorySection,
    TailWindowTrimmer,
)
from skynet.application.history_compaction import estimated_chars
from skynet.domain.agent import ResourceLimits
from skynet.domain.ids import CallId, MemoryId
from skynet.domain.memory import MemoryHit, MemoryKind, MemoryRecord, MemoryScope
from skynet.domain.messages import Role, assistant, system, tool_result, user
from skynet.domain.prompts import PromptRef, RenderedPrompt
from skynet.testing.builders import make_run, make_spec

PROMPT = RenderedPrompt(ref=PromptRef(name="p"), text="you are demo")
SCOPE = MemoryScope.agent("demo")


def hit(content: str, kind: MemoryKind = MemoryKind.SEMANTIC, memory_id: str = "m") -> MemoryHit:
    record = MemoryRecord(id=MemoryId(memory_id), scope=SCOPE, content=content, kind=kind)
    return MemoryHit(record=record, relevance=1.0, score=1.0)


def with_memories(*hits: MemoryHit):
    return make_run().model_copy(update={"prompt": PROMPT, "memories": hits})


# -- memories -----------------------------------------------------------------


def test_no_memories_renders_nothing() -> None:
    assert MemorySection().render(make_run(), ()) == ()


def test_memories_are_grouped_under_their_kind() -> None:
    """A model treats "this is true" and "this happened once" very differently;
    a flat list invites it to read a one-off observation as a standing fact."""
    run = with_memories(
        hit("the port is 5433", MemoryKind.SEMANTIC, "m1"),
        hit("deploys go through the vault", MemoryKind.PROCEDURAL, "m2"),
        hit("the last deploy failed on a timeout", MemoryKind.EPISODIC, "m3"),
    )
    text = MemorySection().render(run, ())[0].content
    for heading in MEMORY_HEADINGS.values():
        assert heading in text
    assert text.index(MEMORY_HEADINGS[MemoryKind.SEMANTIC]) < text.index(
        MEMORY_HEADINGS[MemoryKind.EPISODIC]
    )


def test_only_the_headings_in_use_appear() -> None:
    text = MemorySection().render(with_memories(hit("a fact")), ())[0].content
    assert MEMORY_HEADINGS[MemoryKind.EPISODIC] not in text


def test_heading_order_does_not_depend_on_what_was_recalled() -> None:
    """A prompt that reshuffles between runs cannot be compared between runs."""
    forwards = with_memories(
        hit("a", MemoryKind.SEMANTIC, "m1"), hit("b", MemoryKind.EPISODIC, "m2")
    )
    backwards = with_memories(
        hit("b", MemoryKind.EPISODIC, "m2"), hit("a", MemoryKind.SEMANTIC, "m1")
    )
    assert (
        MemorySection().render(forwards, ())[0].content
        == MemorySection().render(backwards, ())[0].content
    )


def test_memories_are_capped() -> None:
    """They render as system content, which the trimmer will not drop."""
    hits = tuple(hit("x" * 200, memory_id=f"m{index}") for index in range(50))
    rendered = MemorySection(max_chars=1_000).render(with_memories(*hits), ())
    assert len(rendered[0].content) <= 1_100


# -- the trimmer --------------------------------------------------------------


def test_a_conversation_within_budget_is_untouched() -> None:
    messages = (system("prompt"), user("hi"), assistant("hello"))
    assert TailWindowTrimmer().trim(messages, budget=10_000) == messages


def test_the_budget_is_a_parameter_not_trimmer_state() -> None:
    """So this and the compaction trigger cannot disagree about the limit."""
    messages = (system("prompt"), *(user("x" * 500) for _ in range(20)))
    tight = TailWindowTrimmer().trim(messages, budget=2_000)
    loose = TailWindowTrimmer().trim(messages, budget=40_000)
    assert len(tight) < len(loose) == len(messages)


def test_trimming_keeps_the_most_recent_turns() -> None:
    messages = (system("prompt"), *(user(f"turn {index}") for index in range(40)))
    kept = TailWindowTrimmer().trim(messages, budget=600)
    assert kept[-1].content == "turn 39"


def test_trimming_never_orphans_a_tool_result() -> None:
    messages = (
        system("prompt"),
        *(user("x" * 400) for _ in range(10)),
        assistant("calling"),
        tool_result(CallId("c1"), "echo", "y" * 400),
        user("and now?"),
    )
    kept = TailWindowTrimmer().trim(messages, budget=1_200)
    assert not kept[1].is_tool_result if len(kept) > 1 else True
    assert all(
        not message.is_tool_result or index > 0
        for index, message in enumerate(m for m in kept if m.role is not Role.SYSTEM)
    )


def test_system_content_cannot_crowd_out_the_conversation() -> None:
    """Memories and a compaction note are both system messages. Without a share
    cap a long run reaches a request that is all system text and no dialogue --
    the model is then asked to continue something it cannot see."""
    messages = (
        system("the resolved prompt"),
        system("m" * 4_000),
        system("n" * 4_000),
        *(user(f"turn {index}") for index in range(6)),
    )
    kept = TailWindowTrimmer().trim(messages, budget=4_000)
    conversation = [m for m in kept if m.role is not Role.SYSTEM]
    assert conversation, "the whole budget went to system messages"
    assert estimated_chars([m for m in kept if m.role is Role.SYSTEM]) <= 2_000


def test_the_resolved_prompt_is_never_dropped() -> None:
    """An agent without its instructions is a different agent."""
    messages = (system("the resolved prompt"), *(user("x" * 800) for _ in range(20)))
    kept = TailWindowTrimmer().trim(messages, budget=1_000)
    assert kept[0].content == "the resolved prompt"


# -- the builder --------------------------------------------------------------


def test_the_builder_passes_the_agents_budget_to_the_trimmer() -> None:
    spec = make_spec(limits=ResourceLimits(max_context_chars=1_500))
    run = make_run(spec=spec).model_copy(
        update={"prompt": PROMPT, "messages": tuple(user("x" * 400) for _ in range(20))}
    )
    request = DefaultContextBuilder().build(run, ())
    assert estimated_chars(request.messages) <= 1_500
