"""Context construction -- turning run state into a model request.

Pure and section-based. Adding a new kind of context (a plan, a scratchpad,
few-shot examples, retrieved documents) means adding a ``ContextSection`` and
passing it in, never editing the agent loop.

The LLM-backed summariser is *not* here, and must not be: it is not pure, and
hiding it inside a trimmer would put an ``await`` inside ``decide``. It is a
``CompactHistory`` decision, so it stays visible in the event log and replays
under Temporal. This module holds the hard cap that runs when summarisation did
not happen; both measure size with the same ``estimated_chars``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Protocol, runtime_checkable

from skynet.application.history_compaction import estimated_chars
from skynet.domain.agent import FinishMode
from skynet.domain.memory import MemoryKind
from skynet.domain.messages import Message, Role, system, user
from skynet.domain.models import ModelRequest, ToolChoice
from skynet.domain.run import AgentRun
from skynet.domain.tools import ToolSpec


@runtime_checkable
class ContextSection(Protocol):
    def render(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> tuple[Message, ...]: ...


@runtime_checkable
class HistoryTrimmer(Protocol):
    def trim(self, messages: tuple[Message, ...], *, budget: int) -> tuple[Message, ...]:
        """Cut ``messages`` down to ``budget`` characters.

        The budget is a parameter rather than trimmer state so that it and the
        compaction trigger read the same number off ``ResourceLimits``. Two
        numbers would let this truncate history the summariser was about to
        fold, wasting the model call that produced the summary.
        """
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    def build(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> ModelRequest: ...


class SystemSection:
    """The resolved prompt plus any static per-agent instructions."""

    def render(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> tuple[Message, ...]:
        parts: list[str] = []
        if run.prompt is not None and run.prompt.text.strip():
            parts.append(run.prompt.text.strip())
        if run.spec.instructions.strip():
            parts.append(run.spec.instructions.strip())
        if not parts:
            return ()
        return (system("\n\n".join(parts)),)


MEMORY_HEADINGS = {
    MemoryKind.SEMANTIC: "What I know",
    MemoryKind.PROCEDURAL: "How things are done here",
    MemoryKind.EPISODIC: "What happened before",
}


class MemorySection:
    """Recalled memories, grouped by kind under fixed headings.

    Grouping is not decoration. A model treats "this is true", "this is how the
    job is done here" and "this happened once" very differently, and a flat
    bulleted list invites it to treat a one-off observation as a standing fact.

    The cap is not optional either: this renders as a *system* message, and the
    trimmer will not drop those. Without a cap, enough recalled memories could
    crowd the actual conversation out of a request entirely.
    """

    def __init__(self, max_chars: int = 4_000) -> None:
        self._max_chars = max_chars

    def render(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> tuple[Message, ...]:
        if not run.memories:
            return ()

        remaining = self._max_chars
        blocks: list[str] = []
        # Iterate headings, not hits, so ordering is fixed rather than dependent
        # on what happened to be recalled -- a prompt that reshuffles between
        # runs is a prompt whose behaviour cannot be compared.
        for kind, heading in MEMORY_HEADINGS.items():
            lines: list[str] = []
            for hit in run.memories:
                if hit.record.kind is not kind:
                    continue
                line = f"- {hit.record.content.strip()}"
                if len(line) + 1 > remaining:
                    break
                remaining -= len(line) + 1
                lines.append(line)
            if lines:
                blocks.append(f"{heading}:\n" + "\n".join(lines))

        if not blocks:
            return ()
        return (system("\n\n".join(blocks)),)


class TaskSection:
    def render(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> tuple[Message, ...]:
        text = run.task.objective
        if run.task.input:
            text = f"{text}\n\nInput:\n{run.task.input}"
        if run.task.attachments:
            refs = "\n".join(f"- {a.uri}" for a in run.task.attachments)
            text = f"{text}\n\nAttachments:\n{refs}"
        return (user(text),)


class HistorySection:
    def render(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> tuple[Message, ...]:
        return run.messages


class RepairSection:
    """Injected only after the model produced output we could not parse."""

    def __init__(self, reminder: str = "") -> None:
        self._reminder = reminder

    def render(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> tuple[Message, ...]:
        if not run.parse_retries or not self._reminder:
            return ()
        return (system(self._reminder),)


class TailWindowTrimmer:
    """Keep the prompt and the most recent turns, and drop the middle.

    The hard cap, not the intelligent path -- compaction is that, and it runs
    first. This is what happens when compaction had no budget left, no working
    model, or nothing worth folding.

    Never orphans a tool result from the assistant turn that requested it:
    strict providers reject that outright and every model finds it confusing.
    """

    def __init__(self, max_chars: int = 24_000, *, max_system_share: float = 0.5) -> None:
        self._max_chars = max_chars
        self._max_system_share = max_system_share

    def trim(
        self, messages: tuple[Message, ...], *, budget: int | None = None
    ) -> tuple[Message, ...]:
        limit = budget if budget is not None else self._max_chars
        if estimated_chars(messages) <= limit:
            return messages

        leading = self._system(messages, limit)
        rest = [m for m in messages if m.role is not Role.SYSTEM]

        remaining = limit - estimated_chars(leading)
        kept: list[Message] = []
        for message in reversed(rest):
            size = estimated_chars((message,))
            if remaining - size < 0 and kept:
                break
            remaining -= size
            kept.append(message)
        kept.reverse()

        # Drop leading tool results whose requesting assistant turn was trimmed.
        while kept and kept[0].is_tool_result:
            kept.pop(0)
        return (*leading, *kept)

    def _system(self, messages: Sequence[Message], limit: int) -> tuple[Message, ...]:
        """System messages, bounded so they cannot crowd out the conversation.

        This became load-bearing the moment memories and a compaction note
        joined the resolved prompt as system content: without a share cap, a
        long run can reach a request that is all system text and no
        conversation, and the model is then asked to continue a dialogue it
        cannot see. The first system message is the resolved prompt and is never
        dropped -- an agent without its instructions is a different agent.
        """
        system_messages = [m for m in messages if m.role is Role.SYSTEM]
        if not system_messages:
            return ()

        allowance = int(limit * self._max_system_share)
        kept = [system_messages[0]]
        used = estimated_chars(kept)
        for message in system_messages[1:]:
            size = estimated_chars((message,))
            if used + size > allowance:
                continue
            used += size
            kept.append(message)
        return tuple(kept)


class DefaultContextBuilder:
    def __init__(
        self,
        sections: Sequence[ContextSection] | None = None,
        trimmer: HistoryTrimmer | None = None,
    ) -> None:
        self._sections: tuple[ContextSection, ...] = tuple(
            sections
            if sections is not None
            else (
                SystemSection(),
                MemorySection(),
                TaskSection(),
                HistorySection(),
                RepairSection(),
            )
        )
        self._trimmer: HistoryTrimmer = trimmer or TailWindowTrimmer()

    def build(self, run: AgentRun, tools: tuple[ToolSpec, ...]) -> ModelRequest:
        messages = tuple(chain.from_iterable(s.render(run, tools) for s in self._sections))
        return ModelRequest(
            messages=self._trimmer.trim(messages, budget=run.spec.limits.max_context_chars),
            params=run.spec.model,
            tools=tools,
            tool_choice=ToolChoice.AUTO,
            response_schema=run.spec.output_schema,
            max_parallel_tool_calls=(
                1 if run.finish_mode is FinishMode.FINAL_ANSWER_TOOL else None
            ),
        )


__all__ = [
    "MEMORY_HEADINGS",
    "ContextBuilder",
    "ContextSection",
    "DefaultContextBuilder",
    "HistorySection",
    "HistoryTrimmer",
    "MemorySection",
    "RepairSection",
    "SystemSection",
    "TailWindowTrimmer",
    "TaskSection",
]
