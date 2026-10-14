"""Deciding *what* to compact, and building the request that summarises it.

Pure. The model call itself is a ``CompactHistory`` decision executed at the
seam, which is the whole reason this is a separate module rather than something
the trimmer does: summarising inside ``ContextBuilder.build`` would put an
``await`` in the middle of ``decide`` and make the policy impure.

Compaction and the trimmer are not competitors. Compaction is the intelligent
path -- fold the head into a note and keep its content. The trimmer is the hard
cap that runs when there was no time, no budget or no working model. They must
read one number, or the trimmer will truncate history the summariser was about
to fold.
"""

from __future__ import annotations

from collections.abc import Sequence

from skynet.domain.messages import Message, system, user
from skynet.domain.models import ModelParams, ModelRequest, ToolChoice

TRIGGER_RATIO = 0.75
"""Fraction of the context budget at which compaction fires.

Below 1.0 on purpose: summarisation must get its chance *before* the trimmer
starts dropping turns, or the intelligent path never runs at all.
"""

MIN_HEAD_SHARE = 0.5
"""How much of the context the folded head must represent to be worth folding
on its own -- see ``worth_folding`` for the other half of the rule."""

KEEP_LAST = 6
"""Recent messages never folded, so the model keeps its immediate footing."""

PER_MESSAGE_OVERHEAD = 64
"""Rough allowance for role markers and framing, counted in characters."""

INSTRUCTION = (
    "You are compacting an AI agent's conversation so it can keep working with "
    "a smaller context.\n\n"
    "Write a summary of the transcript below that preserves, in this order of "
    "priority:\n"
    "1. the task, and any constraints or requirements stated for it;\n"
    "2. facts, figures, file paths and identifiers that were established;\n"
    "3. what has already been tried, and what the outcome was;\n"
    "4. decisions taken and anything still open.\n\n"
    "Maximise recall first and precision second: a detail you drop is gone for "
    "good, and the agent will act as though it never happened. Write plain "
    "prose or bullets, no preamble, and never invent anything that is not in "
    "the transcript."
)


def estimated_chars(messages: Sequence[Message]) -> int:
    """The single size measure. Characters, not tokens, and deliberately so:
    tokenising is model-specific and would make a pure policy depend on which
    provider is configured."""
    return sum(len(message.content) + PER_MESSAGE_OVERHEAD for message in messages)


def compaction_slice(messages: Sequence[Message], keep_last: int = KEEP_LAST) -> int:
    """How many leading messages to fold. Zero means "nothing worth folding"."""
    return max(0, len(messages) - max(1, keep_last))


def head_share(messages: Sequence[Message], through_index: int) -> float:
    total = estimated_chars(messages)
    if total <= 0:
        return 0.0
    return estimated_chars(messages[:through_index]) / total


def worth_folding(
    messages: Sequence[Message],
    through_index: int,
    *,
    budget: int,
    min_head_share: float = MIN_HEAD_SHARE,
) -> bool:
    """Whether folding this head is worth a model call.

    Two ways to qualify, and both are needed:

    * the fold would bring the request back under budget -- the direct test of
      "will this help?", which a share threshold can only approximate; or
    * the head is a large share of the context, so folding makes real progress
      even if one pass is not enough on its own.

    Failing both means the *tail alone* exceeds the budget, and no amount of
    summarising the head will fix that. Without this the policy would summarise
    forever without ever shrinking; with it, that case falls to the trimmer,
    whose job it is.

    Both branches terminate. If the fold brings the total under budget the
    trigger cannot fire again; if it merely makes progress, the head becomes a
    single note whose share is tiny, so the next pass must qualify on the first
    branch or not at all.
    """
    tail = estimated_chars(messages[through_index:])
    fits_after = tail + _NOTE_ALLOWANCE < budget
    return fits_after or head_share(messages, through_index) >= min_head_share


_NOTE_ALLOWANCE = 2_000
"""Room left for the summary a fold will insert, so "it would fit" is not an
estimate that ignores the thing being added."""


def compaction_request(
    messages: Sequence[Message],
    *,
    through_index: int,
    params: ModelParams,
) -> ModelRequest:
    """Build the summarisation call. No tools, and no schema: this asks for prose."""
    transcript = "\n\n".join(render(message) for message in messages[:through_index])
    return ModelRequest(
        messages=(system(INSTRUCTION), user(transcript)),
        params=params,
        tools=(),
        tool_choice=ToolChoice.NONE,
    )


def render(message: Message) -> str:
    """One message as plain text, tool calls included.

    Tool calls carry most of what an agent actually did; a transcript that
    showed only prose would summarise a run as though it had done nothing.
    """
    body = message.text
    if message.tool_calls:
        invoked = ", ".join(
            f"{call.name}({_arguments(call.arguments)})" for call in message.tool_calls
        )
        body = f"{body}\n[called: {invoked}]" if body else f"[called: {invoked}]"
    return f"{_label(message)}: {body}"


def _label(message: Message) -> str:
    if message.is_tool_result:
        return f"tool:{message.name}" if message.name else "tool"
    return message.role.value


def _arguments(arguments: dict[str, object]) -> str:
    return ", ".join(f"{key}={_clip(str(value))}" for key, value in sorted(arguments.items()))


def _clip(value: str, limit: int = 120) -> str:
    return value if len(value) <= limit else f"{value[:limit]}..."


__all__ = [
    "INSTRUCTION",
    "KEEP_LAST",
    "MIN_HEAD_SHARE",
    "PER_MESSAGE_OVERHEAD",
    "TRIGGER_RATIO",
    "compaction_request",
    "worth_folding",
    "compaction_slice",
    "estimated_chars",
    "head_share",
    "render",
]
