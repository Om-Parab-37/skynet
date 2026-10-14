"""Conversation primitives.

These are the harness's *own* representation of a chat exchange. They
deliberately do not mirror any provider's wire format: adapters translate at the
boundary so that no provider vocabulary reaches the core.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from skynet.domain.ids import CallId


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A model's request to invoke one capability."""

    model_config = ConfigDict(frozen=True)

    id: CallId
    name: str
    arguments: dict[str, Any] = {}
    raw: str | None = None
    """Original text the call was parsed from, when tool calling was emulated."""
    provider_data: dict[str, Any] = {}
    """Opaque data an adapter must echo back on the next turn.

    Some providers attach a continuation token to a tool call and reject the
    conversation if it is not returned verbatim -- Gemini's ``thought_signature``
    is one. The core never reads this; it only has to avoid losing it."""


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: CallId | None = None
    """Set when ``role is Role.TOOL``: which call this message answers."""
    name: str | None = None
    """Tool name, when ``role is Role.TOOL``."""
    reasoning: str | None = None
    """Separate reasoning channel emitted by thinking models.

    Kept apart from ``content`` because it is not the answer -- but when a model
    finishes with an empty ``content``, it is the only answer there is."""

    # Multimodal support is an additive change: a `parts` field with a default
    # keeps every existing construction site valid.

    @property
    def is_tool_result(self) -> bool:
        return self.role is Role.TOOL

    @property
    def text(self) -> str:
        """What the message actually says, whichever channel carried it."""
        return self.content.strip() or (self.reasoning or "").strip()


def system(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def assistant(content: str = "", *, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
    return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)


def tool_result(call_id: CallId, name: str, content: str) -> Message:
    return Message(role=Role.TOOL, content=content, tool_call_id=call_id, name=name)


__all__ = [
    "Message",
    "Role",
    "ToolCall",
    "assistant",
    "system",
    "tool_result",
    "user",
]
