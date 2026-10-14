"""Helpers for describing work to a tracing backend.

Kept in the application layer because both execution paths need the identical
representation -- an in-process run and a Temporal activity must produce
comparable traces, or comparing them is meaningless.
"""

from __future__ import annotations

from typing import Any

from skynet.domain.models import ModelRequest


def conversation_input(request: ModelRequest) -> list[dict[str, Any]]:
    """The messages a model was actually shown.

    This is a generation's input, and it is what makes a trace useful for
    debugging: without it you can see that a call happened but not what was
    asked.
    """
    return [
        {
            "role": str(message.role),
            "content": message.content,
            **(
                {"tool_calls": [call.name for call in message.tool_calls]}
                if message.tool_calls
                else {}
            ),
        }
        for message in request.messages
    ]


__all__ = ["conversation_input"]
