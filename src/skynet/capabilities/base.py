"""Shared plumbing for capability classes.

A capability that needs collaborators (a sandbox, a memory store) is a class
with constructor injection; the schema and argument validation still come from
the decorated method, so there is exactly one way tools are described.
"""

from __future__ import annotations

from skynet.application.function_tool import FunctionTool
from skynet.domain.messages import ToolCall
from skynet.domain.tools import ToolContext, ToolResult, ToolSpec


class BoundTool:
    """Delegates the ``Tool`` port to a ``FunctionTool`` built from a method."""

    _tool: FunctionTool

    @property
    def spec(self) -> ToolSpec:
        return self._tool.spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        return await self._tool.invoke(call, ctx)


__all__ = ["BoundTool"]
