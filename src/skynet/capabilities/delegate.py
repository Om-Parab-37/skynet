"""Sub-agent delegation, expressed as an ordinary capability.

Modelling delegation as a tool rather than a special decision means the model
sees one uniform interface, and the runtime keeps a single routing point: the
effect executor sends ``DELEGATION`` calls to a ``SubAgentSpawner`` in-process,
or to a child workflow under Temporal. Hierarchies and parallel children need no
change to the agent loop.
"""

from __future__ import annotations

from skynet.application.function_tool import tool
from skynet.capabilities.base import BoundTool
from skynet.domain.permissions import Permission
from skynet.domain.tools import ToolKind

DELEGATE = "delegate"


class DelegateTool(BoundTool):
    def __init__(self) -> None:
        self._tool = tool(
            name=DELEGATE,
            description=(
                "Delegate a self-contained sub-task to a specialised agent and "
                "wait for its result. Use this when the sub-task needs focused "
                "work of its own."
            ),
            kind=ToolKind.DELEGATION,
            required_permissions={Permission.SPAWN_AGENT},
            timeout_s=600.0,
        )(self._delegate)

    async def _delegate(self, agent: str, objective: str) -> str:
        """Name of the agent to delegate to, and the objective to hand it."""
        # Never invoked: InProcessEffectExecutor intercepts DELEGATION calls.
        return ""


__all__ = ["DELEGATE", "DelegateTool"]
