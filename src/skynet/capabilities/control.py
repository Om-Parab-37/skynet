"""Control capabilities -- consumed by the policy, never executed.

``final_answer`` exists because completion is ambiguous when tool calling is
emulated: a weaker model produces conversational filler rather than stopping, so
finishing has to be an explicit, unambiguous act.
"""

from __future__ import annotations

from skynet.application.function_tool import tool
from skynet.capabilities.base import BoundTool
from skynet.domain.tools import ToolKind

FINAL_ANSWER = "final_answer"


class FinalAnswerTool(BoundTool):
    def __init__(self) -> None:
        self._tool = tool(
            name=FINAL_ANSWER,
            description=(
                "Provide the final answer and finish the task. "
                "Call this exactly once, when the objective is complete."
            ),
            kind=ToolKind.CONTROL,
        )(self._finish)

    async def _finish(self, answer: str) -> str:
        """The final answer to the user's task."""
        # Never invoked: AgentPolicy consumes CONTROL calls to complete the run.
        return answer


__all__ = ["FINAL_ANSWER", "FinalAnswerTool"]
