"""HTTP contracts.

Kept separate from the domain models on purpose: the wire format must be free to
evolve without dragging the domain with it, and domain models carry internals
(full message history, permissions, trace linkage) that should not leak out of
the process by default.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from skynet.domain.run import AgentRun


class SubmitRunRequest(BaseModel):
    objective: str = Field(min_length=1, description="What the agent should accomplish")
    agent: str = "assistant"
    input: dict[str, Any] = {}
    metadata: dict[str, str] = {}


class SubmitRunResponse(BaseModel):
    run_id: str
    agent: str
    status: str


class RunSummary(BaseModel):
    run_id: str
    agent: str
    status: str
    objective: str
    iterations: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    children: int = 0
    output: str | None = None
    error: str | None = None

    @classmethod
    def of(cls, run: AgentRun) -> RunSummary:
        return cls(
            run_id=str(run.run_id),
            agent=run.spec.name,
            status=str(run.status),
            objective=run.task.objective,
            iterations=run.iteration,
            tool_calls=run.tool_call_count,
            total_tokens=run.usage.total,
            children=len(run.child_runs),
            output=run.result.output if run.result else None,
            error=run.last_error.message if run.last_error else None,
        )


class StepView(BaseModel):
    index: int
    thought: str
    tools: list[dict[str, Any]] = []


class RunDetail(RunSummary):
    steps: list[StepView] = []
    artifacts: list[str] = []

    @classmethod
    def of(cls, run: AgentRun) -> RunDetail:
        summary = RunSummary.of(run).model_dump()
        steps = [
            StepView(
                index=step.index,
                thought=step.assistant.content,
                tools=[
                    {"name": r.name, "status": str(r.status), "output": r.content[:2000]}
                    for r in step.tool_results
                ],
            )
            for step in run.steps
        ]
        # Two sources: oversized tool output spilled during the run, and files
        # harvested from the workspace, which land on the result.
        artifacts = [
            ref.uri for step in run.steps for r in step.tool_results for ref in r.artifacts
        ]
        if run.result is not None:
            artifacts += [ref.uri for ref in run.result.artifacts]
        return cls(**summary, steps=steps, artifacts=artifacts)


class EventView(BaseModel):
    kind: str
    seq: int
    at: datetime | None = None
    detail: dict[str, Any] = {}


class ApprovalRequest(BaseModel):
    call_id: str
    granted: bool = True
    actor: str = "user"


class CancelRequest(BaseModel):
    reason: str = ""


class AgentView(BaseModel):
    name: str
    description: str
    tools: list[str]
    sub_agents: list[str]
    permissions: list[str]


class ToolView(BaseModel):
    name: str
    description: str
    kind: str
    required_permissions: list[str]
    requires_approval: bool
    parameters: dict[str, Any]


__all__ = [
    "AgentView",
    "ApprovalRequest",
    "CancelRequest",
    "EventView",
    "RunDetail",
    "RunSummary",
    "StepView",
    "SubmitRunRequest",
    "SubmitRunResponse",
    "ToolView",
]
