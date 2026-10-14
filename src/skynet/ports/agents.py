"""Agent lookup, execution and delegation ports.

``AgentRunner`` is the abstraction over *where* a run executes. The prototype
ships an in-process runner and a Temporal runner; the API and CLI depend only on
this contract, so switching execution engines is configuration.

``SubAgentSpawner`` is separated (interface segregation) because the effect
executor needs only the ability to spawn a child, not the full runner surface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from skynet.domain.agent import AgentContext, AgentSpec
from skynet.domain.ids import CallId, RunId
from skynet.domain.run import AgentRun
from skynet.domain.task import AgentResult, Task


class RunRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    spec: AgentSpec
    task: Task
    context: AgentContext


@runtime_checkable
class AgentCatalog(Protocol):
    """Pure and synchronous, so it is safe inside a Temporal workflow."""

    def get(self, name: str) -> AgentSpec: ...

    def names(self) -> tuple[str, ...]: ...

    def all(self) -> tuple[AgentSpec, ...]: ...


@runtime_checkable
class RunHandle(Protocol):
    @property
    def run_id(self) -> RunId: ...

    async def result(self) -> AgentRun: ...

    async def snapshot(self) -> AgentRun:
        """Current state without waiting for the run to finish.

        Separate from ``result`` because observers must be able to read progress
        cheaply; racing a completion future with a short timeout is not a status
        check.
        """
        ...

    async def cancel(self, reason: str = "") -> None: ...

    async def approve(self, call_id: CallId, granted: bool, actor: str = "user") -> None: ...


@runtime_checkable
class AgentRunner(Protocol):
    async def start(self, request: RunRequest) -> RunHandle: ...

    async def run(self, request: RunRequest) -> AgentRun: ...


@runtime_checkable
class SubAgentSpawner(Protocol):
    async def spawn(
        self, *, agent: str, task: Task, parent: AgentContext, call_id: CallId
    ) -> tuple[RunId, AgentResult]: ...


__all__ = [
    "AgentCatalog",
    "AgentRunner",
    "RunHandle",
    "RunRequest",
    "SubAgentSpawner",
]
