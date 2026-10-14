"""Isolated execution port.

Covers the three things an agent needs from an execution environment: run a
command, and read/write files. Stronger isolation, resource limits and network
policy are properties of an *implementation*, not of this contract.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class ExecRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: tuple[str, ...]
    cwd: str | None = None
    env: dict[str, str] = {}
    stdin: str | None = None
    timeout_s: float = 30.0


class ExecResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class ResourceProfile(BaseModel):
    """Limits applied when a sandbox is created."""

    model_config = ConfigDict(frozen=True)

    memory_mb: int = 512
    cpus: float = 1.0
    pids: int = 128
    network: bool = False
    workdir_mb: int = 64


@runtime_checkable
class Sandbox(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def workdir(self) -> str: ...

    async def exec(self, request: ExecRequest) -> ExecResult: ...

    async def write_file(self, path: str, data: bytes) -> None: ...

    async def read_file(self, path: str) -> bytes: ...

    async def list_files(self, path: str | None = None) -> tuple[str, ...]:
        """Absolute paths of the regular files under ``path`` (default: workdir).

        Enumeration is what makes a workspace harvestable. Without it, anything
        an agent creates is visible only to the agent, and disappears with the
        sandbox.
        """
        ...


@runtime_checkable
class SandboxProvider(Protocol):
    def acquire(
        self,
        *,
        workspace_id: str,
        image: str | None = None,
        resources: ResourceProfile | None = None,
    ) -> AbstractAsyncContextManager[Sandbox]: ...

    async def exists(self, workspace_id: str) -> bool:
        """Has this workspace been created?

        Lets a caller avoid materialising a sandbox just to discover it is
        empty -- which for a container backend means starting one for nothing.
        """
        ...

    async def release(self, workspace_id: str) -> None:
        """Discard a workspace and everything it holds.

        A workspace outlives a single ``acquire`` -- consecutive tool calls in
        one run must see the same filesystem -- so something has to end it. The
        runtime calls this when a run reaches a terminal state; without it,
        isolated environments accumulate for as long as the process lives.
        """
        ...


__all__ = [
    "ExecRequest",
    "ExecResult",
    "ResourceProfile",
    "Sandbox",
    "SandboxProvider",
]
