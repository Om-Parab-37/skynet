"""Subprocess sandbox.

Isolation here is *workspace* isolation, not security isolation: each workspace
gets its own directory and every command has a timeout. It exists so the core
suite and a laptop demo can run without Docker, and to keep the ``Sandbox`` port
honest by having two real implementations.

Use ``DockerSandboxProvider`` when you need actual containment.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from skynet.domain.errors import SandboxError
from skynet.ports.sandbox import ExecRequest, ExecResult, ResourceProfile


class LocalSandbox:
    def __init__(self, sandbox_id: str, workdir: Path) -> None:
        self._id = sandbox_id
        self._workdir = workdir

    @property
    def id(self) -> str:
        return self._id

    @property
    def workdir(self) -> str:
        return str(self._workdir)

    async def exec(self, request: ExecRequest) -> ExecResult:
        cwd = self._resolve(request.cwd) if request.cwd else self._workdir
        started = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            *request.command,
            cwd=str(cwd),
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self._workdir), **request.env},
            stdin=asyncio.subprocess.PIPE if request.stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request.stdin.encode() if request.stdin else None),
                timeout=request.timeout_s,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return ExecResult(
                exit_code=124,
                stderr=f"command timed out after {request.timeout_s}s",
                timed_out=True,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        return ExecResult(
            exit_code=process.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def write_file(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def read_file(self, path: str) -> bytes:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        return await asyncio.to_thread(target.read_bytes)

    async def list_files(self, path: str | None = None) -> tuple[str, ...]:
        root = self._resolve(path) if path else self._workdir

        def _walk() -> tuple[str, ...]:
            if not root.is_dir():
                return ()
            return tuple(sorted(str(p) for p in root.rglob("*") if p.is_file()))

        return await asyncio.to_thread(_walk)

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        resolved = (candidate if candidate.is_absolute() else self._workdir / candidate).resolve()
        # Containment: a tool must not be able to escape its workspace by path.
        if not resolved.is_relative_to(self._workdir.resolve()):
            raise SandboxError(f"path escapes the sandbox workspace: {path}")
        return resolved


class LocalSandboxProvider:
    def __init__(self, root: str | Path = "/tmp/skynet-workspaces", cleanup: bool = False) -> None:
        self._root = Path(root)
        self._cleanup = cleanup

    @asynccontextmanager
    async def acquire(
        self,
        *,
        workspace_id: str,
        image: str | None = None,
        resources: ResourceProfile | None = None,
    ) -> AsyncIterator[LocalSandbox]:
        workdir = self._root / workspace_id
        workdir.mkdir(parents=True, exist_ok=True)
        # The workspace deliberately outlives this context manager: consecutive
        # tool calls in one run must see the same files. `release` ends it.
        yield LocalSandbox(f"local-{workspace_id}", workdir)

    async def exists(self, workspace_id: str) -> bool:
        return (self._root / workspace_id).is_dir()

    async def release(self, workspace_id: str) -> None:
        await asyncio.to_thread(shutil.rmtree, self._root / workspace_id, True)


__all__ = ["LocalSandbox", "LocalSandboxProvider"]
