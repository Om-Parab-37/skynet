"""Container sandbox.

Real isolation: a throwaway container per workspace, with memory, CPU, PID and
network limits applied at creation. This is where the resource-management and
network-policy requirements stop being aspirational.

The Docker SDK is synchronous, so calls run in threads.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import posixpath
import shlex
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from skynet.domain.errors import ConfigurationError, SandboxError
from skynet.ports.sandbox import ExecRequest, ExecResult, ResourceProfile

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.13-slim"
WORKDIR = "/workspace"

# Bytes are passed to the container inline, so keep clear of ARG_MAX.
_MAX_INLINE_WRITE = 512 * 1024


class DockerSandbox:
    def __init__(self, container: Any, workdir: str = WORKDIR) -> None:
        self._container = container
        self._workdir = workdir

    @property
    def id(self) -> str:
        return str(self._container.id)[:12]

    @property
    def workdir(self) -> str:
        return self._workdir

    async def exec(self, request: ExecRequest) -> ExecResult:
        started = time.monotonic()

        def _run() -> tuple[int, bytes, bytes]:
            result = self._container.exec_run(
                cmd=list(request.command),
                workdir=request.cwd or self._workdir,
                environment=dict(request.env) or None,
                demux=True,
            )
            output = result.output
            stdout, stderr = output if isinstance(output, tuple) else (output, b"")
            return result.exit_code or 0, stdout or b"", stderr or b""

        try:
            exit_code, stdout, stderr = await asyncio.wait_for(
                asyncio.to_thread(_run), timeout=request.timeout_s
            )
        except TimeoutError:
            return ExecResult(
                exit_code=124,
                stderr=f"command timed out after {request.timeout_s}s",
                timed_out=True,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(f"container exec failed: {exc}") from exc

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def write_file(self, path: str, data: bytes) -> None:
        """Write through an exec rather than ``put_archive``.

        The container runs with a read-only root filesystem, which Docker's
        archive-upload API refuses outright even when the target is the writable
        tmpfs workspace. Streaming the bytes through base64 keeps the immutable
        root -- a real isolation property -- instead of trading it away for
        convenience.
        """
        if len(data) > _MAX_INLINE_WRITE:
            raise SandboxError(
                f"file too large to write into the sandbox "
                f"({len(data)} bytes > {_MAX_INLINE_WRITE})"
            )
        directory = posixpath.dirname(path) or self._workdir
        encoded = base64.b64encode(data).decode()
        script = (
            f"mkdir -p {shlex.quote(directory)} && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        )

        result = await self.exec(ExecRequest(command=("/bin/sh", "-c", script), timeout_s=60.0))
        if result.exit_code != 0:
            raise SandboxError(f"failed to write {path}: {result.stderr.strip()}")

    async def read_file(self, path: str) -> bytes:
        """Read through an exec, for the same reason writes do.

        The workspace is a tmpfs mount, and Docker's archive API cannot see
        inside mounts -- it only serves the container's image layers. Going
        through the shell keeps file I/O working whatever the workspace is
        backed by.
        """
        result = await self.exec(
            ExecRequest(
                # No pipeline: a pipeline reports its *last* command's status,
                # which would mask a missing file behind a successful `tr`.
                command=("/bin/sh", "-c", f"base64 {shlex.quote(path)}"),
                timeout_s=60.0,
            )
        )
        if result.exit_code != 0:
            raise FileNotFoundError(path)
        try:
            return base64.b64decode(result.stdout)
        except (ValueError, binascii.Error) as exc:
            raise SandboxError(f"could not decode {path}: {exc}") from exc

    async def list_files(self, path: str | None = None) -> tuple[str, ...]:
        root = path or self._workdir
        result = await self.exec(
            ExecRequest(
                command=("/bin/sh", "-c", f"find {shlex.quote(root)} -type f 2>/dev/null"),
                timeout_s=30.0,
            )
        )
        if result.exit_code != 0:
            return ()
        return tuple(sorted(line for line in result.stdout.splitlines() if line.strip()))


class DockerSandboxProvider:
    """One container per *workspace*, reused across tool calls.

    Each tool acquires the sandbox independently, so a container per
    acquisition would silently discard everything the previous tool wrote.
    Keying by ``workspace_id`` gives a run a stable filesystem while keeping
    different runs isolated from one another.

    Idle containers are reaped on the next acquisition, which bounds the
    container count without needing a background task.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        resources: ResourceProfile | None = None,
        client: Any = None,
        idle_timeout_s: float = 900.0,
        max_age_s: float = 3600.0,
    ) -> None:
        self._image = image
        self._resources = resources or ResourceProfile()
        self._client = client
        self._idle_timeout = idle_timeout_s
        self._max_age = max_age_s
        self._containers: dict[str, Any] = {}
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def _docker(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import docker
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigurationError(
                "docker is not installed; install the 'sandbox' extra"
            ) from exc
        self._client = docker.from_env()
        return self._client

    @asynccontextmanager
    async def acquire(
        self,
        *,
        workspace_id: str,
        image: str | None = None,
        resources: ResourceProfile | None = None,
    ) -> AsyncIterator[DockerSandbox]:
        async with self._lock:
            await self._reap_idle()
            container = self._containers.get(workspace_id)
            if container is None:
                # The in-memory cache does not survive a worker restart, but the
                # container does. Adopting it by label keeps a resumed run's
                # workspace intact and avoids leaking an orphan alongside a new
                # one.
                container = await self._adopt(workspace_id)
            if container is None:
                container = await self._create(workspace_id, image, resources)
            self._containers[workspace_id] = container
            self._last_used[workspace_id] = time.monotonic()

        yield DockerSandbox(container)

        async with self._lock:
            self._last_used[workspace_id] = time.monotonic()

    async def _adopt(self, workspace_id: str) -> Any | None:
        """Reclaim a running container this provider created before a restart."""
        client = self._docker()

        def _find() -> Any | None:
            found = client.containers.list(
                filters={"label": f"skynet.workspace={workspace_id}", "status": "running"}
            )
            return found[0] if found else None

        try:
            return await asyncio.to_thread(_find)
        except Exception as exc:  # noqa: BLE001 - adoption is best-effort
            logger.debug("could not adopt sandbox for %s: %s", workspace_id, exc)
            return None

    async def _create(
        self, workspace_id: str, image: str | None, resources: ResourceProfile | None
    ) -> Any:
        limits = resources or self._resources
        client = self._docker()

        def _run() -> Any:
            return client.containers.run(
                image or self._image,
                command=["sleep", "infinity"],
                detach=True,
                working_dir=WORKDIR,
                labels={"skynet.workspace": workspace_id},
                mem_limit=f"{limits.memory_mb}m",
                nano_cpus=int(limits.cpus * 1_000_000_000),
                pids_limit=limits.pids,
                network_disabled=not limits.network,
                # An immutable root with a writable tmpfs workspace means a
                # runaway process can neither fill the host disk nor persist
                # anything outside its own workspace.
                read_only=True,
                tmpfs={WORKDIR: f"rw,size={limits.workdir_mb}m,exec"},
                auto_remove=False,
            )

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(f"could not create sandbox container: {exc}") from exc

    async def _reap_idle(self) -> None:
        """Discard workspaces this provider knows are idle."""
        now = time.monotonic()
        stale = [
            workspace
            for workspace, last in self._last_used.items()
            if now - last > self._idle_timeout
        ]
        for workspace in stale:
            await self._destroy(workspace)

    async def reap_abandoned(self) -> int:
        """Remove sandbox containers older than ``max_age_s``.

        In-memory bookkeeping does not survive a restart, so a crashed worker
        would otherwise strand its containers forever. Age is read from Docker
        itself, which is the only record that outlives this process. Called at
        worker startup and before each acquisition.
        """
        client = self._docker()
        tracked = {c.id for c in self._containers.values()}
        cutoff = datetime.now(UTC) - timedelta(seconds=self._max_age)

        def _sweep() -> int:
            removed = 0
            for container in client.containers.list(
                all=True, filters={"label": "skynet.workspace"}
            ):
                if container.id in tracked:
                    continue
                started = container.attrs.get("Created")
                if started and _parse_time(started) > cutoff:
                    continue
                with contextlib.suppress(Exception):
                    container.remove(force=True)
                    removed += 1
            return removed

        try:
            return await asyncio.to_thread(_sweep)
        except Exception as exc:  # noqa: BLE001 - housekeeping must not break a run
            logger.debug("could not reap abandoned sandboxes: %s", exc)
            return 0

    async def exists(self, workspace_id: str) -> bool:
        if workspace_id in self._containers:
            return True
        return await self._adopt(workspace_id) is not None

    async def release(self, workspace_id: str) -> None:
        """Discard a workspace. Called when its run reaches a terminal state."""
        async with self._lock:
            await self._destroy(workspace_id)
            # The container may predate this provider instance (adopted after a
            # restart, or stranded by a crash), so also sweep by label.
            await self._destroy_by_label(workspace_id)

    async def _destroy_by_label(self, workspace_id: str) -> None:
        client = self._docker()

        def _sweep() -> None:
            for container in client.containers.list(
                all=True, filters={"label": f"skynet.workspace={workspace_id}"}
            ):
                with contextlib.suppress(Exception):
                    container.remove(force=True)

        with contextlib.suppress(Exception):
            await asyncio.to_thread(_sweep)

    async def _destroy(self, workspace_id: str) -> None:
        container = self._containers.pop(workspace_id, None)
        self._last_used.pop(workspace_id, None)
        if container is not None:
            await asyncio.to_thread(_remove, container)

    async def close(self) -> None:
        async with self._lock:
            for workspace in list(self._containers):
                await self._destroy(workspace)


def _parse_time(value: str) -> datetime:
    """Docker timestamps carry nanosecond precision that fromisoformat rejects."""
    cleaned = value.replace("Z", "+00:00")
    if "." in cleaned:
        head, _, tail = cleaned.partition(".")
        fraction, sign, offset = tail.partition("+")
        cleaned = f"{head}.{fraction[:6]}{sign}{offset}" if sign else f"{head}.{fraction[:6]}"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _remove(container: Any) -> None:
    # Teardown must never mask the error that caused it.
    with contextlib.suppress(Exception):
        container.remove(force=True)


__all__ = ["DEFAULT_IMAGE", "WORKDIR", "DockerSandbox", "DockerSandboxProvider"]
