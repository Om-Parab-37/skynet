"""Preserving what a run produced.

An agent's sandbox is deliberately disposable, which means anything it writes
there disappears when the run ends. That is fine for scratch files and wrong for
results: a report the agent was asked to produce should outlive the environment
that produced it.

The archiver copies a workspace into object storage just before the workspace is
released, and returns references the run's result carries. It is bounded on
purpose -- an agent can fill a workspace, and an unbounded copy would turn a
runaway run into a storage incident.
"""

from __future__ import annotations

import logging
import posixpath

from skynet.domain.ids import RunId
from skynet.domain.storage import ObjectRef
from skynet.ports.sandbox import SandboxProvider
from skynet.ports.storage import ObjectStore

logger = logging.getLogger(__name__)

# Files the runtime itself put there; they are noise in a result listing.
DEFAULT_EXCLUDES: tuple[str, ...] = ("_skynet_snippet.py",)

_CONTENT_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".py": "text/x-python",
    ".html": "text/html",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def content_type_for(path: str) -> str:
    _, _, suffix = path.rpartition(".")
    return _CONTENT_TYPES.get(f".{suffix.lower()}", "application/octet-stream")


class WorkspaceArchiver:
    """Copies a run's workspace into object storage."""

    def __init__(
        self,
        sandboxes: SandboxProvider | None,
        objects: ObjectStore,
        *,
        max_files: int = 50,
        max_file_bytes: int = 5 * 1024 * 1024,
        max_total_bytes: int = 25 * 1024 * 1024,
        excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    ) -> None:
        self._sandboxes = sandboxes
        self._objects = objects
        self._max_files = max_files
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._excludes = excludes

    async def harvest(self, *, run_id: RunId, workspace_id: str) -> tuple[ObjectRef, ...]:
        """Store the workspace's files. Never raises -- a run is already done."""
        if self._sandboxes is None:
            return ()
        try:
            if not await self._sandboxes.exists(workspace_id):
                # Nothing was ever created; do not materialise a sandbox to
                # discover that, which for a container backend costs a start-up.
                return ()
            return await self._copy(run_id, workspace_id)
        except Exception:  # noqa: BLE001 - losing artifacts must not fail a run
            logger.warning("could not harvest workspace for run %s", run_id, exc_info=True)
            return ()

    async def _copy(self, run_id: RunId, workspace_id: str) -> tuple[ObjectRef, ...]:
        assert self._sandboxes is not None
        refs: list[ObjectRef] = []
        budget = self._max_total_bytes

        async with self._sandboxes.acquire(workspace_id=workspace_id) as sandbox:
            paths = [p for p in await sandbox.list_files() if not self._excluded(p)]
            for path in paths[: self._max_files]:
                try:
                    data = await sandbox.read_file(path)
                except Exception:  # noqa: BLE001 - skip what we cannot read
                    logger.debug("skipping unreadable artifact %s", path)
                    continue

                if len(data) > self._max_file_bytes or len(data) > budget:
                    logger.debug("skipping oversized artifact %s (%d bytes)", path, len(data))
                    continue
                budget -= len(data)

                relative = posixpath.relpath(path, sandbox.workdir)
                key = f"runs/{run_id}/workspace/{relative}"
                refs.append(
                    await self._objects.put(
                        key,
                        data,
                        content_type=content_type_for(path),
                        metadata={"run_id": str(run_id), "source_path": path},
                    )
                )

        if len(paths) > self._max_files:
            logger.warning(
                "run %s produced %d files; stored the first %d",
                run_id,
                len(paths),
                self._max_files,
            )
        return tuple(refs)

    def _excluded(self, path: str) -> bool:
        name = posixpath.basename(path)
        return name in self._excludes


__all__ = ["DEFAULT_EXCLUDES", "WorkspaceArchiver", "content_type_for"]
