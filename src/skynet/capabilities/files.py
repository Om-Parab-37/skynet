"""Filesystem capabilities scoped to the sandbox."""

from __future__ import annotations

from skynet.application.function_tool import tool
from skynet.capabilities.base import BoundTool
from skynet.domain.permissions import Permission
from skynet.domain.tools import ToolContext
from skynet.ports.sandbox import SandboxProvider


class WriteFileTool(BoundTool):
    def __init__(self, sandboxes: SandboxProvider) -> None:
        self._sandboxes = sandboxes
        self._tool = tool(
            name="write_file",
            description="Write text to a file in the sandbox workspace.",
            required_permissions={Permission.FILESYSTEM},
        )(self._write)

    async def _write(self, path: str, content: str, ctx: ToolContext) -> str:
        """Path relative to the workspace, and the text to write."""
        async with self._sandboxes.acquire(
            workspace_id=ctx.workspace_id or str(ctx.run_id)
        ) as sandbox:
            full = path if path.startswith("/") else f"{sandbox.workdir}/{path}"
            await sandbox.write_file(full, content.encode())
        return f"Wrote {len(content)} characters to {path}."


class ReadFileTool(BoundTool):
    def __init__(self, sandboxes: SandboxProvider, *, max_chars: int = 20_000) -> None:
        self._sandboxes = sandboxes
        self._max_chars = max_chars
        self._tool = tool(
            name="read_file",
            description="Read a text file from the sandbox workspace.",
            required_permissions={Permission.FILESYSTEM},
            idempotent=True,
        )(self._read)

    async def _read(self, path: str, ctx: ToolContext) -> str:
        """Path relative to the workspace."""
        async with self._sandboxes.acquire(
            workspace_id=ctx.workspace_id or str(ctx.run_id)
        ) as sandbox:
            full = path if path.startswith("/") else f"{sandbox.workdir}/{path}"
            try:
                data = await sandbox.read_file(full)
            except (KeyError, FileNotFoundError):
                return f"ERROR: no such file: {path}"
        text = data.decode(errors="replace")
        if len(text) > self._max_chars:
            return text[: self._max_chars] + "\n[truncated]"
        return text


__all__ = ["ReadFileTool", "WriteFileTool"]
