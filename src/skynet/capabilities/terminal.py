"""Terminal capability.

Runs a command inside the agent's sandbox and reports stdout, stderr and the
exit code. Everything the model needs to reason about a failure is in the
observation, including a truncation marker when output is large.

Long-running and background processes are a future addition to the ``Sandbox``
port (an ``exec_async`` returning a handle); nothing here forecloses that.
"""

from __future__ import annotations

from skynet.application.function_tool import tool
from skynet.capabilities.base import BoundTool
from skynet.domain.permissions import Permission
from skynet.domain.tools import ToolContext
from skynet.ports.sandbox import ExecRequest, SandboxProvider


def format_exec(result, command: str) -> str:
    parts = [f"$ {command}", f"exit code: {result.exit_code}"]
    if result.timed_out:
        parts.append("(timed out)")
    if result.stdout.strip():
        parts.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr.strip():
        parts.append(f"stderr:\n{result.stderr.rstrip()}")
    if not result.stdout.strip() and not result.stderr.strip():
        parts.append("(no output)")
    return "\n".join(parts)


class TerminalTool(BoundTool):
    def __init__(self, sandboxes: SandboxProvider, *, default_timeout_s: float = 30.0) -> None:
        self._sandboxes = sandboxes
        self._default_timeout = default_timeout_s
        self._tool = tool(
            name="terminal",
            description=(
                "Run a shell command in the sandbox and return its stdout, stderr and exit code."
            ),
            required_permissions={Permission.TERMINAL},
            timeout_s=120.0,
        )(self._run)

    async def _run(
        self,
        command: str,
        ctx: ToolContext,
        cwd: str = "",
        timeout_s: float = 0.0,
    ) -> str:
        """The shell command to run, with an optional working directory."""
        async with self._sandboxes.acquire(
            workspace_id=ctx.workspace_id or str(ctx.run_id)
        ) as sandbox:
            result = await sandbox.exec(
                ExecRequest(
                    command=("/bin/sh", "-lc", command),
                    cwd=cwd or None,
                    timeout_s=timeout_s or self._default_timeout,
                )
            )
        return format_exec(result, command)


__all__ = ["TerminalTool", "format_exec"]
