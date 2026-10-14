"""Code execution capability.

The prototype runs Python, which is the only language-specific thing here: the
code is written into the sandbox as a file and executed by an interpreter
command. Supporting another language means another instance of this class with a
different filename and command -- the sandbox port and the agent loop are
unaffected.
"""

from __future__ import annotations

from skynet.application.function_tool import tool
from skynet.capabilities.base import BoundTool
from skynet.domain.permissions import Permission
from skynet.domain.tools import ToolContext
from skynet.ports.sandbox import ExecRequest, SandboxProvider


class CodeExecTool(BoundTool):
    def __init__(
        self,
        sandboxes: SandboxProvider,
        *,
        name: str = "python_exec",
        language: str = "Python",
        filename: str = "_skynet_snippet.py",
        interpreter: tuple[str, ...] = ("python3",),
        default_timeout_s: float = 60.0,
    ) -> None:
        self._sandboxes = sandboxes
        self._filename = filename
        self._interpreter = interpreter
        self._default_timeout = default_timeout_s
        self._tool = tool(
            name=name,
            description=(
                f"Execute a {language} snippet in the sandbox and return its "
                "output. Print the values you want to see."
            ),
            required_permissions={Permission.CODE_EXEC},
            timeout_s=180.0,
        )(self._run)

    async def _run(self, code: str, ctx: ToolContext, timeout_s: float = 0.0) -> str:
        """The source code to execute."""
        async with self._sandboxes.acquire(
            workspace_id=ctx.workspace_id or str(ctx.run_id)
        ) as sandbox:
            path = f"{sandbox.workdir}/{self._filename}"
            await sandbox.write_file(path, code.encode())
            result = await sandbox.exec(
                ExecRequest(
                    command=(*self._interpreter, path),
                    cwd=sandbox.workdir,
                    timeout_s=timeout_s or self._default_timeout,
                )
            )

        if result.timed_out:
            return f"Execution timed out after {timeout_s or self._default_timeout}s."
        parts = [f"exit code: {result.exit_code}"]
        if result.stdout.strip():
            parts.append(f"output:\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            parts.append(f"errors:\n{result.stderr.rstrip()}")
        if len(parts) == 1:
            parts.append("(no output -- remember to print() your results)")
        return "\n".join(parts)


__all__ = ["CodeExecTool"]
