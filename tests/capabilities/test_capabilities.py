"""Built-in capabilities, exercised against fakes."""

from __future__ import annotations

import pytest

from skynet.capabilities.control import FinalAnswerTool
from skynet.capabilities.delegate import DelegateTool
from skynet.capabilities.files import ReadFileTool, WriteFileTool
from skynet.capabilities.memory import MemorySearchTool, MemoryWriteTool
from skynet.capabilities.python_exec import CodeExecTool
from skynet.capabilities.terminal import TerminalTool
from skynet.domain.ids import CallId, RunId
from skynet.domain.messages import ToolCall
from skynet.domain.permissions import Permission, PermissionSet
from skynet.domain.tools import ToolContext, ToolKind, ToolStatus
from skynet.ports.sandbox import ExecResult
from skynet.testing.fakes import (
    FakeSandbox,
    FakeSandboxProvider,
    FrozenClock,
    InMemoryMemoryStore,
    SequentialIdGenerator,
)


def context(**kwargs) -> ToolContext:
    kwargs.setdefault("run_id", RunId("run-1"))
    kwargs.setdefault("agent_name", "demo")
    kwargs.setdefault("permissions", PermissionSet(grants=Permission.all()))
    return ToolContext(**kwargs)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=CallId("c1"), name=name, arguments=arguments)


# -- terminal -----------------------------------------------------------------


async def test_terminal_reports_stdout_and_exit_code():
    sandbox = FakeSandbox(responses={"/bin/sh": ExecResult(exit_code=0, stdout="hello")})
    tool = TerminalTool(FakeSandboxProvider(sandbox))
    result = await tool.invoke(call("terminal", command="echo hello"), context())

    assert result.status is ToolStatus.OK
    assert "exit code: 0" in result.content
    assert "hello" in result.content


async def test_terminal_surfaces_stderr_and_failure():
    sandbox = FakeSandbox(responses={"/bin/sh": ExecResult(exit_code=127, stderr="not found")})
    tool = TerminalTool(FakeSandboxProvider(sandbox))
    result = await tool.invoke(call("terminal", command="nope"), context())

    assert "exit code: 127" in result.content
    assert "not found" in result.content


async def test_terminal_reports_a_timeout_rather_than_raising():
    sandbox = FakeSandbox(responses={"/bin/sh": ExecResult(exit_code=124, timed_out=True)})
    tool = TerminalTool(FakeSandboxProvider(sandbox))
    result = await tool.invoke(call("terminal", command="sleep 99"), context())

    assert result.status is ToolStatus.OK
    assert "timed out" in result.content


def test_terminal_requires_the_terminal_permission():
    assert TerminalTool(FakeSandboxProvider()).spec.required_permissions == frozenset(
        {Permission.TERMINAL}
    )


# -- code execution -----------------------------------------------------------


async def test_code_exec_writes_the_snippet_then_runs_it():
    sandbox = FakeSandbox(responses={"python3": ExecResult(exit_code=0, stdout="42")})
    tool = CodeExecTool(FakeSandboxProvider(sandbox))
    result = await tool.invoke(call("python_exec", code="print(6*7)"), context())

    assert "42" in result.content
    assert await sandbox.read_file("/workspace/_skynet_snippet.py") == b"print(6*7)"
    assert sandbox.commands[-1][0] == "python3"


async def test_code_exec_reports_errors_without_failing_the_tool():
    sandbox = FakeSandbox(responses={"python3": ExecResult(exit_code=1, stderr="NameError: x")})
    tool = CodeExecTool(FakeSandboxProvider(sandbox))
    result = await tool.invoke(call("python_exec", code="print(x)"), context())

    assert result.status is ToolStatus.OK
    assert "exit code: 1" in result.content
    assert "NameError" in result.content


async def test_code_exec_nudges_when_nothing_was_printed():
    sandbox = FakeSandbox(responses={"python3": ExecResult(exit_code=0)})
    tool = CodeExecTool(FakeSandboxProvider(sandbox))
    result = await tool.invoke(call("python_exec", code="6*7"), context())

    assert "print()" in result.content


def test_another_language_needs_no_new_class():
    """The prototype ships Python; a second language is configuration."""
    tool = CodeExecTool(
        FakeSandboxProvider(),
        name="node_exec",
        language="JavaScript",
        filename="snippet.js",
        interpreter=("node",),
    )
    assert tool.spec.name == "node_exec"
    assert "JavaScript" in tool.spec.description


# -- files --------------------------------------------------------------------


async def test_write_then_read_a_file():
    sandbox = FakeSandbox()
    provider = FakeSandboxProvider(sandbox)
    await WriteFileTool(provider).invoke(
        call("write_file", path="note.txt", content="saved"), context()
    )
    result = await ReadFileTool(provider).invoke(call("read_file", path="note.txt"), context())
    assert result.content == "saved"


async def test_reading_a_missing_file_is_an_observation():
    result = await ReadFileTool(FakeSandboxProvider()).invoke(
        call("read_file", path="ghost.txt"), context()
    )
    assert "no such file" in result.content


# -- memory -------------------------------------------------------------------


async def test_memory_write_then_search_round_trips():
    store = InMemoryMemoryStore()
    ctx = context()
    await MemoryWriteTool(store, SequentialIdGenerator(), FrozenClock()).invoke(
        call("memory_write", content="the deploy key lives in vault", tags="ops"), ctx
    )
    result = await MemorySearchTool(store, FrozenClock()).invoke(
        call("memory_search", query="deploy key"), ctx
    )
    assert "vault" in result.content


async def test_memory_is_scoped_per_agent():
    store = InMemoryMemoryStore()
    await MemoryWriteTool(store, SequentialIdGenerator(), FrozenClock()).invoke(
        call("memory_write", content="alpha secret"), context(agent_name="alpha")
    )
    result = await MemorySearchTool(store, FrozenClock()).invoke(
        call("memory_search", query="alpha secret"), context(agent_name="beta")
    )
    assert "No matching memories" in result.content


async def test_memory_search_reports_no_results_clearly():
    result = await MemorySearchTool(InMemoryMemoryStore(), FrozenClock()).invoke(
        call("memory_search", query="anything"), context()
    )
    assert "No matching memories" in result.content


# -- control and delegation ---------------------------------------------------


def test_final_answer_is_a_control_tool():
    spec = FinalAnswerTool().spec
    assert spec.kind is ToolKind.CONTROL
    assert spec.required_permissions == frozenset()


def test_delegate_is_a_delegation_tool_needing_spawn_permission():
    spec = DelegateTool().spec
    assert spec.kind is ToolKind.DELEGATION
    assert spec.required_permissions == frozenset({Permission.SPAWN_AGENT})
    assert set(spec.parameters["required"]) == {"agent", "objective"}


# -- schema generation --------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        TerminalTool(FakeSandboxProvider()),
        CodeExecTool(FakeSandboxProvider()),
        ReadFileTool(FakeSandboxProvider()),
        WriteFileTool(FakeSandboxProvider()),
        MemorySearchTool(InMemoryMemoryStore(), FrozenClock()),
        DelegateTool(),
        FinalAnswerTool(),
    ],
    ids=lambda t: t.spec.name,
)
def test_every_tool_publishes_a_valid_json_schema(tool):
    schema = tool.spec.parameters
    assert schema["type"] == "object"
    assert "properties" in schema
    # `ctx` is injected by the runtime and must never be shown to the model.
    assert "ctx" not in schema["properties"]
    assert tool.spec.description.strip()
