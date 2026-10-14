"""Files an agent produces must outlive the sandbox that produced them.

The sandbox is disposable by design, so without harvesting, a report the agent
was explicitly asked to write is destroyed moments after it is written -- and
`GET /v1/runs/{id}/artifacts` reports nothing for a run that demonstrably
produced files.
"""

from __future__ import annotations

from skynet.application.artifacts import WorkspaceArchiver, content_type_for
from skynet.application.runner import InProcessAgentRunner
from skynet.capabilities.files import WriteFileTool
from skynet.domain.ids import RunId
from skynet.domain.run import AgentStatus
from skynet.ports.agents import RunRequest
from skynet.testing.builders import (
    build_services,
    call,
    calls,
    make_context,
    make_spec,
    make_task,
    says,
)
from skynet.testing.fakes import (
    FakeSandbox,
    FakeSandboxProvider,
    InMemoryObjectStore,
    ScriptedModelProvider,
)


def workspace(files: dict[str, bytes]) -> FakeSandboxProvider:
    return FakeSandboxProvider(FakeSandbox(files=files))


# -- the archiver in isolation ------------------------------------------------


async def test_workspace_files_are_copied_to_object_storage():
    objects = InMemoryObjectStore()
    sandboxes = workspace({"/workspace/report.md": b"# Report", "/workspace/data.csv": b"1\n2\n"})

    refs = await WorkspaceArchiver(sandboxes, objects).harvest(
        run_id=RunId("run-1"), workspace_id="run-1"
    )

    keys = {r.key for r in refs}
    assert keys == {"runs/run-1/workspace/report.md", "runs/run-1/workspace/data.csv"}
    stored = await objects.get(next(r for r in refs if r.key.endswith("report.md")))
    assert stored == b"# Report"


async def test_content_types_are_inferred():
    objects = InMemoryObjectStore()
    sandboxes = workspace({"/workspace/report.md": b"x", "/workspace/blob.bin": b"y"})
    refs = await WorkspaceArchiver(sandboxes, objects).harvest(run_id=RunId("r"), workspace_id="r")
    types = {r.key.rsplit("/", 1)[-1]: r.content_type for r in refs}
    assert types["report.md"] == "text/markdown"
    assert types["blob.bin"] == "application/octet-stream"


async def test_runtime_scratch_files_are_excluded():
    objects = InMemoryObjectStore()
    sandboxes = workspace(
        {"/workspace/_skynet_snippet.py": b"print(1)", "/workspace/kept.txt": b"ok"}
    )
    refs = await WorkspaceArchiver(sandboxes, objects).harvest(run_id=RunId("r"), workspace_id="r")
    assert [r.key.rsplit("/", 1)[-1] for r in refs] == ["kept.txt"]


async def test_oversized_files_are_skipped_not_fatal():
    """An agent can fill a workspace; harvesting must stay bounded."""
    objects = InMemoryObjectStore()
    sandboxes = workspace({"/workspace/big.bin": b"x" * 5_000, "/workspace/small.txt": b"ok"})
    refs = await WorkspaceArchiver(sandboxes, objects, max_file_bytes=1_000).harvest(
        run_id=RunId("r"), workspace_id="r"
    )
    assert [r.key.rsplit("/", 1)[-1] for r in refs] == ["small.txt"]


async def test_file_count_is_capped():
    objects = InMemoryObjectStore()
    sandboxes = workspace({f"/workspace/f{i}.txt": b"x" for i in range(30)})
    refs = await WorkspaceArchiver(sandboxes, objects, max_files=5).harvest(
        run_id=RunId("r"), workspace_id="r"
    )
    assert len(refs) == 5


async def test_released_workspace_yields_nothing():
    objects = InMemoryObjectStore()
    sandboxes = workspace({"/workspace/gone.txt": b"x"})
    await sandboxes.release("r")
    refs = await WorkspaceArchiver(sandboxes, objects).harvest(run_id=RunId("r"), workspace_id="r")
    assert refs == ()


async def test_no_sandbox_configured_is_not_an_error():
    refs = await WorkspaceArchiver(None, InMemoryObjectStore()).harvest(
        run_id=RunId("r"), workspace_id="r"
    )
    assert refs == ()


async def test_storage_failure_does_not_propagate():
    class BrokenStore(InMemoryObjectStore):
        async def put(self, *a, **k):
            raise RuntimeError("storage down")

    refs = await WorkspaceArchiver(workspace({"/workspace/a.txt": b"x"}), BrokenStore()).harvest(
        run_id=RunId("r"), workspace_id="r"
    )
    assert refs == ()


def test_content_type_lookup():
    assert content_type_for("a/b/report.md") == "text/markdown"
    assert content_type_for("data.csv") == "text/csv"
    assert content_type_for("noext") == "application/octet-stream"


# -- end to end through a real run --------------------------------------------


async def run_writing(objective_tools, sandboxes):
    services = build_services(
        model=ScriptedModelProvider(
            [calls(call("write_file", "c1", path="report.md", content="# Findings")), says("done")]
        ),
        tools=[WriteFileTool(sandboxes)],
        sandboxes=sandboxes,
    )
    run = await InProcessAgentRunner(services).run(
        RunRequest(spec=make_spec(tools=("write_file",)), task=make_task(), context=make_context())
    )
    return run, services


async def test_a_run_result_carries_the_files_it_produced():
    sandboxes = FakeSandboxProvider(FakeSandbox())
    run, services = await run_writing(("write_file",), sandboxes)

    assert run.status is AgentStatus.COMPLETED
    assert run.result is not None
    keys = [r.key for r in run.result.artifacts]
    assert "runs/run-1/workspace/report.md" in keys

    stored = await services.objects.get(run.result.artifacts[0])
    assert stored == b"# Findings"


async def test_harvest_happens_before_the_workspace_is_released():
    """Order matters: released first, and there is nothing left to copy."""
    sandboxes = FakeSandboxProvider(FakeSandbox())
    run, _ = await run_writing(("write_file",), sandboxes)

    assert sandboxes.released == ["run-1"]
    assert run.result.artifacts, "artifacts were lost to an early release"
