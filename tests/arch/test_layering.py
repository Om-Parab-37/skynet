"""Executable architecture rules.

The claims this project makes -- that the core is provider-agnostic and that the
agent loop is replay-safe -- are only true while nobody imports the wrong thing.
These tests turn those claims into failures instead of intentions.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "skynet"

# Modules that make the domain non-deterministic, and therefore unsafe to replay.
FORBIDDEN_IN_DOMAIN = {
    "asyncio",
    "random",
    "secrets",
    "time",
    "uuid",
    "httpx",
    "requests",
    "socket",
    "subprocess",
    "threading",
    "os",
    "pathlib",
}

INFRASTRUCTURE_PACKAGES = {
    "temporalio",
    "fastapi",
    "langfuse",
    "minio",
    "docker",
    "uvicorn",
    "boto3",
    "typer",
}


def modules_in(package: str) -> list[pathlib.Path]:
    return sorted((SRC / package).rglob("*.py"))


def imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def skynet_imports_of(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("skynet"):
            parts = node.module.split(".")
            if len(parts) > 1:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "skynet" and len(parts) > 1:
                    found.add(parts[1])
    return found


@pytest.mark.parametrize("path", modules_in("domain"), ids=lambda p: p.name)
def test_domain_is_deterministic(path: pathlib.Path) -> None:
    """No clock, no randomness, no I/O -- this is what makes replay sound."""
    offenders = imports_of(path) & FORBIDDEN_IN_DOMAIN
    assert not offenders, f"{path.name} imports non-deterministic modules: {sorted(offenders)}"


@pytest.mark.parametrize("path", modules_in("domain"), ids=lambda p: p.name)
def test_domain_depends_on_nothing_of_ours(path: pathlib.Path) -> None:
    allowed = {"domain"}
    assert skynet_imports_of(path) <= allowed, f"{path.name} reaches outside the domain"


@pytest.mark.parametrize("path", modules_in("domain"), ids=lambda p: p.name)
def test_domain_has_no_infrastructure(path: pathlib.Path) -> None:
    offenders = imports_of(path) & INFRASTRUCTURE_PACKAGES
    assert not offenders, f"{path.name} imports infrastructure: {sorted(offenders)}"


@pytest.mark.parametrize("path", modules_in("ports"), ids=lambda p: p.name)
def test_ports_depend_only_on_the_domain(path: pathlib.Path) -> None:
    assert skynet_imports_of(path) <= {"domain", "ports"}


@pytest.mark.parametrize("path", modules_in("ports"), ids=lambda p: p.name)
def test_ports_have_no_infrastructure(path: pathlib.Path) -> None:
    offenders = imports_of(path) & INFRASTRUCTURE_PACKAGES
    assert not offenders, f"{path.name} imports infrastructure: {sorted(offenders)}"


@pytest.mark.parametrize("path", modules_in("application"), ids=lambda p: p.name)
def test_application_never_imports_adapters(path: pathlib.Path) -> None:
    """The core must not know which technologies exist."""
    assert "adapters" not in skynet_imports_of(path), f"{path.name} imports an adapter"


@pytest.mark.parametrize("path", modules_in("application"), ids=lambda p: p.name)
def test_application_has_no_infrastructure(path: pathlib.Path) -> None:
    offenders = imports_of(path) & INFRASTRUCTURE_PACKAGES
    assert not offenders, f"{path.name} imports infrastructure: {sorted(offenders)}"


@pytest.mark.parametrize("path", modules_in("capabilities"), ids=lambda p: p.name)
def test_capabilities_never_import_adapters(path: pathlib.Path) -> None:
    assert "adapters" not in skynet_imports_of(path), f"{path.name} imports an adapter"


def code_without_docs(path: pathlib.Path) -> str:
    """Source with docstrings removed and comments dropped by unparsing.

    Prose may discuss Temporal or Ollama; *code* may not reference them.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


@pytest.mark.parametrize(
    "module",
    [
        "policy.py",
        "loop.py",
        "context.py",
        "effects.py",
        "runner.py",
        "history_compaction.py",
        "consolidator.py",
        "memory_consolidation.py",
        "memory_ranking.py",
        "memory_scoping.py",
    ],
)
def test_core_code_never_names_a_provider(module: str) -> None:
    """No provider-specific special-casing anywhere in the agent runtime."""
    banned = ("ollama", "openrouter", "openai", "langfuse", "minio", "temporal")
    code = code_without_docs(SRC / "application" / module)
    for name in banned:
        assert name not in code, f"application/{module} references '{name}' in code"


def test_the_agent_loop_has_exactly_one_await() -> None:
    """The single seam. More than one means side effects have leaked in."""
    tree = ast.parse((SRC / "application" / "loop.py").read_text())
    awaits = [node for node in ast.walk(tree) if isinstance(node, ast.Await)]
    assert len(awaits) == 1, f"expected one await in the loop, found {len(awaits)}"
