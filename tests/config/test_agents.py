"""Agent definitions are configuration: the YAML shape is part of the contract."""

from __future__ import annotations

from skynet.config.agents import memory_from_dict, spec_from_dict
from skynet.domain.memory import MemoryKind, ScopeLevel


def test_memory_defaults_to_private_and_off() -> None:
    memory = memory_from_dict(None)
    assert memory.auto_recall is False
    assert memory.share_project is False
    assert memory.write_level is ScopeLevel.AGENT


def test_memory_block_is_parsed() -> None:
    memory = memory_from_dict(
        {
            "auto_recall": True,
            "recall_limit": 3,
            "recall_kinds": ["semantic", "procedural"],
            "share_project": True,
            "project": "skynet",
            "write_level": "project",
            "consolidate": True,
            "reflect_every": 5,
            "default_ttl_hours": 24.0,
        }
    )
    assert memory.recall_limit == 3
    assert memory.recall_kinds == (MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL)
    assert memory.write_level is ScopeLevel.PROJECT
    assert memory.reflect_every == 5


def test_a_bare_model_name_is_accepted_for_memory_work() -> None:
    """A bare model name should not require learning the shape of ModelParams."""
    assert memory_from_dict({"model": "qwen3.5:2b"}).model.model == "qwen3.5:2b"


def test_a_spec_without_a_memory_block_still_loads() -> None:
    spec = spec_from_dict("demo", {"tools": ["python_exec"]}, default_model="m")
    assert spec.memory.auto_recall is False
