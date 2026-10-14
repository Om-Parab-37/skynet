"""Catalogs available inside the workflow sandbox.

A workflow may only touch deterministic data, so it gets tool *specs* and agent
*specs* -- never the executable tools, which live in the activity worker. This
is exactly why ``ToolCatalog`` was separated from ``ToolRegistry``.

The catalogs are rebuilt from the same YAML the worker loads, so the two sides
cannot drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from skynet.application.registry import InMemoryToolRegistry
from skynet.config.agents import StaticAgentCatalog, load_catalog
from skynet.config.settings import Settings
from skynet.domain.permissions import PermissionSet
from skynet.domain.tools import ToolSpec
from skynet.ports.tools import ToolCatalog


class SpecOnlyCatalog:
    """A ``ToolCatalog`` over specs alone -- safe for workflow code."""

    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def specs_for(self, allowed: Sequence[str], permissions: PermissionSet) -> tuple[ToolSpec, ...]:
        return tuple(
            spec
            for name in allowed
            if (spec := self._specs.get(name)) is not None
            and permissions.allows(spec.required_permissions)
        )


def build_spec_catalog(registry: InMemoryToolRegistry) -> SpecOnlyCatalog:
    return SpecOnlyCatalog(
        [spec for name in registry.names() if (spec := registry.spec(name)) is not None]
    )


@lru_cache(maxsize=1)
def workflow_tool_catalog() -> ToolCatalog:
    from skynet.config.container import SystemClock, UuidGenerator, build_registry
    from skynet.testing.fakes import FakeSandboxProvider, InMemoryMemoryStore

    # Specs are independent of the collaborators, so placeholders are fine here:
    # nothing in the workflow can invoke these tools.
    registry = build_registry(
        sandboxes=FakeSandboxProvider(),
        memory=InMemoryMemoryStore(),
        ids=UuidGenerator(),
        clock=SystemClock(),
    )
    return build_spec_catalog(registry)


@lru_cache(maxsize=1)
def workflow_agent_catalog() -> StaticAgentCatalog:
    settings = Settings()
    return load_catalog(settings.agents_file, default_model=settings.model.name)


__all__ = [
    "SpecOnlyCatalog",
    "build_spec_catalog",
    "workflow_agent_catalog",
    "workflow_tool_catalog",
]
