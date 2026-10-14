"""Capability ports.

``ToolCatalog`` is split from ``ToolRegistry`` deliberately (interface
segregation): the pure policy needs only to *read specs* synchronously, while
only the effect executor needs to *invoke* tools. Under Temporal that split is
load-bearing -- the workflow holds a catalog, the activity worker holds the
registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from skynet.domain.messages import ToolCall
from skynet.domain.permissions import PermissionSet
from skynet.domain.tools import ToolContext, ToolResult, ToolSpec


@runtime_checkable
class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult: ...


@runtime_checkable
class ToolCatalog(Protocol):
    """Pure and synchronous: safe to call inside ``AgentPolicy.decide``."""

    def spec(self, name: str) -> ToolSpec | None: ...

    def names(self) -> tuple[str, ...]: ...

    def specs_for(
        self, allowed: Sequence[str], permissions: PermissionSet
    ) -> tuple[ToolSpec, ...]: ...


@runtime_checkable
class ToolRegistry(ToolCatalog, Protocol):
    def register(self, tool: Tool) -> None: ...

    def get(self, name: str) -> Tool: ...


__all__ = ["Tool", "ToolCatalog", "ToolRegistry"]
