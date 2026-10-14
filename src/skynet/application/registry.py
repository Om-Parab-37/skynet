"""In-memory tool registry.

This is the production implementation, not a test double -- capability lookup
does not need persistence. Registration is explicit, and the agent loop never
names a tool, so adding a capability is a single ``register`` call.
"""

from __future__ import annotations

from collections.abc import Sequence

from skynet.domain.errors import ToolNotFound
from skynet.domain.permissions import PermissionSet
from skynet.domain.tools import ToolSpec
from skynet.ports.tools import Tool


class InMemoryToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFound(
                f"unknown tool '{name}'; registered: {', '.join(sorted(self._tools))}"
            ) from exc

    def spec(self, name: str) -> ToolSpec | None:
        tool = self._tools.get(name)
        return tool.spec if tool else None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def specs_for(self, allowed: Sequence[str], permissions: PermissionSet) -> tuple[ToolSpec, ...]:
        """Specs the agent may both *use* (allowlist) and *is allowed* to use.

        Filtering here rather than at invocation time means the model is never
        shown a capability it cannot exercise.
        """
        specs = []
        for name in allowed:
            tool = self._tools.get(name)
            if tool is None:
                continue
            if not permissions.allows(tool.spec.required_permissions):
                continue
            specs.append(tool.spec)
        return tuple(specs)


__all__ = ["InMemoryToolRegistry"]
