"""Capability-level permissions.

The prototype enforces a simple grant-set check before a tool runs. The
important property for the future is that ``narrowed`` can only *shrink* a set,
so a sub-agent can never hold authority its parent lacked.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class Permission:
    """Well-known permission names.

    Plain strings rather than an enum, so a deployment can introduce its own
    permissions without changing the core.
    """

    TERMINAL = "terminal"
    CODE_EXEC = "code_exec"
    FILESYSTEM = "filesystem"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    NETWORK = "network"
    SPAWN_AGENT = "spawn_agent"

    @classmethod
    def all(cls) -> frozenset[str]:
        """Every built-in permission. Convenient for development and tests."""
        return frozenset(
            value
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, str)
        )


class PermissionSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    grants: frozenset[str] = frozenset()

    def allows(self, required: frozenset[str]) -> bool:
        return required <= self.grants

    def missing(self, required: frozenset[str]) -> frozenset[str]:
        return required - self.grants

    def narrowed(self, keep: Iterable[str]) -> PermissionSet:
        """Intersect with ``keep``. Can only remove grants, never add them."""
        return PermissionSet(grants=self.grants & frozenset(keep))

    @classmethod
    def of(cls, *names: str) -> PermissionSet:
        return cls(grants=frozenset(names))


__all__ = ["Permission", "PermissionSet"]
