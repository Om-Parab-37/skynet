"""Capability (tool) contracts.

A tool is described to the model by a ``ToolSpec`` and answers with a
``ToolResult``. Both are pure data, which is what lets the agent loop stay
ignorant of every specific capability.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from skynet.domain.errors import ErrorInfo
from skynet.domain.ids import CallId, RunId
from skynet.domain.permissions import PermissionSet
from skynet.domain.storage import ObjectRef


class ToolKind(StrEnum):
    STANDARD = "standard"
    """An ordinary side effect. Becomes a Temporal activity."""
    DELEGATION = "delegation"
    """Spawns a sub-agent. Becomes a Temporal child workflow."""
    CONTROL = "control"
    """Never executed; consumed by the policy (e.g. ``final_answer``)."""


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


class ToolSpec(BaseModel):
    """What the model is told about a capability, plus how the runtime governs it."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema object, generated from the tool signature."""

    kind: ToolKind = ToolKind.STANDARD
    required_permissions: frozenset[str] = frozenset()
    requires_approval: bool = False
    """Human-in-the-loop seam: the policy parks the run before invoking."""
    timeout_s: float = 60.0
    idempotent: bool = False
    """Safe to retry. Drives the Temporal retry policy for this tool."""


class ToolResult(BaseModel):
    """An observation fed back to the model.

    Failures are results, not exceptions: a denied, timed-out or malformed call
    still produces a ``ToolResult`` so the model can correct itself.
    """

    model_config = ConfigDict(frozen=True)

    call_id: CallId
    name: str
    status: ToolStatus
    content: str
    """What the model sees. Already truncated to fit the context budget."""
    data: dict[str, Any] | None = None
    """Structured payload for programmatic consumers."""
    artifacts: tuple[ObjectRef, ...] = ()
    error: ErrorInfo | None = None
    truncated: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class ToolContext(BaseModel):
    """Run-scoped *data* handed to a tool at invocation time.

    Deliberately carries no services. Tools receive their collaborators through
    the constructor, which keeps this from degrading into a service locator.
    """

    model_config = ConfigDict(frozen=True)

    run_id: RunId
    agent_name: str
    permissions: PermissionSet
    depth: int = 0
    workspace_id: str | None = None
    deadline: datetime | None = None
    variables: dict[str, str] = {}


__all__ = [
    "ToolContext",
    "ToolKind",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
]
