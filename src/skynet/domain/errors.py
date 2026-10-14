"""Error taxonomy and the serializable error record carried in the event log.

``ErrorInfo`` is the domain's representation of a failure. Exceptions do not
cross the ``EffectExecutor`` boundary as exceptions -- they are converted into
``ErrorInfo`` so that a failure is *data* that can be folded into state, written
to a Temporal history, replayed, and asserted on in tests.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class SkynetError(Exception):
    """Base class for every error raised by the harness."""

    retryable: bool = False
    code: str = "skynet_error"


class ConfigurationError(SkynetError):
    code = "configuration"


class InvalidTransition(SkynetError):
    """A ``RunEvent`` was applied to a run in a state that forbids it."""

    code = "invalid_transition"


class LoopStalled(SkynetError):
    """The agent loop made no progress; a decision produced no events."""

    code = "loop_stalled"


class ToolNotFound(SkynetError):
    code = "tool_not_found"


class AgentNotFound(SkynetError):
    code = "agent_not_found"


class SubAgentFailed(SkynetError):
    code = "sub_agent_failed"


class ProviderError(SkynetError):
    """A model provider failed. Transport-level failures are retryable."""

    code = "provider_error"

    def __init__(self, message: str, *, retryable: bool = False, code: str | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if code:
            self.code = code


class SandboxError(SkynetError):
    code = "sandbox_error"


class StorageError(SkynetError):
    code = "storage_error"


class ErrorCategory(StrEnum):
    PROVIDER = "provider"
    TOOL = "tool"
    LIMIT = "limit"
    TRANSITION = "transition"
    INTERNAL = "internal"


class ErrorInfo(BaseModel):
    """Serializable failure record. The only failure representation in state."""

    model_config = ConfigDict(frozen=True)

    category: ErrorCategory = ErrorCategory.INTERNAL
    code: str = "internal"
    message: str = ""
    retryable: bool = False
    details: dict[str, Any] = {}

    @classmethod
    def from_exc(cls, exc: BaseException) -> ErrorInfo:
        if isinstance(exc, SkynetError):
            return cls(
                category=ErrorCategory.PROVIDER
                if isinstance(exc, ProviderError)
                else ErrorCategory.INTERNAL,
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
            )
        return cls(
            category=ErrorCategory.INTERNAL,
            code=type(exc).__name__,
            message=str(exc),
            retryable=False,
        )

    @classmethod
    def limit(cls, name: str, value: object) -> ErrorInfo:
        return cls(
            category=ErrorCategory.LIMIT,
            code=f"limit_exceeded:{name}",
            message=f"resource limit '{name}' exceeded (limit={value})",
            details={"limit": name, "value": value},
        )

    @classmethod
    def of(cls, code: str, message: str, *, retryable: bool = False) -> ErrorInfo:
        return cls(code=code, message=message, retryable=retryable)


__all__ = [
    "AgentNotFound",
    "ConfigurationError",
    "ErrorCategory",
    "ErrorInfo",
    "InvalidTransition",
    "LoopStalled",
    "ProviderError",
    "SandboxError",
    "SkynetError",
    "StorageError",
    "SubAgentFailed",
    "ToolNotFound",
]
