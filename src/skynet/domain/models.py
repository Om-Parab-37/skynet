"""Model-invocation contracts.

``ModelRequest``/``ModelResponse`` are provider-neutral. Anything a specific
provider needs beyond this shape goes through ``ModelParams.extra``, which
adapters may read and the core never inspects.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from skynet.domain.messages import Message
from skynet.domain.tools import ToolSpec


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    PARSE_ERROR = "parse_error"
    """Emulated tool calling produced output we could not parse."""


class ToolChoice(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ModelParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    temperature: float = 0.0
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    extra: dict[str, Any] = {}
    """Adapter-specific escape hatch. Never interpreted by the core."""


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...]
    params: ModelParams
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = ToolChoice.AUTO
    response_schema: dict[str, Any] | None = None
    max_parallel_tool_calls: int | None = None
    """Emulated tool calling sets 1; native providers usually leave this unset."""


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: Message
    finish_reason: FinishReason
    usage: Usage = Usage()
    model: str = ""
    provider_response_id: str | None = None
    parse_error: str | None = None
    """Set iff ``finish_reason is FinishReason.PARSE_ERROR``."""


__all__ = [
    "FinishReason",
    "ModelParams",
    "ModelRequest",
    "ModelResponse",
    "ToolChoice",
    "Usage",
]
