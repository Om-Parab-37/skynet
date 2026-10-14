"""Model provider port.

One contract for every model, local or hosted. Providers advertise what they can
do through ``ProviderCapabilities`` so the runtime can adapt (for example by
emulating tool calling) instead of branching on a provider name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from skynet.domain.models import ModelRequest, ModelResponse


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    native_tools: bool = False
    parallel_tool_calls: bool = False
    streaming: bool = False
    json_schema_output: bool = False
    emulated_tools: bool = False
    """True when tool calling is being simulated through prompting."""


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


__all__ = ["ModelProvider", "ProviderCapabilities"]
