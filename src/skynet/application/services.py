"""The bundle of ports an agent run needs.

A single explicit container passed by the composition root. It is deliberately a
plain data holder: no lookup logic, no lazy construction, no global state --
just the collaborators, so every dependency of a run is visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from skynet.application.context import ContextBuilder, DefaultContextBuilder
from skynet.ports.agents import AgentCatalog
from skynet.ports.approvals import ApprovalGate
from skynet.ports.clock import Clock, IdGenerator
from skynet.ports.events import EventSink
from skynet.ports.memory import MemoryStore
from skynet.ports.model import ModelProvider
from skynet.ports.prompts import PromptProvider
from skynet.ports.runs import RunStore
from skynet.ports.sandbox import SandboxProvider
from skynet.ports.storage import ObjectStore
from skynet.ports.tools import ToolRegistry
from skynet.ports.tracing import Tracer


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    model: ModelProvider
    prompts: PromptProvider
    registry: ToolRegistry
    memory: MemoryStore
    objects: ObjectStore
    tracer: Tracer
    sink: EventSink
    clock: Clock
    ids: IdGenerator
    approvals: ApprovalGate
    runs: RunStore
    agents: AgentCatalog
    sandboxes: SandboxProvider | None = None
    """Held so the runtime can release a run's workspace when it terminates."""
    context_builder: ContextBuilder = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.context_builder is None:
            object.__setattr__(self, "context_builder", DefaultContextBuilder())


__all__ = ["RuntimeServices"]
