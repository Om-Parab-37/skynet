"""What an agent is asked to do, and what it produces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from skynet.domain.ids import TaskId
from skynet.domain.models import Usage
from skynet.domain.storage import ObjectRef


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: TaskId
    objective: str
    input: dict[str, Any] = {}
    attachments: tuple[ObjectRef, ...] = ()
    metadata: dict[str, str] = {}


class AgentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: str
    data: dict[str, Any] | None = None
    artifacts: tuple[ObjectRef, ...] = ()
    usage: Usage = Usage()
    iterations: int = 0


__all__ = ["AgentResult", "Task"]
