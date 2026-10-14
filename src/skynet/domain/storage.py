"""Reference to a stored binary artifact.

The core never holds artifact *bytes* in state -- only references. This keeps
``AgentRun`` small enough to live in a Temporal workflow history and keeps the
object-storage backend replaceable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ObjectRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    bucket: str
    key: str
    size: int = 0
    content_type: str = "application/octet-stream"
    metadata: dict[str, str] = {}

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


__all__ = ["ObjectRef"]
