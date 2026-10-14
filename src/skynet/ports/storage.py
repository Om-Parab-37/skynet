"""Object/artifact storage port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from skynet.domain.storage import ObjectRef


@runtime_checkable
class ObjectStore(Protocol):
    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectRef: ...

    async def get(self, ref: ObjectRef) -> bytes: ...

    async def delete(self, ref: ObjectRef) -> bool: ...

    async def list(self, prefix: str) -> tuple[ObjectRef, ...]: ...


__all__ = ["ObjectStore"]
