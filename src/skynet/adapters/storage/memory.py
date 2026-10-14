"""Filesystem-backed object storage.

The non-MinIO implementation. Useful for local runs and for demonstrating that
the ``ObjectStore`` port carries no S3 assumptions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from skynet.domain.errors import StorageError
from skynet.domain.storage import ObjectRef


class FileSystemObjectStore:
    def __init__(self, root: str | Path, bucket: str = "skynet") -> None:
        self._root = Path(root)
        self._bucket = bucket
        self._root.mkdir(parents=True, exist_ok=True)

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectRef:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        ref = ObjectRef(
            bucket=self._bucket,
            key=key,
            size=len(data),
            content_type=content_type,
            metadata=dict(metadata or {}),
        )
        path.with_suffix(path.suffix + ".meta.json").write_text(ref.model_dump_json())
        return ref

    async def get(self, ref: ObjectRef) -> bytes:
        path = self._path(ref.key)
        if not path.is_file():
            raise StorageError(f"no such object: {ref.key}")
        return path.read_bytes()

    async def delete(self, ref: ObjectRef) -> bool:
        path = self._path(ref.key)
        if not path.is_file():
            return False
        path.unlink()
        path.with_suffix(path.suffix + ".meta.json").unlink(missing_ok=True)
        return True

    async def list(self, prefix: str) -> tuple[ObjectRef, ...]:
        refs = []
        for meta in sorted(self._root.rglob("*.meta.json")):
            ref = ObjectRef.model_validate(json.loads(meta.read_text()))
            if ref.key.startswith(prefix):
                refs.append(ref)
        return tuple(refs)

    def _path(self, key: str) -> Path:
        resolved = (self._root / key).resolve()
        if not resolved.is_relative_to(self._root.resolve()):
            raise StorageError(f"key escapes the storage root: {key}")
        return resolved


__all__ = ["FileSystemObjectStore"]
