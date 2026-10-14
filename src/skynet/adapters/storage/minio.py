"""Object storage backed by MinIO.

The MinIO SDK is synchronous, so every call is dispatched to a thread; the
``ObjectStore`` port stays async and the rest of the runtime never blocks.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Mapping
from typing import Any

from skynet.domain.errors import ConfigurationError, StorageError
from skynet.domain.storage import ObjectRef


class MinioObjectStore:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def create(
        cls,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> MinioObjectStore:
        try:
            from minio import Minio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigurationError("minio is not installed; install the 'storage' extra") from exc
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        return cls(client, bucket)

    async def ensure_bucket(self) -> None:
        def _ensure() -> None:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)

        await asyncio.to_thread(_ensure)

    async def put(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectRef:
        def _put() -> None:
            self._client.put_object(
                self._bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
                metadata=dict(metadata or {}),
            )

        try:
            await asyncio.to_thread(_put)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to store {key}: {exc}") from exc

        return ObjectRef(
            bucket=self._bucket,
            key=key,
            size=len(data),
            content_type=content_type,
            metadata=dict(metadata or {}),
        )

    async def get(self, ref: ObjectRef) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(ref.bucket or self._bucket, ref.key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        try:
            return await asyncio.to_thread(_get)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to read {ref.key}: {exc}") from exc

    async def delete(self, ref: ObjectRef) -> bool:
        def _delete() -> bool:
            self._client.remove_object(ref.bucket or self._bucket, ref.key)
            return True

        try:
            return await asyncio.to_thread(_delete)
        except Exception:  # noqa: BLE001
            return False

    async def list(self, prefix: str) -> tuple[ObjectRef, ...]:
        def _list() -> tuple[ObjectRef, ...]:
            objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
            refs = []
            for obj in objects:
                # list_objects does not return content type; only a stat does.
                # Listings are per-run and bounded, so the extra calls are cheap,
                # and the alternative is reporting every artifact as a blob.
                content_type = getattr(obj, "content_type", None)
                if not content_type:
                    try:
                        content_type = self._client.stat_object(
                            self._bucket, obj.object_name
                        ).content_type
                    except Exception:  # noqa: BLE001
                        content_type = None
                refs.append(
                    ObjectRef(
                        bucket=self._bucket,
                        key=obj.object_name,
                        size=obj.size or 0,
                        content_type=content_type or "application/octet-stream",
                    )
                )
            return tuple(refs)

        try:
            return await asyncio.to_thread(_list)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"failed to list {prefix}: {exc}") from exc

    async def presigned_url(self, ref: ObjectRef, expires_s: int = 3600) -> str:
        from datetime import timedelta

        def _sign() -> str:
            return self._client.presigned_get_object(
                ref.bucket or self._bucket, ref.key, expires=timedelta(seconds=expires_s)
            )

        return await asyncio.to_thread(_sign)


__all__ = ["MinioObjectStore"]
