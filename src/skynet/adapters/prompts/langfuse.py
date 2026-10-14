"""Prompt management backed by Langfuse.

Only fetching happens here. Rendering stays in the domain, so a prompt behaves
identically whether it came from Langfuse or from a local file.

The Langfuse SDK is a soft dependency: importing this module without it raises a
clear configuration error rather than breaking the package.
"""

from __future__ import annotations

import asyncio
from typing import Any

from skynet.domain.errors import ConfigurationError
from skynet.domain.prompts import PromptRef, PromptTemplate


class LangfusePromptProvider:
    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def create(cls, *, public_key: str, secret_key: str, host: str) -> LangfusePromptProvider:
        try:
            from langfuse import Langfuse
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigurationError(
                "langfuse is not installed; install the 'langfuse' extra"
            ) from exc
        return cls(Langfuse(public_key=public_key, secret_key=secret_key, host=host))

    async def fetch(self, ref: PromptRef) -> PromptTemplate:
        # The SDK is synchronous; keep it off the event loop.
        prompt = await asyncio.to_thread(self._get, ref)
        template = getattr(prompt, "prompt", None)
        if not isinstance(template, str):
            raise ConfigurationError(f"Langfuse prompt '{ref.name}' is not a text prompt")
        return PromptTemplate(
            ref=ref,
            template=template,
            version=getattr(prompt, "version", None),
            source="langfuse",
        )

    def _get(self, ref: PromptRef) -> Any:
        kwargs: dict[str, Any] = {}
        if ref.version is not None:
            kwargs["version"] = ref.version
        else:
            kwargs["label"] = ref.label
        return self._client.get_prompt(ref.name, **kwargs)


__all__ = ["LangfusePromptProvider"]
