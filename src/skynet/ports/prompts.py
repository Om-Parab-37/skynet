"""Prompt management port.

Fetch only. Rendering is pure and lives in ``skynet.domain.prompts.render``, so
prompt semantics cannot drift between backends.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from skynet.domain.prompts import PromptRef, PromptTemplate


@runtime_checkable
class PromptProvider(Protocol):
    async def fetch(self, ref: PromptRef) -> PromptTemplate: ...


__all__ = ["PromptProvider"]
