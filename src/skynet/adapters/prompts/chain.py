"""Try prompt sources in order, falling back on failure.

This is what makes a hosted prompt service safe to depend on: if Langfuse is
down, unreachable, or simply does not have the prompt yet, the local YAML copy
serves it and the agent runs anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from skynet.domain.errors import ConfigurationError
from skynet.domain.prompts import PromptRef, PromptTemplate
from skynet.ports.prompts import PromptProvider

logger = logging.getLogger(__name__)


class ChainPromptProvider:
    def __init__(self, providers: Sequence[PromptProvider]) -> None:
        if not providers:
            raise ConfigurationError("ChainPromptProvider needs at least one provider")
        self._providers = tuple(providers)

    async def fetch(self, ref: PromptRef) -> PromptTemplate:
        errors: list[str] = []
        for provider in self._providers:
            try:
                return await provider.fetch(ref)
            except Exception as exc:  # noqa: BLE001 - fall through to the next source
                errors.append(f"{type(provider).__name__}: {exc}")
                logger.debug("prompt source failed for %s: %s", ref, exc)
        raise ConfigurationError(
            f"could not resolve prompt {ref} from any source: {'; '.join(errors)}"
        )


__all__ = ["ChainPromptProvider"]
