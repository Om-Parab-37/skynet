"""Prompt identity, templates and rendered output.

Prompts are referenced by name+label so that the source (Langfuse, a YAML file,
a database) is an implementation detail behind ``PromptProvider``.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from skynet.domain.templating import render_text


class PromptRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    label: str = "production"
    version: int | None = None

    def __str__(self) -> str:
        return f"{self.name}@{self.version if self.version is not None else self.label}"


class PromptTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref: PromptRef
    template: str
    version: int | None = None
    source: str = "unknown"
    """Which provider served it. Recorded for traceability, never branched on."""


class RenderedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref: PromptRef
    text: str
    version: int | None = None
    source: str = "unknown"


def render(template: PromptTemplate, variables: Mapping[str, str]) -> RenderedPrompt:
    """Pure. The only way a ``PromptTemplate`` becomes a ``RenderedPrompt``."""
    return RenderedPrompt(
        ref=template.ref,
        text=render_text(template.template, variables),
        version=template.version,
        source=template.source,
    )


__all__ = ["PromptRef", "PromptTemplate", "RenderedPrompt", "render"]
