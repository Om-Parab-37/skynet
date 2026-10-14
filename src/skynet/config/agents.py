"""Agent definitions loaded from YAML.

Agents are configuration, not code. Adding one is a YAML entry; it needs no
change to the runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skynet.domain.agent import AgentSpec, MemoryConfig, ResourceLimits
from skynet.domain.errors import AgentNotFound
from skynet.domain.models import ModelParams
from skynet.domain.permissions import PermissionSet
from skynet.domain.prompts import PromptRef


class StaticAgentCatalog:
    def __init__(self, specs: list[AgentSpec] | None = None) -> None:
        self._specs = {spec.name: spec for spec in (specs or [])}

    def register(self, spec: AgentSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> AgentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise AgentNotFound(
                f"unknown agent '{name}'; available: {', '.join(sorted(self._specs))}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def all(self) -> tuple[AgentSpec, ...]:
        return tuple(self._specs.values())


def memory_from_dict(raw: dict[str, Any] | None) -> MemoryConfig:
    """Parse the optional ``memory:`` block.

    ``model`` accepts a bare model name as well as a mapping, because "use the
    cheap model for memory work" is the common case and should not require
    learning the shape of ``ModelParams`` to express.
    """
    settings = dict(raw or {})
    if isinstance(settings.get("model"), str):
        settings["model"] = ModelParams(model=settings["model"])
    return MemoryConfig(**settings)


def spec_from_dict(name: str, raw: dict[str, Any], *, default_model: str) -> AgentSpec:
    limits = ResourceLimits(**(raw.get("limits") or {}))
    return AgentSpec(
        name=name,
        description=raw.get("description", ""),
        prompt=PromptRef(
            name=raw.get("prompt", "skynet/agent"),
            label=raw.get("prompt_label", "production"),
        ),
        instructions=raw.get("instructions", ""),
        model=ModelParams(
            model=raw.get("model") or default_model,
            temperature=float(raw.get("temperature", 0.0)),
        ),
        tools=tuple(raw.get("tools") or ()),
        sub_agents=tuple(raw.get("sub_agents") or ()),
        limits=limits,
        memory=memory_from_dict(raw.get("memory")),
        default_permissions=PermissionSet.of(*(raw.get("permissions") or ())),
    )


def load_catalog(path: str | Path, *, default_model: str) -> StaticAgentCatalog:
    source = Path(path)
    if not source.is_file():
        return StaticAgentCatalog()
    document = yaml.safe_load(source.read_text()) or {}
    agents = document.get("agents") or {}
    return StaticAgentCatalog(
        [
            spec_from_dict(name, raw or {}, default_model=default_model)
            for name, raw in agents.items()
        ]
    )


__all__ = ["StaticAgentCatalog", "load_catalog", "memory_from_dict", "spec_from_dict"]
