"""Prompts from version-controlled YAML files.

The fallback that guarantees the system always boots. Keeping prompts in git
also means they are reviewable and diffable, which a hosted prompt UI alone
cannot give you.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from skynet.domain.errors import ConfigurationError
from skynet.domain.prompts import PromptRef, PromptTemplate


class YamlPromptProvider:
    """Loads ``<root>/<name>.yaml`` where ``name`` may contain directories.

    File format::

        template: |
          You are {{agent_name}}.
        version: 3
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    async def fetch(self, ref: PromptRef) -> PromptTemplate:
        path = self._resolve(ref)
        if path is None:
            raise ConfigurationError(f"no prompt file for '{ref.name}' under {self._root}")
        loaded = yaml.safe_load(path.read_text()) or {}
        if isinstance(loaded, str):
            loaded = {"template": loaded}
        template = loaded.get("template")
        if not template:
            raise ConfigurationError(f"prompt file {path} has no 'template' key")
        return PromptTemplate(
            ref=ref,
            template=str(template),
            version=loaded.get("version"),
            source=f"yaml:{path.name}",
        )

    def _resolve(self, ref: PromptRef) -> Path | None:
        for suffix in (".yaml", ".yml"):
            candidate = self._root / f"{ref.name}{suffix}"
            if candidate.is_file():
                return candidate
        return None


__all__ = ["YamlPromptProvider"]
