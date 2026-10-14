"""Publish the local YAML prompts to Langfuse.

Langfuse becomes the editable source of truth; the YAML files stay in git as the
reviewable copy and the fallback the runtime uses when Langfuse is unavailable.
Run this once after bringing up the langfuse profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from skynet.config.settings import Settings


def main() -> int:
    settings = Settings()
    if not settings.langfuse.configured:
        print("Langfuse keys are not configured; nothing to do.", file=sys.stderr)
        return 1

    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.host,
    )

    root = Path(settings.prompts.directory)
    published = 0
    for path in sorted(root.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text()) or {}
        template = document.get("template")
        if not template:
            continue
        name = str(path.relative_to(root).with_suffix("")).replace("\\", "/")
        client.create_prompt(
            name=name,
            prompt=template,
            labels=["production"],
            type="text",
        )
        print(f"published {name}")
        published += 1

    client.flush()
    print(f"\n{published} prompt(s) published to {settings.langfuse.host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
