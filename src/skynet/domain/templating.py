"""Pure ``{{variable}}`` substitution.

Rendering lives in the domain, not in the prompt adapter. That is what lets the
Langfuse adapter be a pure *fetch* and keeps prompt behaviour identical no
matter where the template came from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}")


def render_text(template: str, variables: Mapping[str, str]) -> str:
    """Replace ``{{name}}`` with ``variables[name]``.

    Unknown placeholders are left untouched rather than raising, so a prompt
    edited in Langfuse cannot take a running agent down.
    """

    def _replace(match: re.Match[str]) -> str:
        return str(variables.get(match.group(1), match.group(0)))

    return _PLACEHOLDER.sub(_replace, template)


def placeholders(template: str) -> frozenset[str]:
    return frozenset(m.group(1) for m in _PLACEHOLDER.finditer(template))


__all__ = ["placeholders", "render_text"]
