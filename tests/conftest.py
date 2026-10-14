"""Test fixtures.

The builders live in ``skynet.testing.builders`` so adapter suites can reuse
them; this module only re-exports them for convenience.
"""

from __future__ import annotations

import pytest

from skynet.domain.agent import AgentSpec
from skynet.testing.builders import (  # noqa: F401
    build_services,
    call,
    calls,
    make_context,
    make_run,
    make_spec,
    make_task,
    says,
)


@pytest.fixture
def spec() -> AgentSpec:
    return make_spec()
