"""Scope authority: the one place that decides what an agent may read."""

from __future__ import annotations

from skynet.application.memory_scoping import can_write, recall_scopes, write_scope
from skynet.domain.agent import MemoryConfig
from skynet.domain.memory import MemoryScope, ScopeLevel
from skynet.testing.builders import make_context, make_run, make_spec


def run_with(memory: MemoryConfig, **variables: str):
    return make_run(
        spec=make_spec(memory=memory),
        context=make_context(variables=dict(variables)),
    )


def test_an_agent_reads_its_own_scopes_by_default() -> None:
    """Private by default. Sharing another agent's notes is a grant, not a default."""
    run = run_with(MemoryConfig())
    assert recall_scopes(run) == (MemoryScope.run(run.run_id), MemoryScope.agent("demo"))


def test_recall_scopes_are_ordered_most_specific_first() -> None:
    run = run_with(MemoryConfig(share_project=True, project="skynet", share_user=True, user="ada"))
    assert [scope.level for scope in recall_scopes(run)] == [
        ScopeLevel.RUN,
        ScopeLevel.AGENT,
        ScopeLevel.PROJECT,
        ScopeLevel.USER,
    ]


def test_a_shared_scope_needs_the_grant() -> None:
    run = run_with(MemoryConfig(project="skynet"))
    assert MemoryScope.project("skynet") not in recall_scopes(run)


def test_run_variables_override_the_spec_owner() -> None:
    """One agent definition has to serve many projects."""
    run = run_with(MemoryConfig(share_project=True, project="default"), project="acme")
    assert MemoryScope.project("acme") in recall_scopes(run)


def test_a_granted_scope_with_no_owner_is_dropped() -> None:
    """Better no shared scope than one nameless bucket holding every agent's memories."""
    run = run_with(MemoryConfig(share_project=True))
    assert [scope.level for scope in recall_scopes(run)] == [ScopeLevel.RUN, ScopeLevel.AGENT]


# -- writes -------------------------------------------------------------------


def test_writes_land_in_the_agent_scope_by_default() -> None:
    assert write_scope(run_with(MemoryConfig())) == MemoryScope.agent("demo")


def test_writes_can_be_directed_at_the_run() -> None:
    run = run_with(MemoryConfig(write_level=ScopeLevel.RUN))
    assert write_scope(run) == MemoryScope.run(run.run_id)


def test_writes_can_be_directed_at_a_shared_scope() -> None:
    run = run_with(MemoryConfig(write_level=ScopeLevel.PROJECT, project="skynet"))
    assert write_scope(run) == MemoryScope.project("skynet")


def test_a_shared_write_with_no_owner_falls_back_to_private() -> None:
    """Writing somewhere private beats writing somewhere nameless, and both
    beat losing the memory."""
    run = run_with(MemoryConfig(write_level=ScopeLevel.USER))
    assert write_scope(run) == MemoryScope.agent("demo")


def test_can_write_mirrors_the_grants() -> None:
    private = MemoryConfig()
    shared = MemoryConfig(share_project=True)
    assert can_write(private, MemoryScope.agent("demo")) is True
    assert can_write(private, MemoryScope.project("skynet")) is False
    assert can_write(shared, MemoryScope.project("skynet")) is True
