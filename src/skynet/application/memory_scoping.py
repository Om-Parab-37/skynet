"""Who may read and write which memories. Pure, and the only authority.

Every other layer takes the scopes it is handed. The executor never widens them,
the store never infers them: the scopes named in a decision *are* the grant. One
function to audit is the whole point -- "which agents can see this fact?" has to
be answerable by reading a screen of code, not by tracing a call graph.

Private by default. An agent always reads its own run and agent scopes; project
and user scopes require an explicit grant in ``MemoryConfig``, and
``AgentContext.variables`` overrides the spec's owners so a single agent
definition can serve many projects.
"""

from __future__ import annotations

from skynet.domain.agent import MemoryConfig
from skynet.domain.memory import MemoryScope, ScopeLevel
from skynet.domain.run import AgentRun

PROJECT_VARIABLE = "project"
USER_VARIABLE = "user"


def _owner(run: AgentRun, variable: str, fallback: str) -> str:
    return (run.context.variables.get(variable) or fallback).strip()


def project_owner(run: AgentRun) -> str:
    return _owner(run, PROJECT_VARIABLE, run.spec.memory.project)


def user_owner(run: AgentRun) -> str:
    return _owner(run, USER_VARIABLE, run.spec.memory.user)


def recall_scopes(run: AgentRun) -> tuple[MemoryScope, ...]:
    """The scopes this run may read, most specific first.

    A granted-but-unnamed scope is dropped rather than defaulted: a shared
    namespace with an empty owner would silently pool every agent's memories
    into one bucket, which is the worst possible reading of "share_project".
    """
    memory = run.spec.memory
    scopes = [MemoryScope.run(run.run_id), MemoryScope.agent(run.spec.name)]
    if memory.share_project and (owner := project_owner(run)):
        scopes.append(MemoryScope.project(owner))
    if memory.share_user and (owner := user_owner(run)):
        scopes.append(MemoryScope.user(owner))
    return tuple(scopes)


def write_scope(run: AgentRun) -> MemoryScope:
    """Where this run's new memories land.

    Falls back to the private agent scope when a shared level is configured but
    has no owner -- writing beats losing the memory, and writing it somewhere
    nameless would be worse than either.
    """
    match run.spec.memory.write_level:
        case ScopeLevel.RUN:
            return MemoryScope.run(run.run_id)
        case ScopeLevel.PROJECT if owner := project_owner(run):
            return MemoryScope.project(owner)
        case ScopeLevel.USER if owner := user_owner(run):
            return MemoryScope.user(owner)
        case _:
            return MemoryScope.agent(run.spec.name)


def can_write(memory: MemoryConfig, scope: MemoryScope) -> bool:
    """Whether a configured policy permits writing to an explicitly named scope.

    Used where a caller supplies the scope (a tool argument, a consolidation
    operation) rather than having it derived.
    """
    if scope.level is ScopeLevel.PROJECT:
        return memory.share_project
    if scope.level is ScopeLevel.USER:
        return memory.share_user
    return True


__all__ = [
    "PROJECT_VARIABLE",
    "USER_VARIABLE",
    "can_write",
    "project_owner",
    "recall_scopes",
    "user_owner",
    "write_scope",
]
