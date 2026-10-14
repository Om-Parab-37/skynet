"""Process-local memory store.

Re-exported from the shipped fakes: this genuinely is the prototype's memory
implementation, and pretending otherwise by duplicating the code would be worse.
A persistent or vector-backed store implements the same ``MemoryStore`` port.
"""

from __future__ import annotations

from skynet.testing.fakes import InMemoryMemoryStore

__all__ = ["InMemoryMemoryStore"]
