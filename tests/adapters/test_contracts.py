"""Run the port contracts against every implementation we ship.

The Docker and MinIO suites are marked ``integration`` and excluded by default,
so the core suite stays offline and fast -- but they are the same assertions.
"""

from __future__ import annotations

import contextlib
import os

import pytest

from skynet.adapters.sandbox.local import LocalSandboxProvider
from skynet.adapters.storage.memory import FileSystemObjectStore
from skynet.testing.contracts import (
    MemoryStoreContract,
    ObjectStoreContract,
    RestartableSandboxContract,
    SandboxContract,
)
from skynet.testing.fakes import InMemoryMemoryStore, InMemoryObjectStore


class TestInMemoryObjectStore(ObjectStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryObjectStore()


class TestFileSystemObjectStore(ObjectStoreContract):
    @pytest.fixture
    def store(self, tmp_path):
        return FileSystemObjectStore(tmp_path)


class TestInMemoryMemoryStore(MemoryStoreContract):
    @pytest.fixture
    def memory(self):
        return InMemoryMemoryStore()


MEMORY_DSN = os.environ.get(
    "SKYNET_TEST_MEMORY_DSN", ""
)


@pytest.mark.integration
class TestPostgresMemoryStore(MemoryStoreContract):
    """The same assertions as the in-memory store, against Postgres + pgvector.

    This is the point of the contract suite: "the memory store is replaceable"
    stops being a claim in a README and becomes a test that fails when it is not
    true. It runs with the offline embedder so the vector arm is exercised for
    real without needing a model server.
    """

    @pytest.fixture
    async def memory(self):
        from skynet.adapters.embedding.hashing import HashingEmbedder
        from skynet.adapters.memory.postgres import PostgresMemoryStore

        store = PostgresMemoryStore(MEMORY_DSN, embedder=HashingEmbedder())
        pool = await store.ready()
        # Contract records use fixed ids, so a leftover row from a previous run
        # would make the next one pass or fail for the wrong reason.
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM memories WHERE namespace = ANY(%s)",
                (["agent:ns", "agent:other"],),
            )
        yield store
        await store.aclose()


class TestLocalSandbox(SandboxContract):
    @pytest.fixture
    def provider(self, tmp_path):
        return LocalSandboxProvider(root=tmp_path)


# Workspace ids the contract suite uses. Docker containers outlive the test
# process, so the suite has to clean up after itself or it leaks exactly the way
# the runtime used to.
CONTRACT_WORKSPACES = (
    "contract",
    "persistent",
    "disposable",
    "restart-ws",
    "listing",
    "ws-a",
    "ws-b",
)


@pytest.mark.integration
class TestDockerSandbox(RestartableSandboxContract):
    @pytest.fixture
    def provider(self):
        from skynet.adapters.sandbox.docker import DockerSandboxProvider

        provider = DockerSandboxProvider()
        yield provider
        _remove_contract_containers()

    def fresh_provider(self):
        from skynet.adapters.sandbox.docker import DockerSandboxProvider

        return DockerSandboxProvider()


def _remove_contract_containers() -> None:
    import docker

    client = docker.from_env()
    for workspace in CONTRACT_WORKSPACES:
        for container in client.containers.list(
            all=True, filters={"label": f"skynet.workspace={workspace}"}
        ):
            with contextlib.suppress(Exception):
                container.remove(force=True)


class TestLocalSandboxRestart(RestartableSandboxContract):
    @pytest.fixture
    def provider(self, tmp_path):
        self._root = tmp_path
        return LocalSandboxProvider(root=tmp_path)

    def fresh_provider(self):
        return LocalSandboxProvider(root=self._root)
