"""Composition root.

The one place that knows both the abstractions and the concrete adapters. Every
mandated technology appears here as a table entry, so replacing one is a single
line and touches no core code -- that is the whole architectural claim, made
concrete and readable in about a hundred lines.

Deliberately plain functions rather than a DI framework: at this size a
container library would add indirection without removing any.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from skynet.adapters.embedding.registry import create_embedder
from skynet.adapters.memory.inmemory import InMemoryMemoryStore
from skynet.adapters.memory.postgres import PostgresMemoryStore
from skynet.adapters.model.registry import create_provider
from skynet.adapters.prompts.chain import ChainPromptProvider
from skynet.adapters.prompts.yaml_files import YamlPromptProvider
from skynet.adapters.sandbox.local import LocalSandboxProvider
from skynet.adapters.storage.memory import FileSystemObjectStore
from skynet.adapters.tracing.noop import ConsoleTracer, NoopTracer
from skynet.application.registry import InMemoryToolRegistry
from skynet.application.services import RuntimeServices
from skynet.application.tool_calling import wrap_for_tool_calling
from skynet.capabilities.control import FinalAnswerTool
from skynet.capabilities.delegate import DelegateTool
from skynet.capabilities.files import ReadFileTool, WriteFileTool
from skynet.capabilities.memory import MemorySearchTool, MemoryWriteTool
from skynet.capabilities.python_exec import CodeExecTool
from skynet.capabilities.terminal import TerminalTool
from skynet.config.agents import load_catalog
from skynet.config.settings import Settings
from skynet.domain.agent import FinishMode
from skynet.ports.embedding import Embedder
from skynet.ports.memory import MemoryStore
from skynet.ports.model import ModelProvider
from skynet.ports.prompts import PromptProvider
from skynet.ports.sandbox import ResourceProfile, SandboxProvider
from skynet.ports.storage import ObjectStore
from skynet.ports.tools import ToolRegistry
from skynet.ports.tracing import Tracer
from skynet.testing.fakes import AutoApproveGate, InMemoryObjectStore, InMemoryRunStore

logger = logging.getLogger(__name__)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class UuidGenerator:
    def new_id(self, prefix: str = "") -> str:
        value = uuid4().hex[:12]
        return f"{prefix}-{value}" if prefix else value

    def new_trace_id(self) -> str:
        return uuid4().hex


# -- adapter tables -----------------------------------------------------------
#
# Each mapping is an extension point. A new backend is one entry.


def _build_model(settings: Settings) -> ModelProvider:
    kwargs: dict[str, Any] = {"timeout_s": settings.model.timeout_s}
    if settings.model.provider == "ollama":
        kwargs["base_url"] = settings.model.base_url
    elif settings.model.provider in {"openrouter", "gemini"}:
        kwargs["api_key"] = settings.model.api_key
        if settings.model.base_url_override:
            kwargs["base_url"] = settings.model.base_url_override
    else:
        kwargs["base_url"] = settings.model.base_url
        kwargs["api_key"] = settings.model.api_key or None
    return create_provider(settings.model.provider, **kwargs)


def _build_tracer(settings: Settings) -> Tracer:
    backend = settings.tracing.backend
    if backend == "langfuse":
        if not settings.langfuse.configured:
            logger.warning("langfuse tracing requested without keys; falling back to console")
            return ConsoleTracer()
        from skynet.adapters.tracing.langfuse import LangfuseTracer

        return LangfuseTracer.create(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.host,
        )
    return {"noop": NoopTracer, "console": ConsoleTracer}[backend]()


def _build_prompts(settings: Settings) -> PromptProvider:
    """Langfuse first when configured, always with the local YAML copy behind it."""
    sources: list[PromptProvider] = []
    if settings.prompts.use_langfuse and settings.langfuse.configured:
        from skynet.adapters.prompts.langfuse import LangfusePromptProvider

        sources.append(
            LangfusePromptProvider.create(
                public_key=settings.langfuse.public_key,
                secret_key=settings.langfuse.secret_key,
                host=settings.langfuse.host,
            )
        )
    sources.append(YamlPromptProvider(settings.prompts.directory))
    return ChainPromptProvider(sources)


def _build_objects(settings: Settings) -> ObjectStore:
    backend = settings.storage.backend
    if backend == "minio":
        from skynet.adapters.storage.minio import MinioObjectStore

        return MinioObjectStore.create(
            endpoint=settings.storage.endpoint,
            access_key=settings.storage.access_key,
            secret_key=settings.storage.secret_key,
            bucket=settings.storage.bucket,
            secure=settings.storage.secure,
        )
    if backend == "filesystem":
        return FileSystemObjectStore(settings.storage.directory, settings.storage.bucket)
    return InMemoryObjectStore(settings.storage.bucket)


def _build_sandboxes(settings: Settings) -> SandboxProvider:
    if settings.sandbox.backend == "docker":
        from skynet.adapters.sandbox.docker import DockerSandboxProvider

        return DockerSandboxProvider(
            image=settings.sandbox.image,
            resources=ResourceProfile(
                memory_mb=settings.sandbox.memory_mb,
                cpus=settings.sandbox.cpus,
                network=settings.sandbox.network,
            ),
        )
    return LocalSandboxProvider(root=settings.sandbox.root)


def build_registry(
    *,
    sandboxes: SandboxProvider,
    memory: MemoryStore,
    ids: Any,
    clock: Any,
) -> ToolRegistry:
    """Every built-in capability, registered once.

    The agent loop never sees this list; agents select from it by name.
    """
    return InMemoryToolRegistry(
        [
            TerminalTool(sandboxes),
            CodeExecTool(sandboxes),
            ReadFileTool(sandboxes),
            WriteFileTool(sandboxes),
            MemoryWriteTool(memory, ids, clock),
            MemorySearchTool(memory, clock),
            DelegateTool(),
            FinalAnswerTool(),
        ]
    )


def _build_embedder(settings: Settings) -> Embedder:
    embedding = settings.embedding
    kwargs: dict[str, Any] = {"dimensions": embedding.dimensions}
    if embedding.backend not in ("hashing", "null"):
        kwargs |= {
            "model": embedding.model,
            "base_url": embedding.base_url,
            "timeout_s": embedding.timeout_s,
        }
        if embedding.api_key:
            kwargs["api_key"] = embedding.api_key
    if embedding.backend == "hashing":
        kwargs["model"] = embedding.model or "hashing"
    return create_embedder(embedding.backend, **kwargs)


def _build_memory(settings: Settings) -> MemoryStore:
    """The memory backend, chosen the same way every other port is.

    It was hardcoded until now, which made it the one port in the README's
    replaceability table with a single implementation -- in a section whose
    whole thesis is that an abstraction with one implementation is unproven.
    """
    if settings.memory.backend == "postgres":
        return PostgresMemoryStore(
            settings.memory.dsn,
            embedder=_build_embedder(settings),
            dimensions=settings.embedding.dimensions,
            statement_timeout_ms=settings.memory.statement_timeout_ms,
            min_size=settings.memory.pool_min,
            max_size=settings.memory.pool_max,
            migrate=settings.memory.migrate,
        )
    return InMemoryMemoryStore()


def build_services(settings: Settings) -> RuntimeServices:
    clock = SystemClock()
    ids = UuidGenerator()

    memory = _build_memory(settings)
    sandboxes = _build_sandboxes(settings)
    provider = wrap_for_tool_calling(
        _build_model(settings), ids, force_emulation=settings.model.force_tool_emulation
    )

    return RuntimeServices(
        model=provider,
        prompts=_build_prompts(settings),
        registry=build_registry(sandboxes=sandboxes, memory=memory, ids=ids, clock=clock),
        memory=memory,
        objects=_build_objects(settings),
        tracer=_build_tracer(settings),
        sink=_build_sink(settings),
        clock=clock,
        ids=ids,
        approvals=AutoApproveGate(),
        runs=InMemoryRunStore(),
        agents=load_catalog(settings.agents_file, default_model=settings.model.name),
        sandboxes=sandboxes,
    )


def build_event_log(settings: Settings):
    """The shared event log, when the runtime spans processes."""
    if settings.execution.event_log_dir is None:
        return None
    from skynet.adapters.runs.file_log import FileEventLog

    return FileEventLog(settings.execution.event_log_dir)


def _build_sink(settings: Settings):
    from skynet.adapters.runs.sink import CompositeEventSink, LoggingEventSink

    sinks = [LoggingEventSink()]
    if (shared := build_event_log(settings)) is not None:
        sinks.append(shared)
    return CompositeEventSink(sinks)


def finish_mode_for(services: RuntimeServices) -> FinishMode:
    return (
        FinishMode.FINAL_ANSWER_TOOL
        if services.model.capabilities.emulated_tools
        else FinishMode.NO_TOOL_CALLS
    )


__all__ = [
    "SystemClock",
    "UuidGenerator",
    "build_registry",
    "build_services",
    "finish_mode_for",
]
