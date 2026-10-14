"""Configuration.

Everything the runtime needs to choose its adapters, as environment-driven
settings. Nested with ``__``: ``SKYNET_MODEL__PROVIDER=openrouter``.

Nothing here assumes Docker Compose, or any particular deployment shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["ollama", "openrouter", "gemini", "openai_compat"]
TracingBackend = Literal["noop", "console", "langfuse"]
StorageBackend = Literal["memory", "filesystem", "minio"]
SandboxBackend = Literal["local", "docker"]
MemoryBackend = Literal["inmemory", "postgres"]
EmbeddingBackend = Literal["hashing", "null", "ollama", "gemini", "openai_compat"]
ExecutionEngine = Literal["inprocess", "temporal"]


class ModelSettings(BaseModel):
    provider: ModelBackend = "ollama"
    name: str = "qwen3.5:9b"
    base_url: str = "http://localhost:11434/v1"
    """Endpoint for self-hosted providers (Ollama, any OpenAI-compatible server)."""
    base_url_override: str = ""
    """Optional endpoint override for hosted providers, which default correctly."""
    api_key: str = ""
    temperature: float = 0.0
    max_output_tokens: int | None = None
    timeout_s: float = 300.0
    force_tool_emulation: bool = False
    """Force prompt-based tool calling even if the provider claims native support."""


class PromptSettings(BaseModel):
    directory: Path = Path("prompts")
    use_langfuse: bool = False


class TracingSettings(BaseModel):
    backend: TracingBackend = "noop"


class LangfuseSettings(BaseModel):
    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.public_key and self.secret_key)


class StorageSettings(BaseModel):
    backend: StorageBackend = "memory"
    bucket: str = "skynet"
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = False
    directory: Path = Path(".skynet/artifacts")


class EmbeddingSettings(BaseModel):
    backend: EmbeddingBackend = "hashing"
    """Defaults to the offline embedder so `docker compose up` always yields a
    working store. Switching to a real one is one variable and no code."""
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    dimensions: int = 768
    """768 keeps a local model and Gemini (via its Matryoshka `dimensions`
    parameter) interchangeable, and stays under pgvector's 2000-dim index cap."""
    timeout_s: float = 60.0


class MemorySettings(BaseModel):
    backend: MemoryBackend = "inmemory"
    dsn: str = ""
    pool_min: int = 1
    pool_max: int = 8
    statement_timeout_ms: int = 10_000
    """`RecallMemory` is dispatched outside the tool timeout, so this is not
    defence in depth -- it is the only defence against a hung query."""
    migrate: bool = True


class SandboxSettings(BaseModel):
    backend: SandboxBackend = "local"
    image: str = "python:3.13-slim"
    root: Path = Path("/tmp/skynet-workspaces")
    memory_mb: int = 512
    cpus: float = 1.0
    network: bool = False


class TemporalSettings(BaseModel):
    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "skynet-agents"
    workflow_timeout_s: int = 3600


class ExecutionSettings(BaseModel):
    engine: ExecutionEngine = "inprocess"
    event_log_dir: Path | None = None
    """Shared directory for the run event log.

    Required when the API and the worker are separate processes: the worker
    produces events and the API serves them. Left unset in-process, where a
    memory sink suffices."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SKYNET_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)

    agents_file: Path = Path("agents.yaml")
    log_level: str = "INFO"


__all__ = [
    "EmbeddingSettings",
    "ExecutionSettings",
    "LangfuseSettings",
    "MemorySettings",
    "ModelSettings",
    "PromptSettings",
    "SandboxSettings",
    "Settings",
    "StorageSettings",
    "TemporalSettings",
    "TracingSettings",
]
