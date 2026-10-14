"""Skynet -- a general-purpose agent harness.

Layering (enforced by tests/arch): ``domain`` <- ``ports`` <- ``application``
<- ``capabilities`` <- ``adapters``. The domain and application layers know
nothing about Temporal, Langfuse, MinIO, Ollama or FastAPI.
"""

__version__ = "0.1.0"
