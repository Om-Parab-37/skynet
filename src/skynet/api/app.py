"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from skynet.api.deps import build_runtime
from skynet.api.routes import router
from skynet.config.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname).1s %(name)s: %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = await build_runtime(settings)
        try:
            yield
        finally:
            await app.state.runtime.close()

    app = FastAPI(
        title="Skynet Agent Harness",
        version="0.1.0",
        summary="A general-purpose agent runtime",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
