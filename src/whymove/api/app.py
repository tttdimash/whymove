"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from whymove import __version__
from whymove.api.routes import router
from whymove.container import AppConfig, Container


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or AppConfig.from_env()
    container = Container(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: eagerly initialize the pipeline to fail fast
        try:
            container.pipeline()
        except Exception as e:
            # Log but don't crash — health endpoint will report degraded
            import warnings
            warnings.warn(f"Pipeline initialization failed: {e}", stacklevel=1)
        app.state.container = container
        yield
        # Shutdown: release engine resources
        container.close()

    app = FastAPI(
        title="WhyMove",
        description=(
            "Chess move intent explanation API. "
            "Given a position and move, returns likely strategic/tactical intents "
            "with confidence scores and a human-readable explanation."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/v1")

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "error_type": "validation"},
        )

    return app
