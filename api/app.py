"""FastAPI application factory for Trismegistus."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import backtest, health, predictions, status
from api.schemas import RootResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="Trismegistus API",
        description="HTTP platform for the football prediction pipeline",
        version="0.3.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(status.router)
    app.include_router(predictions.router)
    app.include_router(backtest.router)

    @app.get("/", response_model=RootResponse, tags=["meta"])
    def root() -> RootResponse:
        return RootResponse(
            name="Trismegistus",
            version="0.3.0",
            docs="/docs",
            endpoints={
                "health": "/health",
                "status": "/status",
                "fixtures": "/fixtures/upcoming",
                "predictions": "/predictions",
                "backtest": "/backtest",
            },
        )

    return app


app = create_app()