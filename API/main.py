from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from API.core.config import settings
from API.routes.generations import (
    router as generations_router,
)
from API.routes.health import (
    router as health_router,
)


app = FastAPI(
    title="See2Sound API",
    description=(
        "API do backend do See2Sound para "
        "processamento audiovisual e geração "
        "automática de audiodescrição."
    ),
    version="0.1.0",
)


allow_all_origins = (
    settings.cors_origins == ["*"]
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=(
        not allow_all_origins
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    health_router
)

app.include_router(
    generations_router
)


@app.get("/")
def root():
    return {
        "service": "See2Sound API",
        "version": "0.1.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "generations": (
            "/api/v1/generations"
        ),
    }