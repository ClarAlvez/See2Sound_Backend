import logging
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from API.core.errors import StoredDataError
from API.services.runtime import RuntimeLease, recover_interrupted_jobs
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


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings.create_runtime_directories()
    with RuntimeLease():
        await run_in_threadpool(recover_interrupted_jobs)
        yield


app = FastAPI(
    lifespan=lifespan,
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

@app.exception_handler(StoredDataError)
async def stored_data_error_handler(request, error):
    logging.getLogger(__name__).error("Stored generation data unavailable", exc_info=error)
    return JSONResponse(status_code=500, content={"detail": str(error)})
