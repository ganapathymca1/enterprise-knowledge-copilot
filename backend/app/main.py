"""FastAPI application entry point.

Run from the repository root:

    uvicorn backend.app.main:app --reload --port 8000

The frontend is served by this same process from ``/`` so the demo is a single
command and has no cross-origin configuration to get wrong. CORS is still
enabled for localhost dev servers, because a React or Vite frontend pointed at
this API is the expected next step.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import chat_router, feedback_router, system_router
from .config import REPO_ROOT, get_settings
from .logging_utils import configure_logging
from .models import ErrorResponse
from .rag import KnowledgeIndex
from .services import ConversationStore, FeedbackStore
from .services.orchestrator import build_orchestrator
from .tools import ToolRegistry, load_records

logger = logging.getLogger(__name__)
FRONTEND_DIR = REPO_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the expensive singletons once, and tear them down cleanly."""
    settings = get_settings()
    configure_logging(settings.log_level, redact_pii=settings.redact_pii_in_logs)

    started = time.perf_counter()
    index = KnowledgeIndex.from_directory(
        settings.knowledge_base_dir,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    tools = ToolRegistry(load_records(settings.hr_records_dir))
    conversations = ConversationStore(
        ttl_minutes=settings.session_ttl_minutes,
    )
    feedback = FeedbackStore(settings.state_dir / "copilot.sqlite3")
    orchestrator = await build_orchestrator(settings, index, tools, conversations, feedback)

    app.state.settings = settings
    app.state.index = index
    app.state.tools = tools
    app.state.conversations = conversations
    app.state.feedback = feedback
    app.state.orchestrator = orchestrator

    logger.info(
        "copilot ready: %d documents, %d chunks, provider=%s model=%s (%d ms)",
        len(index.documents),
        len(index.chunks),
        orchestrator.llm.name,
        orchestrator.llm.model,
        int((time.perf_counter() - started) * 1000),
    )
    try:
        yield
    finally:
        feedback.close()
        logger.info("copilot shut down")


app = FastAPI(
    title="Enterprise Knowledge Copilot",
    version=__version__,
    description=(
        "Grounded internal assistant for HR policies and employee records. "
        "Every answer is retrieved, cited and verified before it is shown."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Localhost only: this service is internal and never public-facing.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and log one structured line per request."""
    request_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-Id"] = request_id
    if request.url.path.startswith("/api"):
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "trace_id": request_id,
                "route": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return the same error envelope the frontend renders everywhere else."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="invalid_request",
            detail=exc.errors()[0].get("msg", "The request could not be validated."),
            retryable=False,
        ).model_dump(),
    )


app.include_router(chat_router)
app.include_router(feedback_router)
app.include_router(system_router)


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
