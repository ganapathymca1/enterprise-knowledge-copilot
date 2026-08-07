"""Health, readiness and corpus introspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import __version__
from ..models import HealthResponse
from ..rag import KnowledgeIndex
from .deps import get_index

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, index: KnowledgeIndex = Depends(get_index)) -> HealthResponse:
    """Liveness plus a real readiness signal.

    The LLM provider is probed here rather than only at startup, so a
    provider that dies after boot shows as ``degraded`` instead of only
    surfacing as failed answers.
    """
    orchestrator = request.app.state.orchestrator
    llm = orchestrator.llm
    try:
        provider_ready = await llm.health()
    except Exception:  # noqa: BLE001
        provider_ready = False

    notes: list[str] = []
    if llm.name == "extractive":
        notes.append(
            "No language model is configured. Answers are extracted verbatim from the "
            "policy text. Set COPILOT_LLM_PROVIDER and a key or run Ollama locally."
        )
    if not provider_ready and llm.name != "extractive":
        notes.append(
            f"Provider '{llm.name}' did not respond to a health probe. Requests will "
            "fall back to the offline extractive answerer."
        )

    return HealthResponse(
        status="ok" if provider_ready else "degraded",
        version=__version__,
        provider=llm.name,
        model=llm.model,
        provider_ready=provider_ready,
        documents=len(index.documents),
        chunks=len(index.chunks),
        index_built_at=index.built_at,
        notes=notes,
    )


@router.get("/corpus")
async def corpus(index: KnowledgeIndex = Depends(get_index)) -> dict:
    """What the copilot can actually answer from — shown in the UI sidebar.

    Publishing the corpus is a governance feature, not a debugging one: an
    employee who can see the boundary of the knowledge base is much less likely
    to over-trust an answer that falls outside it.
    """
    return {
        "stats": index.stats,
        "documents": [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "category": document.category,
                "owner": document.owner,
                "version": document.version,
                "effective_date": document.effective_date,
                "review_date": document.review_date,
                "source_path": document.source_path,
                "chunks": sum(1 for chunk in index.chunks if chunk.doc_id == document.doc_id),
            }
            for document in index.documents
        ],
    }


@router.get("/documents/{doc_id}")
async def document_detail(doc_id: str, index: KnowledgeIndex = Depends(get_index)) -> dict:
    """Full text of one policy, so a citation can be opened and read in context."""
    for document in index.documents:
        if document.doc_id.upper() == doc_id.upper():
            return {
                "doc_id": document.doc_id,
                "title": document.title,
                "category": document.category,
                "owner": document.owner,
                "version": document.version,
                "effective_date": document.effective_date,
                "audience": document.audience,
                "body": document.body,
            }
    raise HTTPException(status_code=404, detail=f"No policy document with id '{doc_id}'")
