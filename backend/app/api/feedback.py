"""Feedback capture and the governance read-outs built on top of it."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..models import FeedbackRequest, FeedbackResponse
from ..services import FeedbackStore
from ..services.guardrails import redact
from .deps import get_feedback_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    try:
        feedback_id, routed_to = store.record_feedback(
            trace_id=payload.trace_id,
            session_id=payload.session_id,
            rating=payload.rating,
            reason=payload.reason,
            # Free text is redacted before storage: employees paste anything
            # into a comment box, including details about their own health.
            comment=redact(payload.comment),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to store feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Could not record feedback") from exc
    logger.info(
        "feedback recorded", extra={"trace_id": payload.trace_id, "route": "/api/feedback"}
    )
    return FeedbackResponse(ok=True, feedback_id=feedback_id, routed_to=routed_to)


@router.get("/feedback/stats")
async def feedback_stats(store: FeedbackStore = Depends(get_feedback_store)) -> dict:
    """Aggregate quality metrics. In production this backs a governance dashboard."""
    return store.stats()


@router.get("/feedback/failing-passages")
async def failing_passages(
    limit: int = 10,
    store: FeedbackStore = Depends(get_feedback_store),
) -> list[dict]:
    """Passages most often present in a downvoted answer — the rewrite queue."""
    return store.chunk_failure_report(limit=limit)


@router.get("/feedback/regression-candidates")
async def regression_candidates(
    limit: int = 50,
    store: FeedbackStore = Depends(get_feedback_store),
) -> list[dict]:
    """Downvoted questions, shaped for pasting into eval/golden_set.json."""
    return store.export_downvoted_questions(limit=limit)
