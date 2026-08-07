"""Request/response contracts for the Copilot API.

These models are the published contract between the frontend and the backend.
They are intentionally explicit: every answer carries its citations, the tools
that ran, a confidence band and a trace id, because the UI renders all of them
and the governance story depends on them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class AnswerType(str, Enum):
    """How the answer was produced — drives the badge shown in the UI."""

    grounded = "grounded"          # answered from retrieved policy passages
    tool = "tool"                  # answered from a structured HR record tool
    hybrid = "hybrid"              # tool result explained with policy passages
    abstained = "abstained"        # nothing relevant found; copilot declined
    refused = "refused"            # blocked by a guardrail
    error = "error"                # upstream failure, surfaced honestly


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Citation(BaseModel):
    """A single grounded source shown in the explainability panel."""

    marker: int = Field(description="1-based marker used as [1], [2] in the answer text")
    chunk_id: str
    doc_id: str
    title: str
    section: str = ""
    category: str = ""
    owner: str = ""
    version: str = ""
    effective_date: str = ""
    snippet: str
    score: float = Field(description="Fused retrieval score, 0-1, higher is better")
    source_path: str = ""


class ToolCall(BaseModel):
    """Record of one tool/function invocation, surfaced to the user."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    latency_ms: int = 0
    summary: str = ""
    error: str | None = None


class ChatMessage(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=_utcnow)


class ChatRequest(BaseModel):
    """POST /api/chat request body."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="Omit to start a new conversation; the response returns the id.",
    )
    employee_id: str | None = Field(
        default=None,
        description=(
            "Demo-only impersonation. In production this comes from the SSO "
            "token and is never accepted from the client."
        ),
    )
    top_k: int | None = Field(default=None, ge=1, le=12)
    use_tools: bool | None = None


class TimingBreakdown(BaseModel):
    total_ms: int = 0
    guardrail_ms: int = 0
    rewrite_ms: int = 0
    retrieval_ms: int = 0
    tool_ms: int = 0
    generation_ms: int = 0
    verification_ms: int = 0


class ChatResponse(BaseModel):
    """POST /api/chat response body."""

    trace_id: str
    session_id: str
    message_id: str
    answer: str
    answer_type: AnswerType
    confidence: Confidence
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    notices: list[str] = Field(
        default_factory=list,
        description="User-facing caveats, e.g. degraded provider or stale data.",
    )
    provider: str = ""
    model: str = ""
    rewritten_query: str | None = None
    grounding_score: float = 0.0
    timings: TimingBreakdown = Field(default_factory=TimingBreakdown)
    created_at: datetime = Field(default_factory=_utcnow)


class FeedbackRequest(BaseModel):
    """POST /api/feedback — thumbs up/down plus optional free text."""

    trace_id: str
    session_id: str | None = None
    rating: Literal["up", "down"]
    reason: Literal[
        "incorrect",
        "incomplete",
        "outdated",
        "wrong_source",
        "unclear",
        "not_relevant",
        "helpful",
        "other",
    ] = "other"
    comment: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str
    routed_to: str = Field(
        default="",
        description="Policy owner the feedback is routed to, from document metadata.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    provider: str
    model: str
    provider_ready: bool
    documents: int
    chunks: int
    index_built_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    turns: int
    title: str


class ErrorResponse(BaseModel):
    """Uniform error envelope so the frontend can render one error component."""

    error: str
    detail: str = ""
    trace_id: str = ""
    retryable: bool = False
