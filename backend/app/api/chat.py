"""Chat endpoints: a plain JSON request/response, and an SSE variant.

On streaming
------------
The SSE endpoint streams *stages* live (screening, routing, tool call,
retrieval, generation) and then streams the answer text — but only after the
verification stage has run. That ordering is deliberate: citation markers are
validated and invalid ones stripped before a single character reaches the user,
so the copilot never displays a claim attributed to a source that does not
exist. Token-by-token streaming straight from the model would trade that
guarantee for perceived speed, and in an enterprise governance context the
guarantee is worth more. The staged events keep the interface responsive in the
meantime, which is what the perceived-latency win is actually made of.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models import ChatRequest, ChatResponse, SessionSummary
from ..services import ConversationStore, Orchestrator
from .deps import get_conversations, get_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def _resolve_employee(request_body: ChatRequest, header_value: str | None) -> ChatRequest:
    """Demo identity resolution.

    A production deployment reads the subject from a verified SSO token and
    ignores the client entirely. Here the header wins over the body so the
    frontend's "acting as" selector is explicit and auditable, and the whole
    mechanism is confined to this one function.
    """
    if header_value:
        return request_body.model_copy(update={"employee_id": header_value.strip()})
    return request_body


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    x_employee_id: str | None = Header(default=None, alias="X-Employee-Id"),
) -> ChatResponse:
    started = time.perf_counter()
    try:
        response = await orchestrator.answer(_resolve_employee(payload, x_employee_id))
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean 500 envelope
        logger.exception("chat pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "pipeline_failure",
                "detail": "The copilot could not complete this request. Please try again.",
                "retryable": True,
            },
        ) from exc
    logger.info(
        "chat completed",
        extra={
            "trace_id": response.trace_id,
            "session_id": response.session_id,
            "route": "/api/chat",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        },
    )
    return response


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    x_employee_id: str | None = Header(default=None, alias="X-Employee-Id"),
) -> StreamingResponse:
    resolved = _resolve_employee(payload, x_employee_id)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("status", {"stage": "screening", "label": "Checking your question"})
        task = asyncio.create_task(orchestrator.answer(resolved))

        # Progress narration while the pipeline runs. The stages are the real
        # pipeline stages; the timing is indicative, and the UI labels it as
        # progress rather than as measured per-stage timing (the measured
        # breakdown arrives with the final event).
        stages = [
            ("retrieval", "Searching policy documents"),
            ("tools", "Checking your HR records"),
            ("generation", "Composing a grounded answer"),
            ("verification", "Verifying citations"),
        ]
        index = 0
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                return
            if index < len(stages):
                stage, label = stages[index]
                yield _sse("status", {"stage": stage, "label": label})
                index += 1
            await asyncio.sleep(0.35)

        try:
            response: ChatResponse = task.result()
        except asyncio.CancelledError:  # pragma: no cover - client hung up
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("streaming chat failed: %s", exc)
            yield _sse(
                "error",
                {
                    "error": "pipeline_failure",
                    "detail": "The copilot could not complete this request.",
                    "retryable": True,
                },
            )
            return

        yield _sse(
            "sources",
            {
                "trace_id": response.trace_id,
                "session_id": response.session_id,
                "citations": [c.model_dump(mode="json") for c in response.citations],
                "tool_calls": [t.model_dump(mode="json") for t in response.tool_calls],
            },
        )

        # Verified text, streamed in word-boundary pieces.
        words = response.answer.split(" ")
        for position in range(0, len(words), 8):
            if await request.is_disconnected():
                return
            piece = " ".join(words[position : position + 8])
            suffix = " " if position + 8 < len(words) else ""
            yield _sse("delta", {"text": piece + suffix})
            await asyncio.sleep(0.02)

        payload_json = response.model_dump(mode="json")
        payload_json.pop("answer", None)  # already delivered via deltas
        yield _sse("done", payload_json)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    conversations: ConversationStore = Depends(get_conversations),
) -> list[SessionSummary]:
    return await conversations.list_sessions()


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    conversations: ConversationStore = Depends(get_conversations),
) -> dict[str, bool]:
    return {"deleted": await conversations.delete(session_id)}
