"""FastAPI dependencies.

Singletons live on ``app.state`` and are built once during lifespan startup:
the index and the record store are read-only and expensive enough that
rebuilding them per request would dominate latency.
"""

from __future__ import annotations

from fastapi import Depends, Request

from ..config import Settings, get_settings
from ..rag import KnowledgeIndex
from ..services import ConversationStore, FeedbackStore, Orchestrator


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def get_index(request: Request) -> KnowledgeIndex:
    return request.app.state.index


def get_conversations(request: Request) -> ConversationStore:
    return request.app.state.conversations


def get_feedback_store(request: Request) -> FeedbackStore:
    return request.app.state.feedback


def get_app_settings(settings: Settings = Depends(get_settings)) -> Settings:
    return settings
