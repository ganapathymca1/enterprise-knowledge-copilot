"""Shared fixtures.

Every test runs against the offline extractive provider, so the suite needs no
API key, no network and no model download, and is deterministic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("COPILOT_LLM_PROVIDER", "extractive")
os.environ.setdefault("COPILOT_STATE_DIR", str(REPO_ROOT / "var" / "test"))

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag import KnowledgeIndex  # noqa: E402
from backend.app.tools import ToolRegistry, load_records  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def index(settings) -> KnowledgeIndex:
    return KnowledgeIndex.from_directory(
        settings.knowledge_base_dir,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )


@pytest.fixture(scope="session")
def tools(settings) -> ToolRegistry:
    return ToolRegistry(load_records(settings.hr_records_dir))


@pytest.fixture(scope="session")
def client():
    """TestClient with the app's real lifespan (index build, stores, provider)."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client
