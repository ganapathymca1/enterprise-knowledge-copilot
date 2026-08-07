"""Application configuration.

Every setting is overridable from the environment with the ``COPILOT_`` prefix
or from a local ``.env`` file. Defaults are chosen so that `uvicorn` starts and
serves useful answers with **no configuration at all** — the LLM layer falls
back to a deterministic extractive answerer when no provider is reachable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

LLMProvider = Literal["auto", "ollama", "gemini", "openai_compatible", "extractive"]


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_prefix="COPILOT_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- service ---------------------------------------------------------
    app_name: str = "Enterprise Knowledge Copilot"
    environment: str = "local"
    log_level: str = "INFO"

    # --- LLM -------------------------------------------------------------
    llm_provider: LLMProvider = "auto"
    temperature: float = 0.1
    max_output_tokens: int = 800
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""

    # --- retrieval -------------------------------------------------------
    top_k: int = 5
    candidate_k: int = 20
    min_retrieval_score: float = 0.12
    chunk_tokens: int = 220
    chunk_overlap_tokens: int = 45
    mmr_lambda: float = 0.7

    # --- conversation ----------------------------------------------------
    history_turns: int = 6
    session_ttl_minutes: int = 120
    max_question_chars: int = 1200

    # --- governance ------------------------------------------------------
    redact_pii_in_logs: bool = True
    enable_tools: bool = True
    require_grounding: bool = True

    # --- paths -----------------------------------------------------------
    knowledge_base_dir: Path = Field(default=Path("data/knowledge_base"))
    hr_records_dir: Path = Field(default=Path("data/hr_records"))
    state_dir: Path = Field(default=Path("var"))

    # --- demo identity ---------------------------------------------------
    # A real deployment takes the caller identity from SSO / a signed JWT.
    # For the POC the frontend sends a header and this is the fallback.
    default_employee_id: str = "EMP-1002"

    @field_validator("knowledge_base_dir", "hr_records_dir", "state_dir")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value)

    def resolved_openai_api_key(self) -> str:
        """Fall back to the conventional ``OPENAI_API_KEY`` variable."""
        return self.openai_api_key or os.getenv("OPENAI_API_KEY", "")

    def resolved_gemini_api_key(self) -> str:
        return (
            self.gemini_api_key
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (also the FastAPI dependency)."""
    settings = Settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    return settings
