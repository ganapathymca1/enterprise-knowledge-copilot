"""LLM abstraction: providers, prompts, retry policy and provider selection."""

from __future__ import annotations

import logging

from ..config import Settings
from .base import (
    LLMClient,
    LLMError,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Message,
    with_retries,
)
from .providers import (
    ExtractiveClient,
    GeminiClient,
    OllamaClient,
    OpenAICompatibleClient,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractiveClient",
    "GeminiClient",
    "LLMClient",
    "LLMError",
    "LLMRateLimited",
    "LLMResult",
    "LLMTimeout",
    "LLMUnavailable",
    "Message",
    "OllamaClient",
    "OpenAICompatibleClient",
    "build_client",
    "with_retries",
]


async def build_client(settings: Settings) -> LLMClient:
    """Resolve the configured provider, or auto-detect the first usable one.

    Auto-detection order is deliberate: an explicitly configured hosted key is
    the user's clearest signal of intent, a local Ollama is the best free
    option, and the extractive answerer is the always-available floor. The app
    never fails to start because no LLM is available.
    """
    provider = settings.llm_provider

    def openai_client() -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.resolved_openai_api_key(),
            timeout=settings.llm_timeout_seconds,
        )

    def gemini_client() -> GeminiClient:
        return GeminiClient(
            model=settings.gemini_model,
            api_key=settings.resolved_gemini_api_key(),
            timeout=settings.llm_timeout_seconds,
        )

    def ollama_client() -> OllamaClient:
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "openai_compatible":
        return openai_client()
    if provider == "gemini":
        return gemini_client()
    if provider == "ollama":
        return ollama_client()
    if provider == "extractive":
        return ExtractiveClient()

    # provider == "auto"
    if settings.resolved_openai_api_key():
        logger.info("LLM auto-selection: OpenAI-compatible endpoint (%s)", settings.openai_model)
        return openai_client()
    if settings.resolved_gemini_api_key():
        logger.info("LLM auto-selection: Gemini (%s)", settings.gemini_model)
        return gemini_client()
    candidate = ollama_client()
    if await candidate.health():
        logger.info("LLM auto-selection: local Ollama (%s)", settings.ollama_model)
        return candidate
    logger.warning(
        "No LLM provider configured or reachable — falling back to the offline "
        "extractive answerer. Answers will quote sources verbatim."
    )
    return ExtractiveClient()
