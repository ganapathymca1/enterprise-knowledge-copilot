"""LLM provider interface, retry policy and error taxonomy.

Every provider implements the same three-method surface, so the orchestrator
never knows which model answered. That matters for two reasons the brief calls
out: it lets the app degrade gracefully when a provider is slow or down, and it
lets a reviewer run the whole system with no API key at all.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for provider failures."""

    retryable = False


class LLMTimeout(LLMError):
    retryable = True


class LLMRateLimited(LLMError):
    retryable = True


class LLMUnavailable(LLMError):
    """Provider is not configured or not reachable."""

    retryable = False


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: int = 0
    finish_reason: str = "stop"
    truncated: bool = False
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    """Structural type implemented by every provider."""

    name: str
    model: str

    async def health(self) -> bool: ...

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> LLMResult: ...

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> AsyncIterator[str]: ...


async def with_retries(
    operation,
    *,
    attempts: int,
    base_delay: float = 0.6,
    label: str = "llm",
):
    """Retry transient LLM failures with exponential backoff and jitter.

    Only errors flagged ``retryable`` are retried: retrying a 401 or a bad
    request wastes the user's time and hides the real problem.
    """
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await operation()
        except LLMError as exc:
            last_error = exc
            if not exc.retryable or attempt >= attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) * (0.7 + 0.6 * random.random())
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.2fs",
                label, attempt, attempts, exc.__class__.__name__, delay,
            )
            await asyncio.sleep(delay)
        except asyncio.TimeoutError as exc:  # pragma: no cover - defensive
            last_error = LLMTimeout(str(exc))
            if attempt >= attempts:
                raise last_error from exc
            await asyncio.sleep(base_delay * attempt)
    raise last_error or LLMError("unknown LLM failure")
