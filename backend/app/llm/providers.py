"""Concrete LLM providers plus the offline extractive fallback.

Supported out of the box:

* ``ollama``            — free and fully local (native /api/chat, streaming)
* ``openai_compatible`` — OpenAI, Groq free tier, LM Studio, OpenRouter, vLLM
* ``gemini``            — Google AI Studio free tier (REST, no extra SDK)
* ``extractive``        — no LLM at all; deterministic, quotes the sources

The extractive provider is not a toy. It is what makes the "clean environment,
no API key, no network" reproduction path in the README real, it keeps the test
suite deterministic and free, and it is the last rung of the degradation ladder
when a configured provider fails mid-request.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import AsyncIterator

import httpx

from .base import (
    LLMError,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------
class OllamaClient:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def _payload(self, messages, temperature, max_tokens, stream):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    async def complete(self, messages, *, temperature=0.1, max_tokens=800) -> LLMResult:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=self._payload(messages, temperature, max_tokens, False),
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"ollama timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"ollama unreachable: {exc}") from exc
        if response.status_code >= 500:
            raise LLMTimeout(f"ollama returned {response.status_code}")
        if response.status_code >= 400:
            raise LLMError(f"ollama error {response.status_code}: {response.text[:200]}")
        data = response.json()
        return LLMResult(
            text=(data.get("message") or {}).get("content", "").strip(),
            provider=self.name,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=data.get("done_reason", "stop"),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    async def stream(self, messages, *, temperature=0.1, max_tokens=800) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=self._payload(messages, temperature, max_tokens, True),
                ) as response:
                    if response.status_code >= 400:
                        raise LLMError(f"ollama error {response.status_code}")
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        piece = (chunk.get("message") or {}).get("content", "")
                        if piece:
                            yield piece
        except httpx.TimeoutException as exc:
            raise LLMTimeout("ollama stream timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"ollama unreachable: {exc}") from exc


# ---------------------------------------------------------------------------
# Any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
class OpenAICompatibleClient:
    name = "openai_compatible"

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def health(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("openai package is not installed") from exc
        if not self.api_key:
            raise LLMUnavailable("no API key configured for the OpenAI-compatible provider")
        # max_retries=0: retry policy lives in one place (llm.base.with_retries)
        # so behaviour is identical across providers and observable in logs.
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        text = str(exc)
        name = exc.__class__.__name__
        if "RateLimit" in name or "429" in text:
            return LLMRateLimited(text[:300])
        if "Timeout" in name or "timed out" in text.lower():
            return LLMTimeout(text[:300])
        if "Connection" in name or "APIConnection" in name:
            return LLMUnavailable(text[:300])
        if "Authentication" in name or "401" in text:
            return LLMUnavailable(f"authentication failed: {text[:200]}")
        error = LLMError(text[:300])
        if "InternalServer" in name or "503" in text or "502" in text:
            error.retryable = True
        return error

    async def complete(self, messages, *, temperature=0.1, max_tokens=800) -> LLMResult:
        started = time.perf_counter()
        try:
            client = self._client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=(choice.message.content or "").strip(),
            provider=self.name,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=choice.finish_reason or "stop",
            truncated=choice.finish_reason == "length",
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
        )

    async def stream(self, messages, *, temperature=0.1, max_tokens=800) -> AsyncIterator[str]:
        try:
            client = self._client()
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for event in stream:
                if not event.choices:
                    continue
                piece = event.choices[0].delta.content
                if piece:
                    yield piece
        except LLMError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc


# ---------------------------------------------------------------------------
# Google Gemini (REST — avoids pulling in another SDK)
# ---------------------------------------------------------------------------
class GeminiClient:
    name = "gemini"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, api_key: str, timeout: float = 45.0) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def health(self) -> bool:
        return bool(self.api_key)

    async def complete(self, messages, *, temperature=0.1, max_tokens=800) -> LLMResult:
        if not self.api_key:
            raise LLMUnavailable("no Gemini API key configured")
        started = time.perf_counter()
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.endpoint}/{self.model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url, json=payload, headers={"x-goog-api-key": self.api_key}
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeout("gemini timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"gemini unreachable: {exc}") from exc
        if response.status_code == 429:
            raise LLMRateLimited("gemini rate limit reached")
        if response.status_code >= 500:
            raise LLMTimeout(f"gemini returned {response.status_code}")
        if response.status_code >= 400:
            raise LLMError(f"gemini error {response.status_code}: {response.text[:200]}")
        data = response.json()
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts)
        return LLMResult(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=(candidates[0].get("finishReason", "stop") if candidates else "stop"),
        )

    async def stream(self, messages, *, temperature=0.1, max_tokens=800) -> AsyncIterator[str]:
        # Gemini supports SSE streaming; chunking the completed response keeps
        # one code path and is indistinguishable at this response size.
        result = await self.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        for piece in _soft_chunks(result.text):
            yield piece


# ---------------------------------------------------------------------------
# Offline extractive answerer
# ---------------------------------------------------------------------------
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveClient:
    """Answers by quoting the retrieved passages. No model, no network.

    It reads the same prompt the real providers receive, pulls the numbered
    context blocks back out of it, and selects the sentences with the highest
    overlap with the question. The output is always grounded by construction —
    it can only ever repeat text that was retrieved.
    """

    name = "extractive"
    model = "rule-based-extractive"

    async def health(self) -> bool:
        return True

    async def complete(self, messages, *, temperature=0.1, max_tokens=800) -> LLMResult:
        started = time.perf_counter()
        prompt = "\n\n".join(m.content for m in messages if m.role == "user")
        question = _last_question(prompt)
        blocks = _parse_context_blocks(prompt)
        tool_section = _parse_tool_results(prompt)

        parts: list[str] = []
        if tool_section:
            # Record data is exact and already scoped to the requester, so it
            # leads the answer rather than being buried under policy quotes.
            parts.append(f"From your HR records:\n\n{tool_section}")
        if blocks:
            heading = "What the policy says:" if tool_section else "Based on the retrieved policy text:"
            parts.append(f"{heading}\n\n{_extract_answer(question, blocks)}")
        if not parts:
            parts.append(
                "I could not find anything in the knowledge base that answers "
                "that. Try rephrasing, or contact People Operations."
            )
        text = "\n\n".join(parts)
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream(self, messages, *, temperature=0.1, max_tokens=800) -> AsyncIterator[str]:
        result = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        for piece in _soft_chunks(result.text):
            yield piece


CONTEXT_BLOCK_RE = re.compile(
    r"\[(\d+)\]\s*(?P<title>[^\n]*)\n(?P<body>.*?)(?=\n\[\d+\]\s|\Z)", re.DOTALL
)
QUESTION_RE = re.compile(r"(?:Question|User question):\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _last_question(prompt: str) -> str:
    matches = QUESTION_RE.findall(prompt)
    return matches[-1].strip() if matches else prompt.strip()[-300:]


TRAILER_RE = re.compile(r"\n(?:Question:|CONVERSATION SO FAR|Write the grounded answer)", re.M)


def _parse_context_blocks(prompt: str) -> list[tuple[int, str, str]]:
    """Extract the numbered SOURCES blocks, and only those.

    The final block runs to the end of the prompt, which would otherwise swallow
    the question and the closing instruction and let them be quoted back as if
    they were policy text.
    """
    sources_start = prompt.find("SOURCES")
    region = prompt[sources_start:] if sources_start >= 0 else prompt
    blocks: list[tuple[int, str, str]] = []
    for marker, title, body in CONTEXT_BLOCK_RE.findall(region):
        trailer = TRAILER_RE.search(body)
        if trailer:
            body = body[: trailer.start()]
        blocks.append((int(marker), title.strip(), body.strip()))
    return blocks


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2}


def _parse_tool_results(prompt: str) -> str:
    """Pull the TOOL RESULTS section (already grounded, already scoped) out."""
    start = prompt.find("TOOL RESULTS")
    if start < 0:
        return ""
    end = prompt.find("SOURCES", start)
    section = prompt[start : end if end > 0 else len(prompt)]
    lines = section.splitlines()[1:]  # drop the header line
    return "\n".join(line for line in lines if line.strip()).strip()


def _extract_answer(question: str, blocks: list[tuple[int, str, str]]) -> str:
    query_terms = _tokens(question)
    scored: list[tuple[float, int, str]] = []
    for marker, _title, body in blocks:
        for sentence in _candidate_sentences(body):
            terms = _tokens(sentence)
            if not terms:
                continue
            overlap = len(query_terms & terms) / (len(query_terms) or 1)
            # Slight preference for sentences carrying concrete numbers —
            # policy answers are usually thresholds, days and deadlines.
            if re.search(r"\d", sentence):
                overlap += 0.05
            if overlap > 0:
                scored.append((overlap, marker, sentence.strip()))
    scored.sort(key=lambda item: -item[0])
    if not scored:
        first_marker, _title, body = blocks[0]
        first = _candidate_sentences(body)[:2]
        return " ".join(f"{s.strip()} [{first_marker}]" for s in first)

    seen: set[str] = set()
    lines: list[str] = []
    for _score, marker, sentence in scored:
        key = sentence[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {sentence} [{marker}]")
        if len(lines) >= 5:
            break
    return "\n".join(lines)


LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")


def _candidate_sentences(body: str) -> list[str]:
    """Turn a markdown block into whole, quotable sentences.

    Source documents are hard-wrapped, so splitting on physical lines yields
    fragments like "not reduce the annual leave balance." Paragraph lines are
    therefore unwrapped first; list items and table rows stay as their own
    units because they are already self-contained.
    """
    units: list[str] = []
    paragraph: list[str] = []
    in_list_item = False

    def flush() -> None:
        nonlocal in_list_item
        if paragraph:
            units.append(" ".join(paragraph).strip())
            paragraph.clear()
        in_list_item = False

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
            if len(cells) >= 2 and not set("".join(cells)) <= set("-: "):
                units.append(": ".join(cells) + ".")
            continue
        if stripped.startswith("#"):
            flush()
            continue
        if LIST_ITEM_RE.match(line):
            flush()
            units.append(LIST_ITEM_RE.sub("", line).strip())
            in_list_item = True
            continue
        if in_list_item and line.startswith(("  ", "\t")):
            # Wrapped continuation of the bullet above.
            units[-1] = f"{units[-1]} {stripped}".strip()
            continue
        paragraph.append(stripped)
    flush()

    sentences: list[str] = []
    for unit in units:
        for sentence in SENTENCE_RE.split(unit):
            sentence = sentence.strip()
            if 25 < len(sentence) <= 320:
                sentences.append(sentence)
    return sentences


def _soft_chunks(text: str, size: int = 24) -> list[str]:
    """Split text into word-boundary pieces so simulated streaming reads well."""
    words = text.split(" ")
    return [" ".join(words[i : i + size]) + (" " if i + size < len(words) else "")
            for i in range(0, len(words), size)] or [text]
