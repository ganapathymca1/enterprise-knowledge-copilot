"""The chat orchestration pipeline.

    screen → rewrite → route → (tool) → retrieve → generate → verify → record

Each stage is small, individually testable, and fails soft: a stage that breaks
degrades the answer rather than the request. The order is not arbitrary —

* screening runs first so a refusal costs no retrieval and no tokens;
* rewriting runs before routing so the router sees a resolved, standalone
  question rather than "what about part-timers?";
* a tool result is fetched *before* retrieval so its ``policy_hint`` can steer
  which policy passage gets pulled alongside the record;
* verification runs after generation and can downgrade confidence or attach a
  warning, but never silently edits the model's claims.

The whole pipeline is instrumented: every stage's duration lands in the
response, so latency is attributable in the UI rather than a single number.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..llm import (
    ExtractiveClient,
    LLMClient,
    LLMError,
    Message,
    build_client,
    with_retries,
)
from ..llm.prompts import (
    ABSTAIN_SENTENCE,
    build_answer_messages,
    build_planner_messages,
    build_rewrite_messages,
)
from ..models import (
    AnswerType,
    ChatRequest,
    ChatResponse,
    Citation,
    Confidence,
    Role,
    TimingBreakdown,
    ToolCall,
)
from ..rag import KnowledgeIndex, ScoredChunk
from ..tools import POLICY_SEARCH, ToolRegistry, route_deterministically
from . import guardrails
from .conversation import ConversationStore
from .feedback import FeedbackStore

logger = logging.getLogger(__name__)

# Confidence thresholds. Kept as named constants because they are product
# decisions, not implementation details — see docs/ACCURACY_AND_LIMITATIONS.md.
HIGH_RELEVANCE = 0.45
MEDIUM_RELEVANCE = 0.28
HIGH_GROUNDING = 0.80
MEDIUM_GROUNDING = 0.55

# Query-coverage bands. Calibrated on the evaluation set, where questions the
# corpus cannot answer score at or below 0.16 and answerable ones at or above
# 0.19 (see docs/ACCURACY_AND_LIMITATIONS.md). Below the first band the copilot
# declines; between the two it answers with an explicit caution.
ABSTAIN_COVERAGE = 0.20
CAUTION_COVERAGE = 0.30

DEGRADED_NOTICE = (
    "The configured language model was unavailable, so this answer was produced "
    "by the offline fallback, which quotes the policy text directly instead of "
    "summarising it."
)


@dataclass
class PipelineContext:
    """Everything a single turn accumulates. Also what gets audited."""

    trace_id: str
    session_id: str
    employee_id: str
    question: str
    rewritten: str = ""
    scored: list[ScoredChunk] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_summaries: list[str] = field(default_factory=list)
    tool_payloads: list[dict[str, Any]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)
    timings: TimingBreakdown = field(default_factory=TimingBreakdown)


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        index: KnowledgeIndex,
        tools: ToolRegistry,
        conversations: ConversationStore,
        feedback: FeedbackStore,
        llm: LLMClient,
    ) -> None:
        self.settings = settings
        self.index = index
        self.tools = tools
        self.conversations = conversations
        self.feedback = feedback
        self.llm = llm
        self.fallback = ExtractiveClient()

    # -- public API -------------------------------------------------------
    async def answer(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        settings = self.settings
        employee_id = request.employee_id or settings.default_employee_id
        session = await self.conversations.get_or_create(request.session_id, employee_id)
        context = PipelineContext(
            trace_id=f"trc_{uuid.uuid4().hex[:16]}",
            session_id=session.session_id,
            employee_id=employee_id,
            question=request.message.strip(),
        )

        # 1. Screening ----------------------------------------------------
        stage = time.perf_counter()
        verdict = guardrails.screen_input(context.question, max_chars=settings.max_question_chars)
        context.timings.guardrail_ms = _ms(stage)
        if verdict.action is guardrails.Action.refuse:
            return await self._finalise_refusal(context, verdict, started)
        context.notices.extend(verdict.notices)

        await self.conversations.append(session.session_id, Role.user, context.question)
        history = await self.conversations.history(session.session_id, settings.history_turns)
        prior = history[:-1]  # exclude the message we just appended

        # 2. Query rewriting ---------------------------------------------
        stage = time.perf_counter()
        context.rewritten = await self._rewrite(context.question, prior)
        context.timings.rewrite_ms = _ms(stage)

        # 3. Routing and tool execution ----------------------------------
        stage = time.perf_counter()
        use_tools = settings.enable_tools if request.use_tools is None else request.use_tools
        policy_hint = ""
        if use_tools:
            policy_hint = await self._run_tools(context)
        context.timings.tool_ms = _ms(stage)

        # 4. Retrieval ----------------------------------------------------
        stage = time.perf_counter()
        search_query = f"{context.rewritten} {policy_hint}".strip()
        context.scored = self.index.search(
            search_query,
            top_k=request.top_k or settings.top_k,
            candidate_k=settings.candidate_k,
            min_score=settings.min_retrieval_score,
            mmr_lambda=settings.mmr_lambda,
        )
        context.timings.retrieval_ms = _ms(stage)

        tool_succeeded = any(call.ok for call in context.tool_calls)
        if not context.scored and not tool_succeeded:
            return await self._finalise_abstention(context, started)

        # Vocabulary check. Words the corpus has never seen are the clearest
        # signal that a fluent-looking answer may be about the wrong thing —
        # "what is the relocation bonus?" retrieves the compensation policy with
        # a healthy similarity score while the corpus has no relocation policy
        # at all. Two thresholds, because the right response is graded: refuse
        # when the question is mostly unfamiliar, caution when it is partly so.
        context.unknown_terms = self.index.unknown_terms(context.rewritten)
        top_coverage = max((item.coverage for item in context.scored), default=0.0)
        if not tool_succeeded:
            if top_coverage < ABSTAIN_COVERAGE:
                # Coverage alone decides: it is calibrated, bounded and
                # comparable across queries. The unknown-term list is used only
                # to *explain* the refusal when it happens to be available.
                return await self._finalise_abstention(context, started)
            if context.unknown_terms and top_coverage < CAUTION_COVERAGE:
                context.notices.append(_unknown_terms_notice(context.unknown_terms))

        # 5. Generation ---------------------------------------------------
        stage = time.perf_counter()
        answer_text, provider, model, degraded = await self._generate(context, prior)
        context.timings.generation_ms = _ms(stage)
        if degraded:
            context.notices.append(DEGRADED_NOTICE)

        # 6. Verification -------------------------------------------------
        stage = time.perf_counter()
        citations = self._build_citations(context.scored)
        answer_text = guardrails.strip_invalid_markers(answer_text, len(citations))
        abstained = ABSTAIN_SENTENCE.lower() in answer_text.lower()
        report = guardrails.verify_answer(answer_text, len(citations), abstained=abstained)
        context.notices.extend(report.notices)
        context.timings.verification_ms = _ms(stage)

        answer_type = self._classify(context, abstained)
        confidence = self._confidence(context, report.score, abstained)
        context.timings.total_ms = _ms(started)

        response = ChatResponse(
            trace_id=context.trace_id,
            session_id=context.session_id,
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            answer=answer_text,
            answer_type=answer_type,
            confidence=confidence,
            citations=citations,
            tool_calls=context.tool_calls,
            followups=self._followups(context),
            notices=_dedupe(context.notices),
            provider=provider,
            model=model,
            rewritten_query=context.rewritten if context.rewritten != context.question else None,
            grounding_score=report.score,
            timings=context.timings,
        )
        await self.conversations.append(session.session_id, Role.assistant, answer_text)
        self._audit(context, response)
        return response

    # -- stages -----------------------------------------------------------
    async def _rewrite(self, question: str, history) -> str:
        """Resolve follow-ups into a standalone query. Falls back to the input."""
        if not history:
            return question
        if isinstance(self.llm, ExtractiveClient):
            # No model available: a heuristic still beats nothing for pronouns.
            return _heuristic_rewrite(question, history)
        try:
            result = await with_retries(
                lambda: self.llm.complete(
                    build_rewrite_messages(question, history),
                    temperature=0.0,
                    max_tokens=60,
                ),
                attempts=1,
                label="rewrite",
            )
            candidate = result.text.strip().strip('"').splitlines()[0] if result.text.strip() else ""
            # Guard against a model that "helpfully" answers instead of rewriting.
            if 3 <= len(candidate) <= 200 and not candidate.lower().startswith("i "):
                return candidate
        except (LLMError, Exception) as exc:  # noqa: BLE001 - never fail the turn
            logger.info("query rewrite failed (%s); using the raw question", exc.__class__.__name__)
        return question

    async def _run_tools(self, context: PipelineContext) -> str:
        """Decide whether a record lookup is needed, and run it.

        The deterministic router runs first: it is free, instant and cannot
        hallucinate an argument. The LLM planner is consulted only for
        questions the rules do not recognise, which keeps token spend
        proportional to genuine ambiguity.
        """
        decision = route_deterministically(context.rewritten) or route_deterministically(
            context.question
        )
        if decision is None and not isinstance(self.llm, ExtractiveClient):
            decision = await self._plan_with_llm(context)
        if decision is None:
            return ""

        name, arguments = decision
        if name == POLICY_SEARCH or name not in self.tools:
            return ""

        result, latency_ms = self.tools.call(name, context.employee_id, arguments)
        context.tool_calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                ok=result.ok,
                latency_ms=latency_ms,
                summary=result.summary[:400] if result.ok else "",
                error=result.error,
            )
        )
        if not result.ok:
            logger.info("tool %s did not return data: %s", name, result.error)
            if result.error:
                context.notices.append(result.error)
            return ""
        context.tool_summaries.append(result.summary)
        context.tool_payloads.append(result.data)
        return result.policy_hint

    async def _plan_with_llm(self, context: PipelineContext) -> tuple[str, dict[str, Any]] | None:
        try:
            result = await with_retries(
                lambda: self.llm.complete(
                    build_planner_messages(
                        context.rewritten, self.tools.describe(), context.employee_id
                    ),
                    temperature=0.0,
                    max_tokens=120,
                ),
                attempts=1,
                label="planner",
            )
        except Exception as exc:  # noqa: BLE001 - planning is best-effort
            logger.info("planner unavailable (%s); defaulting to retrieval", exc.__class__.__name__)
            return None
        payload = _extract_json(result.text)
        if not payload:
            return None
        name = str(payload.get("tool", "")).strip()
        if not name or name == POLICY_SEARCH or name not in self.tools:
            return None
        arguments = payload.get("arguments")
        return name, arguments if isinstance(arguments, dict) else {}

    async def _generate(self, context: PipelineContext, history) -> tuple[str, str, str, bool]:
        messages = build_answer_messages(
            context.question,
            context.scored,
            history=history,
            tool_summary="\n\n".join(context.tool_summaries),
        )
        try:
            result = await with_retries(
                lambda: self.llm.complete(
                    messages,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_output_tokens,
                ),
                attempts=self.settings.llm_max_retries,
                label="answer",
            )
            if result.text.strip():
                return result.text.strip(), result.provider, result.model, False
            logger.warning("provider %s returned an empty answer; falling back", result.provider)
        except Exception as exc:  # noqa: BLE001 - degrade, never 500
            logger.warning(
                "generation failed on %s (%s); falling back to the extractive answerer",
                getattr(self.llm, "name", "unknown"),
                exc.__class__.__name__,
            )
        fallback = await self.fallback.complete(messages)
        return fallback.text, fallback.provider, fallback.model, True

    # -- helpers ----------------------------------------------------------
    def _build_citations(self, scored: list[ScoredChunk]) -> list[Citation]:
        return [
            Citation(
                marker=marker,
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                title=item.chunk.title,
                section=item.chunk.section,
                category=item.chunk.category,
                owner=item.chunk.owner,
                version=item.chunk.version,
                effective_date=item.chunk.effective_date,
                snippet=item.chunk.citation_snippet(),
                score=item.score,
                source_path=item.chunk.source_path,
            )
            for marker, item in enumerate(scored, start=1)
        ]

    def _classify(self, context: PipelineContext, abstained: bool) -> AnswerType:
        if abstained:
            return AnswerType.abstained
        tool_ok = any(call.ok for call in context.tool_calls)
        if tool_ok and context.scored:
            return AnswerType.hybrid
        if tool_ok:
            return AnswerType.tool
        return AnswerType.grounded

    def _confidence(self, context: PipelineContext, grounding: float, abstained: bool) -> Confidence:
        if abstained:
            return Confidence.high  # a correct refusal is a confident answer
        top = context.scored[0].score if context.scored else 0.0
        tool_ok = any(call.ok for call in context.tool_calls)
        if context.unknown_terms and not tool_ok:
            # The question contains vocabulary the corpus has never seen: the
            # retrieved passages may be about a neighbouring topic entirely.
            return Confidence.low if len(context.unknown_terms) > 1 else Confidence.medium
        if tool_ok and grounding >= MEDIUM_GROUNDING:
            # Structured records are exact; confidence is limited by the
            # surrounding policy explanation, not by the number itself.
            return Confidence.high if (top >= MEDIUM_RELEVANCE or not context.scored) else Confidence.medium
        if top >= HIGH_RELEVANCE and grounding >= HIGH_GROUNDING:
            return Confidence.high
        if top >= MEDIUM_RELEVANCE and grounding >= MEDIUM_GROUNDING:
            return Confidence.medium
        return Confidence.low

    def _followups(self, context: PipelineContext) -> list[str]:
        """Suggest next questions from sibling sections of the cited documents.

        Deterministic on purpose: it costs no extra LLM call, it can only
        propose questions the knowledge base can actually answer, and it doubles
        as a discovery aid for a corpus the employee has never read.
        """
        if not context.scored:
            return []
        seen_docs: list[str] = []
        suggestions: list[str] = []
        cited_sections = {item.chunk.section for item in context.scored}
        for item in context.scored:
            doc_id = item.chunk.doc_id
            if doc_id in seen_docs:
                continue
            seen_docs.append(doc_id)
            for chunk in self.index.chunks:
                if chunk.doc_id != doc_id or chunk.section in cited_sections:
                    continue
                label = _section_label(chunk.section)
                if not label:
                    continue
                suggestions.append(f"What does the {chunk.title} say about {label.lower()}?")
                break
            if len(suggestions) >= 3:
                break
        return suggestions[:3]

    def _audit(self, context: PipelineContext, response: ChatResponse) -> None:
        """Persist the answer trace. Everything is redacted before it is stored."""
        try:
            self.feedback.record_answer(
                {
                    "trace_id": context.trace_id,
                    "session_id": context.session_id,
                    "employee_id": context.employee_id,
                    "question": guardrails.redact(context.question),
                    "rewritten": guardrails.redact(context.rewritten),
                    "answer": guardrails.redact(response.answer),
                    "answer_type": response.answer_type.value,
                    "confidence": response.confidence.value,
                    "grounding": response.grounding_score,
                    "provider": response.provider,
                    "model": response.model,
                    "latency_ms": response.timings.total_ms,
                    "chunk_ids": [c.chunk_id for c in response.citations],
                    "doc_ids": [c.doc_id for c in response.citations],
                    "owners": [c.owner for c in response.citations if c.owner],
                    "top_score": response.citations[0].score if response.citations else 0.0,
                    "tools": [call.name for call in response.tool_calls],
                }
            )
        except Exception as exc:  # noqa: BLE001 - auditing must never break chat
            logger.error("failed to write audit record: %s", exc)

    async def _finalise_refusal(self, context, verdict, started: float) -> ChatResponse:
        context.timings.total_ms = _ms(started)
        response = ChatResponse(
            trace_id=context.trace_id,
            session_id=context.session_id,
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            answer=verdict.replacement_answer,
            answer_type=AnswerType.refused,
            confidence=Confidence.high,
            notices=verdict.notices,
            provider="guardrail",
            model=verdict.reason,
            followups=_REFUSAL_FOLLOWUPS.get(verdict.escalation_topic, []),
            timings=context.timings,
        )
        self._audit(context, response)
        return response

    async def _finalise_abstention(self, context: PipelineContext, started: float) -> ChatResponse:
        """No passage cleared the relevance floor — decline, deterministically.

        Abstention is not delegated to the model. A model asked to decline will
        often produce a fluent near-miss from whatever context it was given;
        returning a fixed message here makes the behaviour testable and makes
        "we don't have that policy" indistinguishable from "we do, and here it
        is" impossible.
        """
        topics = sorted({document.title for document in self.index.documents})[:6]
        if context.unknown_terms:
            listed = ", ".join(f"**{term}**" for term in context.unknown_terms[:3])
            gap = (
                f"Nothing in the published policies mentions {listed}, so I have no "
                "grounded source to answer from.\n\n"
            )
        else:
            gap = (
                "I only answer from Northwind Systems' published policy documents, and "
                "nothing in them matched your question closely enough for me to give you "
                "a reliable answer.\n\n"
            )
        answer = (
            f"{ABSTAIN_SENTENCE}\n\n"
            f"{gap}"
            "You could try naming the policy area directly, or contact People Operations "
            "(people-ops@northwind-example.com) if it is an HR question, IT ServiceDesk "
            "for tooling, or Information Security for anything security related.\n\n"
            "Topics I do cover include: " + ", ".join(topics) + "."
        )
        context.timings.total_ms = _ms(started)
        response = ChatResponse(
            trace_id=context.trace_id,
            session_id=context.session_id,
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            answer=answer,
            answer_type=AnswerType.abstained,
            confidence=Confidence.high,
            notices=_dedupe(context.notices),
            provider="retrieval",
            model="abstention",
            tool_calls=context.tool_calls,
            timings=context.timings,
        )
        await self.conversations.append(context.session_id, Role.assistant, answer)
        self._audit(context, response)
        return response


_REFUSAL_FOLLOWUPS = {
    "conduct": ["What are the formal grievance timelines?", "How do I raise a concern anonymously?"],
    "crisis": ["What mental health support does the company provide?"],
    "immigration": ["Can I work from another country temporarily?"],
    "legal": ["What is the appeal process for a grievance outcome?"],
}


def _unknown_terms_notice(terms: list[str]) -> str:
    plural = len(terms) > 1
    listed = ", ".join(f"'{term}'" for term in terms[:3])
    return (
        f"The term{'s' if plural else ''} {listed} {'do' if plural else 'does'} not appear "
        "anywhere in the knowledge base. There may be no policy covering this — treat the "
        "answer below as related material rather than an answer to that specific point."
    )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


SECTION_NUMBER_RE = re.compile(r"^\d+\.\s*")


def _section_label(section: str) -> str:
    label = SECTION_NUMBER_RE.sub("", (section or "").split("›")[-1]).strip()
    return label if 3 <= len(label) <= 60 else ""


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response that may be wrapped in prose."""
    match = JSON_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


PRONOUN_RE = re.compile(r"\b(it|that|this|they|those|them|there)\b", re.I)


def _heuristic_rewrite(question: str, history) -> str:
    """Offline stand-in for LLM rewriting.

    If a short follow-up leans on a pronoun, prepend the salient nouns from the
    previous user turn. Crude, but it recovers most "what about X?" follow-ups
    and costs nothing.
    """
    if len(question.split()) > 12 or not PRONOUN_RE.search(question):
        return question
    previous = next(
        (m.content for m in reversed(history) if m.role == Role.user),
        "",
    )
    if not previous:
        return question
    keywords = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", previous.lower())
        if word not in {"what", "when", "does", "with", "from", "have", "many", "much", "policy"}
    ][:6]
    return f"{question} {' '.join(keywords)}".strip()


async def build_orchestrator(
    settings: Settings,
    index: KnowledgeIndex,
    tools: ToolRegistry,
    conversations: ConversationStore,
    feedback: FeedbackStore,
) -> Orchestrator:
    llm = await build_client(settings)
    return Orchestrator(
        settings=settings,
        index=index,
        tools=tools,
        conversations=conversations,
        feedback=feedback,
        llm=llm,
    )
