"""Input screening, PII redaction and post-generation grounding verification.

Three checks, in order of when they run:

1. ``screen_input``   — before anything is spent: size limits, prompt-injection
   patterns, PII the employee should not have pasted, and sensitive topics that
   need a human rather than a chatbot.
2. ``redact``         — applied to everything written to logs and to the audit
   trail. The copilot can process an employee's email address; it should not
   permanently store it in a log file.
3. ``verify_answer``  — after generation: are the cited markers real, and is
   every factual claim actually cited? This produces the grounding score that
   drives the confidence badge and the "unverified" warning in the UI.

None of this is a substitute for a real safety stack. It is the lightweight,
concrete layer the brief asks for, and it is deliberately explainable: every
decision names the rule that fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3}[\s-]?\d{3,4}[\s-]?\d{0,4}(?!\w)")),
    ("national_id", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("dob", re.compile(r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b")),
]

# Health/medical detail an employee might volunteer. We do not block it — that
# would be hostile — but we never write it to the audit log.
SENSITIVE_TERMS = re.compile(
    r"\b(diagnos\w+|cancer|pregnan\w+|therapy|depress\w+|anxiety|disabilit\w+|"
    r"medication|surgery|miscarriage)\b",
    re.I,
)

INJECTION_PATTERNS = [
    re.compile(r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)\b", re.I),
    re.compile(r"\b(disregard|forget|override) (your|the|all) (instructions|rules|system prompt|guardrails)\b", re.I),
    re.compile(r"\b(reveal|print|show|repeat|output) (me )?(your |the )?(system )?(prompt|instructions|rules)\b", re.I),
    re.compile(r"\byou are now\b|\bact as (an? )?(unrestricted|jailbroken|dan)\b", re.I),
    re.compile(r"\bwithout (any )?(citations?|sources?)\b.*\b(answer|respond)\b", re.I),
    re.compile(r"\b(pretend|imagine) (that )?(you|the policy) (are|is|allows?)\b", re.I),
]

# Questions the copilot must not attempt, with the human route to give instead.
ESCALATION_TOPICS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(suicid\w+|kill myself|self[- ]harm|end my life|want to die)\b", re.I),
        "crisis",
    ),
    (
        re.compile(r"\b(harass\w+|discriminat\w+|bullie?d|bullying|assault\w*|retaliat\w+)\b", re.I),
        "conduct",
    ),
    (
        re.compile(r"\b(visa|work permit|immigration|green card|h1b|h-1b)\b", re.I),
        "immigration",
    ),
    (
        re.compile(r"\b(sue|lawsuit|legal action|solicitor|attorney|tribunal)\b", re.I),
        "legal",
    ),
]

ESCALATION_MESSAGES = {
    "crisis": (
        "I'm not the right place for this, and I don't want to give you a policy "
        "paragraph when a person would help more.\n\n"
        "Northwind Systems' Employee Assistance Programme is free, confidential and "
        "available 24×7 to you and your household — it is not visible to your manager "
        "or to HR. Please use it, or contact your local emergency services if you are "
        "in immediate danger.\n\n"
        "If it would help, I can also show you what the Benefits and Wellbeing policy "
        "says about mental health support."
    ),
    "conduct": (
        "This needs a person, not a chatbot, so let me point you to the route rather "
        "than summarise it.\n\n"
        "You can raise this with your line manager, your skip-level manager or your HR "
        "Business Partner, by email to speak-up@northwind-example.com (seen only by HR "
        "and Legal), or anonymously through the independent whistleblowing line. "
        "Retaliation for raising a concern in good faith is itself a disciplinary "
        "offence.\n\n"
        "I can show you the formal grievance timelines from the Code of Conduct if that "
        "would be useful."
    ),
    "immigration": (
        "I can't advise on visas, work permits or immigration status — those depend on "
        "your personal circumstances and the law where you are, and a wrong answer here "
        "is costly.\n\n"
        "Please contact People Operations (people-ops@northwind-example.com), who will "
        "route you to the mobility team. I can still answer questions about the "
        "working-from-another-country policy if that is what you need."
    ),
    "legal": (
        "I can't give legal advice or comment on a dispute.\n\n"
        "For anything with a legal dimension, contact your HR Business Partner, who will "
        "involve Legal. I can tell you what the internal grievance and appeal process "
        "looks like if that helps you decide what to do next."
    ),
}


class Action(str, Enum):
    allow = "allow"
    allow_with_notice = "allow_with_notice"
    refuse = "refuse"


@dataclass
class InputVerdict:
    action: Action
    reason: str = ""
    notices: list[str] = field(default_factory=list)
    replacement_answer: str = ""
    detected_pii: list[str] = field(default_factory=list)
    escalation_topic: str = ""


def redact(text: str) -> str:
    """Replace PII with typed placeholders. Used for every log and audit write."""
    redacted = text
    for label, pattern in PII_PATTERNS:
        redacted = pattern.sub(f"[{label} redacted]", redacted)
    if SENSITIVE_TERMS.search(redacted):
        redacted = SENSITIVE_TERMS.sub("[health detail redacted]", redacted)
    return redacted


def detect_pii(text: str) -> list[str]:
    return [label for label, pattern in PII_PATTERNS if pattern.search(text)]


def screen_input(question: str, *, max_chars: int = 1200) -> InputVerdict:
    """Screen a user message before any retrieval or generation happens."""
    stripped = question.strip()
    if not stripped:
        return InputVerdict(Action.refuse, reason="empty", replacement_answer="Please type a question.")

    if len(stripped) > max_chars:
        return InputVerdict(
            Action.refuse,
            reason="too_long",
            replacement_answer=(
                f"That message is {len(stripped)} characters, which is longer than this "
                f"copilot accepts ({max_chars}). Please ask one focused question — for "
                "example, the specific policy point you need."
            ),
        )

    for pattern in INJECTION_PATTERNS:
        if pattern.search(stripped):
            return InputVerdict(
                Action.refuse,
                reason="prompt_injection",
                replacement_answer=(
                    "I can only answer from Northwind Systems' published policy documents, "
                    "and I can't change those rules or share my own configuration.\n\n"
                    "If you're trying to get at something specific — an exception to a "
                    "policy, or how a decision was made — ask me the underlying question "
                    "and I'll show you what the documents say, or point you to the owner."
                ),
            )

    for pattern, topic in ESCALATION_TOPICS:
        if pattern.search(stripped):
            return InputVerdict(
                Action.refuse,
                reason=f"escalation:{topic}",
                replacement_answer=ESCALATION_MESSAGES[topic],
                escalation_topic=topic,
            )

    detected = detect_pii(stripped)
    notices: list[str] = []
    if detected:
        notices.append(
            "Your message contained personal details (" + ", ".join(detected) + "). "
            "They were not stored in the conversation log. Please avoid pasting personal "
            "data into the copilot."
        )
        return InputVerdict(Action.allow_with_notice, reason="pii_in_input", notices=notices, detected_pii=detected)

    return InputVerdict(Action.allow)


# ---------------------------------------------------------------------------
# Post-generation verification
# ---------------------------------------------------------------------------
MARKER_RE = re.compile(r"\[(\d{1,2})\]")
TRAILING_MARKER_RE = re.compile(r"([.!?])(\s*)((?:\[\d{1,2}\])+)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# A "checkable claim" is a sentence carrying a number, a currency amount, a
# duration or a named process — exactly the things a hallucination gets wrong.
CLAIM_RE = re.compile(
    r"(\b\d+(\.\d+)?\s*(days?|weeks?|months?|years?|hours?|minutes?|%)\b)"
    r"|(\bUSD\s?\d)|(\$\s?\d)|(\b\d{1,2}\s(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\b)|(\b(SEV[1-4]|P[1-4])\b)",
    re.I,
)


@dataclass
class GroundingReport:
    score: float
    valid: bool
    invalid_markers: list[int] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    used_markers: list[int] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def verify_answer(answer: str, citation_count: int, *, abstained: bool = False) -> GroundingReport:
    """Check that an answer is actually supported by the sources it was given.

    The score is the share of checkable claims that carry a citation, penalised
    for any citation marker that does not exist. It is a cheap proxy — it
    verifies *attribution*, not truth — and it is described that way in the UI.
    """
    if abstained:
        return GroundingReport(score=1.0, valid=True, notices=[])

    used = [int(marker) for marker in MARKER_RE.findall(answer)]
    invalid = sorted({marker for marker in used if marker < 1 or marker > citation_count})

    # Writers — models and humans alike — put the citation after the full stop
    # ("...24 days. [1]"). Splitting naively would orphan the marker onto the
    # next sentence and score a perfectly cited answer as ungrounded.
    normalised = TRAILING_MARKER_RE.sub(r" \3\1", answer)
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalised) if s.strip()]
    claims = [s for s in sentences if CLAIM_RE.search(s)]
    uncited = [s for s in claims if not MARKER_RE.search(s)]

    if not claims:
        # No checkable claim: score on whether the answer cites anything at all.
        score = 1.0 if used and not invalid else (0.6 if not citation_count else 0.5)
    else:
        score = (len(claims) - len(uncited)) / len(claims)

    if invalid:
        score = min(score, 0.35)

    notices: list[str] = []
    if invalid:
        notices.append(
            "This answer referenced a source number that does not exist; treat it with "
            "caution and check the linked policy."
        )
    elif uncited and len(uncited) >= max(1, len(claims) // 2):
        notices.append(
            "Some figures in this answer are not linked to a source. Please verify them "
            "against the cited policy before relying on them."
        )

    return GroundingReport(
        score=round(max(0.0, min(1.0, score)), 3),
        valid=not invalid,
        invalid_markers=invalid,
        uncited_claims=uncited[:3],
        used_markers=sorted(set(used)),
        notices=notices,
    )


def strip_invalid_markers(answer: str, citation_count: int) -> str:
    """Remove citation markers pointing at sources that were never supplied."""
    def replace(match: re.Match[str]) -> str:
        marker = int(match.group(1))
        return match.group(0) if 1 <= marker <= citation_count else ""

    return re.sub(r"\s*\[(\d{1,2})\]", lambda m: replace(m), answer).strip()
