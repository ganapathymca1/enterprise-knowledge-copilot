"""Prompt design.

Three prompts do three jobs, each small enough to reason about and test:

1. ``REWRITE`` — turns a follow-up ("what about part-timers?") into a
   standalone retrieval query. Retrieval quality collapses on multi-turn chat
   without this; it is the single highest-leverage accuracy improvement here.
2. ``PLANNER`` — chooses between answering from documents and calling an HR
   record tool, and extracts the tool arguments. Emitted as JSON so it works on
   providers with no native function-calling API.
3. ``ANSWER`` — writes the grounded, cited reply.

Design rules applied throughout (see docs/PROMPT_DESIGN.md for the rationale
and the failures each rule was written against):

* **Instruction before context, question last.** Models attend most reliably to
  the beginning and the end of a long prompt.
* **Numbered context blocks with metadata.** The marker the model must cite is
  the same integer the UI renders, so a citation cannot drift from its source.
* **Abstention is an explicit, named option**, not an afterthought. Without a
  concrete "say this exact thing" instruction, models paraphrase adjacent
  policy text instead of declining.
* **Negative constraints are paired with a positive alternative** ("do not
  guess — say what you could not find and who to ask"), which models follow far
  more reliably than a bare prohibition.
* **No chain-of-thought in the output.** Reasoning is not shown to employees;
  it inflates latency and invites the model to argue itself into a conclusion
  the sources do not support.
"""

from __future__ import annotations

from .base import Message

ABSTAIN_SENTENCE = (
    "I could not find this in the knowledge base."
)

SYSTEM_PROMPT = """You are the Enterprise Knowledge Copilot for Northwind Systems, an \
internal assistant that answers employee questions about HR policies, ways of working \
and internal processes.

GROUNDING RULES — these override everything else:
1. Answer ONLY from the numbered SOURCES and TOOL RESULTS provided in the user turn. \
Your own knowledge of how other companies work is not evidence.
2. Cite the source for every factual claim using its marker, like [1] or [2][3]. Every \
sentence containing a number, a deadline, an entitlement or a named process must carry \
a citation.
3. If the sources do not contain the answer, reply exactly: "{abstain}" — then say what \
you *did* find that is nearby, and point the employee to the right owner (People \
Operations for HR topics, IT ServiceDesk for tooling, Information Security for security). \
Never fill a gap with a plausible-sounding policy.
4. Never invent a marker, a policy number, a threshold, a date or a monetary amount.
5. If sources disagree or one looks out of date, say so and cite both, rather than \
silently choosing one.

SCOPE AND SAFETY:
6. You provide information about policy. You do not approve requests, make decisions, or \
give legal, medical, immigration or tax advice. For those, direct the employee to the \
named owner.
7. Only discuss the personal records of the employee who is asking. If asked about \
another named individual's leave, pay, performance or personal data, decline and explain \
that the copilot only exposes the requester's own records plus the public directory.
8. For anything involving harassment, discrimination, safety or a possible data breach, \
give the policy route and the human contact — and lead with the human contact.

STYLE:
9. Lead with the direct answer in one or two sentences, then the supporting detail as \
short bullets. Keep it under 200 words unless the question genuinely needs a procedure.
10. Use the employee's own vocabulary. Prefer concrete numbers and named systems from the \
sources over vague summaries.
11. Plain markdown only: short paragraphs, bullets, and bold for key numbers. No headings, \
no preamble like "Certainly" and no restatement of the question.

CITATION FORMAT — put the marker at the end of every line that states a fact, before the \
full stop. Follow this shape exactly:

Full-time employees get **24 days** of paid annual leave per calendar year [1].

- Up to **5 unused days** carry over, and they expire on **31 March** [1].
- Requests of 10 days or more need **20 working days** of notice [2].

A line with a number, a date, a threshold or a named process and no marker is a defect, \
even if the fact is obviously right.
""".format(abstain=ABSTAIN_SENTENCE)


REWRITE_PROMPT = """Rewrite the user's latest message into a single standalone search \
query for a policy knowledge base.

Rules:
- Resolve pronouns and ellipsis using the conversation ("what about part-timers?" after a \
question about leave becomes "part-time employee annual leave entitlement").
- Keep the domain vocabulary the user used; add the obvious policy term if the user used \
an abbreviation.
- No question mark, no preamble, no explanation. Output the query text only, at most 20 \
words.
- If the latest message is already standalone, output it unchanged.

Conversation so far:
{history}

Latest message: {question}

Standalone search query:"""


PLANNER_PROMPT = """You route an internal HR assistant's questions. Decide whether the \
question needs a live record lookup, or whether it can be answered from policy documents.

Available tools:
{tools}

Rules:
- Use a tool ONLY when the question asks about live, employee-specific or calendar data \
(balances, dates, who someone's manager is, upcoming holidays).
- Use "search_policies" for anything about what the policy says, thresholds, processes or \
entitlements.
- The requester is employee_id={employee_id}. Never fill an employee_id argument with \
anyone else's identifier; if the question names another person, still use the requester's \
id and let the tool enforce access.
- Reply with a single JSON object and nothing else:
  {{"tool": "<tool name or search_policies>", "arguments": {{...}}, "reason": "<12 words max>"}}

Question: {question}

JSON:"""


ANSWER_PROMPT = """{tool_section}SOURCES
Each block starts with its citation marker. Cite these markers in your answer.

{sources}

{history_section}Question: {question}

Write the grounded answer now, following the grounding rules. Cite markers inline."""


NO_SOURCES_PROMPT = """No policy documents matched this question, and no tool returned \
data.

Question: {question}

Reply with exactly "{abstain}", then in one or two sentences suggest how the employee \
could rephrase, and name the team to contact (People Operations for HR, IT ServiceDesk \
for tooling, Information Security for security). Do not answer from general knowledge."""


def format_sources(scored_chunks) -> str:
    """Render retrieved chunks as numbered, metadata-bearing context blocks.

    The header line carries doc id, version and effective date so the model can
    reason about freshness and conflicts, and so a reviewer reading the trace
    sees exactly what the model saw.
    """
    blocks: list[str] = []
    for marker, scored in enumerate(scored_chunks, start=1):
        chunk = scored.chunk
        meta = " · ".join(
            part
            for part in (
                chunk.doc_id,
                f"v{chunk.version}" if chunk.version else "",
                f"effective {chunk.effective_date}" if chunk.effective_date else "",
                f"owner: {chunk.owner}" if chunk.owner else "",
            )
            if part
        )
        heading = f"{chunk.title} — {chunk.section}" if chunk.section else chunk.title
        blocks.append(f"[{marker}] {heading} ({meta})\n{chunk.text}")
    return "\n\n".join(blocks)


def format_history(history, limit: int = 6) -> str:
    recent = history[-limit:]
    if not recent:
        return ""
    lines = [
        f"{'Employee' if message.role.value == 'user' else 'Copilot'}: {message.content.strip()}"
        for message in recent
    ]
    return "\n".join(lines)


def build_rewrite_messages(question: str, history) -> list[Message]:
    return [
        Message(role="system", content="You rewrite chat messages into search queries."),
        Message(
            role="user",
            content=REWRITE_PROMPT.format(
                history=format_history(history, limit=4) or "(no previous turns)",
                question=question,
            ),
        ),
    ]


def build_planner_messages(question: str, tools_description: str, employee_id: str) -> list[Message]:
    return [
        Message(role="system", content="You output only compact JSON. No prose."),
        Message(
            role="user",
            content=PLANNER_PROMPT.format(
                tools=tools_description,
                question=question,
                employee_id=employee_id,
            ),
        ),
    ]


def build_answer_messages(
    question: str,
    scored_chunks,
    *,
    history=None,
    tool_summary: str = "",
) -> list[Message]:
    if not scored_chunks and not tool_summary:
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=NO_SOURCES_PROMPT.format(
                    question=question, abstain=ABSTAIN_SENTENCE
                ),
            ),
        ]
    history_text = format_history(history or [])
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=ANSWER_PROMPT.format(
                tool_section=(
                    f"TOOL RESULTS (live records for the employee asking)\n{tool_summary}\n\n"
                    if tool_summary
                    else ""
                ),
                sources=format_sources(scored_chunks) or "(none)",
                history_section=(
                    f"CONVERSATION SO FAR\n{history_text}\n\n" if history_text else ""
                ),
                question=question,
            ),
        ),
    ]
