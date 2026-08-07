# Prompt design and AI interaction improvements

All prompts live in [`backend/app/llm/prompts.py`](../backend/app/llm/prompts.py).

## 1. Three prompts, three jobs

One large prompt doing everything is hard to test and hard to fix: when the
answer is wrong you cannot tell whether retrieval, routing or writing failed.
The work is split into three small prompts, each independently observable in the
response payload.

| Prompt | Input | Output | Visible as |
|---|---|---|---|
| **Rewrite** | last message + 4 turns of history | one standalone search query | `rewritten_query` |
| **Plan** | question + tool descriptions | `{"tool", "arguments", "reason"}` JSON | `tool_calls[]` |
| **Answer** | question + numbered sources + tool results | grounded markdown with `[n]` markers | `answer`, `citations[]` |

Rewrite and plan run at `temperature=0.0` with tight token caps (60 and 120);
the answer runs at `0.1`. Low temperature throughout, because this is a factual
assistant and creative variance is a defect here.

## 2. Rules the system prompt is built on

Each rule exists because of a specific failure mode.

**Instruction before context, question last.** Models attend most reliably to
the start and the end of a long prompt. The grounding rules open the system
message; the employee's question is the final line before the answer.

**Numbered context blocks carrying metadata.**

```
[1] Paid Time Off and Leave Policy — 2. Annual leave (PTO) (HR-POL-001 · v4.2 · effective 2026-01-01 · owner: People Operations)
Full-time employees are entitled to **24 working days** …
```

The integer the model must cite is the same integer the UI renders, so a
citation cannot drift from its source. Version and effective date are included
so the model can reason about freshness and flag conflicts, and so a reviewer
reading the trace sees exactly what the model saw.

**Abstention is a named option with exact wording.** Without a concrete "reply
with exactly this sentence" instruction, models paraphrase adjacent policy text
instead of declining. The sentence is also what the orchestrator matches on to
classify the turn as `abstained`.

**Negative constraints are paired with a positive alternative.** "Do not guess"
alone is weakly followed; "do not guess — say what you *did* find and name the
team to contact" is followed reliably and produces a more useful answer.

**No chain-of-thought in the output.** Employees do not want it, it inflates
latency, and it lets the model argue itself into a conclusion the sources do not
support.

**Scope limits are explicit.** The copilot informs; it does not approve
requests, make decisions, or give legal, medical, immigration or tax advice, and
it only discusses the requester's own records. The last of these is defence in
depth — the real enforcement is in `ToolRegistry.call`.

## 3. Measured improvement: the citation-format example

The first version of the system prompt said, correctly and in prose, that every
factual claim must carry a citation. Measured on 49 evaluation cases with
`gpt-4o-mini`, **mean grounding was 0.605**: the model would open with an
uncited summary sentence containing the key number, then cite inside the bullets
underneath.

Adding a worked example — showing the marker at the end of each line, before the
full stop, plus one line naming the defect ("a line with a number and no marker
is a defect, even if the fact is obviously right") — moved it to **0.890**, with
key-fact recall unchanged at 0.93.

| Metric | Prose rule only | With worked example |
|---|---|---|
| Mean grounding score | 0.605 | **0.890** |
| Key-fact recall | 0.900 | 0.933 |
| Answer-type routing accuracy | 0.939 | 0.959 |

The lesson generalises: for formatting contracts, one example outperforms three
sentences of description.

## 4. Other AI interaction improvements

### 4.1 Query rewriting for follow-ups

Retrieval quality collapses on multi-turn chat without it. "What about carrying
them over?" contains no retrievable content — the rewriter turns it into
"annual leave carry over policy" using the previous turns.

There is an offline heuristic version too (`_heuristic_rewrite`): when a short
follow-up leans on a pronoun, salient nouns from the previous user turn are
appended. Crude, but it recovers most "what about X?" follow-ups at zero cost
and keeps the no-LLM path usable. The rewritten query is shown in the UI under
**Sources & reasoning**, so the user can see what was actually searched.

### 4.2 Rules-first tool routing

The deterministic router handles the common phrasings for free and cannot
hallucinate an argument; the LLM planner is consulted only for phrasing the
rules do not recognise. Measured tool-selection accuracy on the evaluation set:
**1.00**, both offline and with an LLM.

### 4.3 Tool results steer retrieval

Every tool returns a `policy_hint` that is appended to the search query, so a
balance lookup also retrieves the carry-over rules. Answers become "here is your
number, and here is the rule that governs it" instead of a bare figure.

### 4.4 Post-generation verification

`verify_answer` checks that every cited marker exists and that sentences
carrying a number, currency amount, duration or named process are attributed.
It normalises the common "…24 days. [1]" placement before splitting sentences —
without that, a perfectly cited answer scores as ungrounded. Markers pointing at
non-existent sources are stripped before display, the grounding score feeds the
confidence badge, and a low score attaches a visible caution.

This verifies **attribution, not truth** — that limit is stated in the UI
tooltip and in [ACCURACY_AND_LIMITATIONS](ACCURACY_AND_LIMITATIONS.md).

### 4.5 Deterministic abstention on calibrated coverage

Abstention is not delegated to the model. If the best passage covers less than
20% of the question's idf-weighted content terms, the copilot declines with a
fixed message naming the topics it *does* cover. On the evaluation set,
unanswerable questions score at or below 0.16 coverage and answerable ones at or
above 0.19 — a clean separation. Between 0.20 and 0.30, and where the query
contains vocabulary absent from the corpus, the answer is shown with an explicit
caution instead.

### 4.6 Follow-up suggestions without an extra call

Suggestions are generated from sibling section headings of the cited documents.
They cost no tokens, can only propose questions the knowledge base can actually
answer, and double as a discovery aid for a corpus nobody has read end to end.

## 5. Prompt-injection handling

Injection patterns are caught **before** retrieval, in
[`guardrails.py`](../backend/app/services/guardrails.py), so an attack costs no
tokens. The refusal explains what the copilot can do rather than lecturing.

Retrieved document text is not trusted either: it is delivered inside labelled
`[n]` blocks, the model is told sources are evidence and not instructions, and
markers that do not correspond to a supplied source are stripped after
generation. A poisoned document could still influence an answer — see the
threat note in [RESPONSIBLE_AI](RESPONSIBLE_AI.md#5-known-gaps).

## 6. What I would try next

- **Self-consistency on numeric answers.** Generate twice at temperature 0 and
  0.3; if the extracted figures disagree, show the disagreement rather than
  picking one.
- **A cross-encoder re-ranker** over the top 20 candidates. `leave-04` (notice
  period for a two-week holiday) fails because the offboarding policy's
  notice-period language out-matches the leave policy's table; a re-ranker that
  reads the question and passage together is the standard fix.
- **Structured output for answers** (JSON with `answer`, `claims[]`,
  `citations[]`) so verification becomes exact per-claim rather than
  sentence-heuristic. Deferred because it complicates streaming and is less
  reliable on small local models.
- **Automated prompt regression.** The evaluation harness already gives the
  numbers; wiring it into CI so a prompt edit that drops grounding below a
  threshold fails the build is a small step and a large safety net.
