# Responsible AI and governance

A lightweight but concrete governance approach for an internal copilot, covering
intended use, explainability, feedback handling, accuracy and the controls that
back each of them. Everything described here is implemented in this repository
unless it is explicitly listed as a gap.

## 1. Intended use

**In scope.** Helping employees find and understand published internal policy,
and read their *own* HR records (leave balances, requests, holiday calendar,
profile).

**Out of scope, by design.**

- Making or approving decisions — leave approval, promotion, disciplinary
  outcomes, pay. The copilot describes process; humans decide.
- Legal, medical, immigration or tax advice.
- Anything about another employee beyond the public directory.
- Any use in a disciplinary or performance process. The Code of Conduct in the
  corpus states this explicitly, and the copilot repeats it.

**Users.** All employees. **Owner.** People Operations, jointly with the
Information Security team for the security corpus.

## 2. The controls

### 2.1 Grounding and abstention

- Answers may only use retrieved passages and tool results; the system prompt
  says so in its first rule.
- If the best passage covers less than 20% of the question's idf-weighted
  content terms, the copilot **declines** with a fixed message naming the topics
  it does cover. This is deterministic code, not a model instruction — a model
  asked to decline often produces a fluent near-miss instead.
- Between 20% and 30% coverage, or where the question contains vocabulary absent
  from the corpus, the answer carries a visible caution.
- Measured: **100%** of unanswerable evaluation questions were declined or
  flagged. See [ACCURACY_AND_LIMITATIONS](ACCURACY_AND_LIMITATIONS.md).

### 2.2 Explainability

Not a debug panel — the explanation is part of the answer.

| Surfaced | Where |
|---|---|
| Answer type: from policy / from your records / records + policy / declined / redirected | Badge on every answer |
| Confidence: high / medium / low, with the reason on hover | Badge on every answer |
| Every retrieved passage with doc id, **version**, **effective date** and owning team | Sources panel |
| Retrieval relevance per passage, as a percentage | Sources panel |
| Which passages the answer actually cited | Highlighted in the panel; `[n]` chips jump to them |
| Full policy text, scrolled to the cited section | "Open full policy" |
| Tools called, their arguments, latency and result | Sources panel + record card |
| The **search query actually used** after rewriting | Sources panel |
| Per-stage latency and the trace id | Sources panel |
| The complete corpus the copilot can answer from | Sidebar, always visible |

Publishing the corpus boundary is a governance control, not a convenience: an
employee who can see the edge of the knowledge base is much less likely to
over-trust an answer that falls outside it.

### 2.3 Verification before display

After generation and **before any text reaches the user**, the copilot checks
that every citation marker exists and that claims carrying numbers, dates,
durations or named processes are attributed. Invalid markers are stripped, the
grounding score sets the confidence badge, and a low score attaches a caution.
The SSE stream deliberately sends verified text rather than model tokens for
this reason.

This verifies attribution, not truth. That limit is stated in the UI.

### 2.4 Privacy and data protection

- **Synthetic data only.** No real, personal or company-confidential data is in
  this repository ([DATA_CARD](../data/DATA_CARD.md)).
- **Redaction at the boundary.** PII patterns (email, phone, national id, card
  number, date of birth) and volunteered health terms are replaced with typed
  placeholders by the *logging handler* and before every audit write — so a
  future `logger.info(user_message)` added in a hurry still cannot leak.
- **Free-text feedback is redacted before storage.** Employees paste anything
  into a comment box.
- **Pasted PII is flagged to the user**, not silently swallowed: the answer
  carries a notice saying the details were not stored.
- **Data minimisation.** The conversation store keeps a bounded window and
  expires sessions after 120 minutes.

### 2.5 Access control

Enforced in code, not in the prompt:

- The requester identity is server-side. Any `employee_id` the model produces is
  **discarded** in `ToolRegistry.call`, so a hallucinated id cannot read another
  employee's records.
- Colleague lookups return only `DIRECTORY_FIELDS` — name, role, team, location,
  work pattern, manager, email. Pay, level, hire date, balances and requests are
  unreachable for anyone but the requester.
- Measured: **0 leaks** on the access-control probe in the evaluation set.

### 2.6 Safety and escalation

Four topic classes are intercepted **before** retrieval and answered with a
human route rather than a policy summary:

| Topic | Response |
|---|---|
| Crisis / self-harm | Employee Assistance Programme first — free, confidential, 24×7, not visible to the manager or HR — and emergency services if in immediate danger |
| Harassment, discrimination, bullying, retaliation | Manager / skip-level / HRBP, `speak-up@`, or the anonymous whistleblowing line; states that retaliation is itself a disciplinary offence |
| Visa, work permit, immigration | People Operations and the mobility team |
| Legal disputes | HR Business Partner, who involves Legal |

In each case the copilot offers to show the relevant policy afterwards — the
human route leads, the document follows.

### 2.7 Prompt injection

Injection patterns are screened before retrieval, so an attack costs no tokens
and reaches no model. The refusal explains what the copilot *can* do instead of
lecturing. Retrieved text is delivered as labelled evidence, and post-generation
marker stripping limits what a manipulated answer can claim.

## 3. Feedback handling

Feedback that cannot be acted on is theatre. The loop here is closed:

1. **Capture.** Thumbs up/down on every answer; a downvote asks for a reason
   (incorrect, incomplete, out of date, wrong source, unclear, not relevant) and
   optional free text.
2. **Join to evidence.** Every answer already wrote an audit row — question,
   retrieved chunk ids, scores, tools, provider, model, grounding, latency — so
   feedback joins to *the passages that produced it*, not just to a message id.
3. **Route.** Each chunk carries its owning team in front matter, so a downvote
   is routed to that owner (`Finance Operations`, `Information Security`, …) and
   the UI tells the user where it went.
4. **Prioritise.** `GET /api/feedback/failing-passages` ranks the passages that
   appear most often in downvoted answers — the rewrite queue for the policy
   owner.
5. **Regress.** `GET /api/feedback/regression-candidates` exports downvoted
   questions in the shape of the evaluation set, so a fix can be proven and
   cannot silently regress.
6. **Monitor.** `GET /api/feedback/stats` gives volume, satisfaction rate,
   answer-type mix, mean grounding, mean latency and downvote reasons — the
   backing for a governance dashboard.

Suggested operating rhythm for a real deployment: weekly triage of the failing-
passage report by People Operations; monthly review of satisfaction, abstention
rate and grounding; any change to prompts, thresholds or the corpus re-runs the
evaluation set before release.

## 4. Accountability

| Question | Answer |
|---|---|
| Who owns the answers? | The policy owner named in each document's front matter |
| Who owns the system? | People Operations, with Information Security for the security corpus |
| Can an answer be reconstructed? | Yes — `trace_id` retrieves the question, retrieved chunk ids and scores, tools, provider, model and grounding score |
| Is the model recorded? | Yes, provider and model are on every audit row, so a behaviour change after a model swap is attributable |
| Can a user contest an answer? | Yes — thumbs down routes to the owning team; the underlying policies have their own appeal routes |
| Is AI used for decisions about people? | No, and the corpus itself states that AI tools are never used for disciplinary decisions |

## 5. Known gaps

Stated plainly, because a governance note that claims completeness is not
credible.

- **Identity is a demo header.** Anyone can act as anyone in this build. A real
  deployment reads an SSO-verified JWT subject; that is one function
  (`_resolve_employee`).
- **No document-level ACLs.** Every employee can retrieve every passage. Real
  handbooks contain manager-only and region-only content; the fix is an audience
  claim on each chunk, filtered before ranking.
- **A poisoned document could still influence answers.** Injection screening
  covers user input, not corpus content. Ingestion needs review and provenance
  checks before a corpus is writable by many authors.
- **No retention job.** The schema supports TTL and delete-by-subject; nothing
  runs them.
- **Redaction is regex-based.** It catches common formats, not all of them.
- **Thresholds are tuned on 49 cases.** They are calibrated, not proven.
- **No human-in-the-loop review queue** for low-confidence answers, and no
  rate limiting or abuse monitoring.
- **No bias evaluation.** The corpus is small and English-only; a real
  deployment should test for differential answer quality across topics that
  correlate with protected characteristics — parental leave, disability
  adjustments, religious holiday swaps — all of which exist in this corpus and
  none of which are separately measured here.
