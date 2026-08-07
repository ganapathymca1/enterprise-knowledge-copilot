# Assumption log

Decisions taken where the brief left room, and what each one costs.

## Scope and dataset

**A1. Synthesized corpus rather than a public dump.** The brief allows either.
Authoring it gives ground truth for evaluation, guarantees no confidential data,
and — crucially — lets the corpus be *deliberately incomplete* so abstention is
testable. Cost: retrieval numbers reflect an 11-document corpus, not a real
handbook. Rationale in [DATA_CARD](../data/DATA_CARD.md).

**A2. Structured records as well as documents.** "Internal employee related
data" in the brief is not answerable from prose — "how many days do I have
left?" needs a database. Four CSV record sets exist so the tool-calling path is
genuinely exercised rather than simulated.

**A3. The company is fictional (Northwind Systems)** with fictional systems
(PeopleHub, ExpenseHub, ServiceDesk). Any resemblance to a real organisation is
coincidental.

**A4. Records are a point-in-time snapshot**, not a live HRIS feed. Every record
answer states its `as_of` date so a stale number is visible as stale.

## Product behaviour

**A5. The copilot informs; it never decides.** No approvals, no
recommendations on pay, performance or discipline. This is asserted in the
prompt and reinforced by the corpus itself.

**A6. Abstention beats a plausible answer.** Where the corpus is thin the
copilot declines and names a human. For an internal HR assistant a wrong answer
about entitlements is more expensive than an unanswered question.

**A7. Employees see only their own records.** Directory lookups return published
fields only. Enforced in code, not by prompt instruction.

**A8. English, single jurisdiction.** No localisation or regional policy
variants.

**A9. Today's date is the system date.** The holiday calendar covers 2026;
running this in a later year will return no upcoming holidays. Deliberate — the
tool reports emptiness rather than inventing dates.

## Technical

**A10. FastAPI over Node.** The brief allows either. Python keeps retrieval,
evaluation and service in one language and one dependency set.

**A11. Local lexical + LSA retrieval rather than transformer embeddings.**
Transformer weights mean a ~90 MB first-run download, which breaks clean-
environment reproduction on a locked-down machine. The embedding backend is one
isolated class so the upgrade is a single swap. Cost: no true semantic matching;
quantified in [ACCURACY_AND_LIMITATIONS](ACCURACY_AND_LIMITATIONS.md).

**A12. A zero-build frontend.** Plain ES modules served by the same process, so
the documented run command is one command with no node toolchain in the
reproduction path. Cost: no component framework, no client-side type checking or
test runner. Accepted for a POC of this size.

**A13. The app must run with no API key.** The extractive provider is a
first-class path, not a stub — it keeps the test suite free, offline and
deterministic, and doubles as the last rung of the degradation ladder.

**A14. Answer text streams after verification.** No unverified claim is ever
displayed. Costs a little perceived speed; stage events cover it.

**A15. In-process session storage.** A dict with TTL, behind an async interface
shaped like the Redis implementation that would replace it. Restarting the
server loses conversations.

**A16. Identity via an `X-Employee-Id` header.** Demo only, confined to one
function so the SSO swap is contained. **Anyone can act as anyone** in this
build.

**A17. Fixed pipeline, not an agent loop.** Predictable latency and cost, an
auditable path, and no runaway tool loops. The tasks here are well-bounded
enough not to need planning-by-model.

**A18. Deterministic abstention thresholds** (coverage < 0.20 declines, < 0.30
cautions) calibrated on the evaluation set, where unanswerable questions score
≤ 0.16 and answerable ones ≥ 0.19. Calibrated, not proven — a larger corpus
would need re-tuning, which is why they are named constants with a comment
pointing at the measurement.

**A19. Dependencies pinned to the exact versions built and tested against**
(Python 3.14 on Windows), with a seeded data generator and a fixed SVD seed, so
an offline evaluation run reproduces exactly.

**A20. `gpt-4o-mini` used for the reported LLM measurements** because a key was
available in the development environment. Nothing in the code depends on it —
Ollama, Groq and Gemini are configured paths, and the README leads with the free
options.
