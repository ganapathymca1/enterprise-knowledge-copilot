# What I would improve with more time

Ordered by value per unit of effort, with the reason each was cut.

## 1. Retrieval: cross-encoder re-ranking (highest value)

The one measured failure with a real model — `leave-04`, notice required for a
two-week holiday — happens because the offboarding policy's "notice period"
language out-matches the leave policy's notice table. First-stage ranking cannot
fix this; a cross-encoder that reads question and passage *together* over the
top 20 candidates is the standard remedy, typically worth several points of
precision@1.

Cut because a cross-encoder means model weights, and the zero-download
constraint was worth more for a submission that has to reproduce anywhere.

## 2. Transformer embeddings behind the existing seam

`EmbeddingBackend` exposes `fit_transform`/`transform` and nothing else knows
what is behind it. Swapping in `all-MiniLM-L6-v2` (or a hosted embedding
endpoint) is one class, and would remove the hand-curated synonym map in
`index.py`, which is a stopgap that does not scale past a small corpus.

Would come with: a persisted index artefact so startup does not re-embed, and an
ANN store (pgvector, Qdrant, FAISS-IVF) once the corpus exceeds a few thousand
passages.

## 3. Real identity and document-level access control

Two changes that matter more than anything else before this could touch real
data:

- Replace the demo `X-Employee-Id` header with an SSO-verified JWT subject. One
  function (`_resolve_employee`) changes.
- Attach an audience claim to each chunk's front matter and filter candidates by
  the caller's groups *before* ranking, so retrieval cannot surface a passage the
  user may not read. Manager-only and region-only content is normal in a real
  handbook.

## 4. Semantic answer caching

Policy questions repeat heavily. A cache keyed on the rewritten query, storing
the answer with its citations and invalidated when a source document's version
changes, is the single largest cost and latency lever at scale — likely a large
majority of production traffic. It needs the version-aware invalidation to be
correct, which is why it is not a two-line addition.

## 5. Structured answer output and per-claim verification

Have the model emit `{answer, claims[{text, citation}], caveats[]}` rather than
prose with inline markers. Verification becomes exact per claim instead of
sentence-heuristic, and the UI can highlight an individual unsupported claim.
Deferred because it complicates streaming and is less reliable on small local
models — it would need a graceful downgrade path.

## 6. Evaluation in CI, and an LLM judge

The harness already produces the numbers. Wiring it into CI so a prompt or
threshold change that drops grounding or hit@k below a floor fails the build is
small and high-leverage.

Beyond that, the current `must_include` substring labels are brittle — `it-01`
"fails" only because the model wrote "30-minute" instead of "30 minutes". A
second model grading answers against reference answers (with human spot-checks
on disagreements) would measure semantic correctness rather than string
presence, and would let the set grow without hand-labelling every fact.

## 7. Conversation and record durability

Move `ConversationStore` to Redis (the interface is already the right shape), so
conversations survive a restart and the service scales horizontally. Move the
audit and feedback tables to Postgres, add retention TTLs and a delete-by-subject
job — the schema supports both today, but nothing runs them.

## 8. Frontend: React port and client tests

The zero-build choice was right for reproducibility, and it is the first thing I
would revisit for a real product. A Vite + React + TypeScript port would give
component tests, type-checked API contracts generated from the OpenAPI schema,
and virtualised transcript rendering for long conversations. The API contract
does not move, so this is a rewrite of the render layer only.

Also worth adding: an automated accessibility check (axe-core) in CI, and
keyboard-driven navigation of the citation panel.

## 9. Operational hardening

- Per-user and per-tenant rate limiting, and a request queue with backpressure
  in front of the LLM provider.
- A circuit breaker per provider that trips to the extractive path rather than
  timing out repeatedly.
- OpenTelemetry traces spanning the pipeline stages that are already timed.
- A `/metrics` endpoint (answer types, abstention rate, grounding distribution,
  provider errors) so the governance dashboard is scraped rather than polled.

## 10. Content operations

The feedback loop identifies failing passages but stops at reporting them.
Closing it properly means: a review queue for policy owners, a diff view when a
document version changes, automatic re-indexing on change, and a regression run
against the evaluation set before a corpus update goes live.

## 11. Coverage the corpus does not have

Conflicting and superseded policy versions are the interesting hard case in real
handbooks, and this corpus barely exercises them. I would add deliberately
conflicting documents with different `effective_date` values and measure whether
the copilot surfaces the conflict and cites both, as the prompt instructs — that
behaviour is currently asserted rather than measured.
