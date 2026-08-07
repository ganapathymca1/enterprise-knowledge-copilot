# Architecture

## 1. One-screen view

```
 Browser (frontend/, no build step)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ app.js ── api.js ──► POST /api/chat/stream        (Server-Sent Events)│
 │   │                                                                   │
 │   └── renders: answer · citations · record cards · confidence · trace │
 └──────────────────────────────────────────────────────────────────────┘
                                    │
 FastAPI (backend/app/)             ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ api/chat.py ─► services/orchestrator.py                              │
 │                                                                       │
 │  1 screen      guardrails.screen_input   injection · PII · escalation │
 │  2 rewrite     llm + history             "what about it?" → standalone│
 │  3 route       tools.route_deterministically → llm planner (fallback) │
 │  4 tool        tools/registry.py         access control enforced here │
 │  5 retrieve    rag/index.py              BM25 ∪ LSA → RRF → MMR       │
 │  6 generate    llm/providers.py          retries, timeout, fallback   │
 │  7 verify      guardrails.verify_answer  markers valid? claims cited? │
 │  8 record      services/feedback.py      redacted audit row (SQLite)  │
 └──────────────────────────────────────────────────────────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
  data/knowledge_base   data/hr_records        LLM provider
  11 markdown policies  4 CSV record sets      ollama | openai-compatible
  → 117 chunks          → 5 tools              | gemini | extractive
```

## 2. Request lifecycle

A turn is a fixed pipeline, not an agent loop. Each stage is timed and returned
in `timings`, so latency is attributable in the UI rather than a single opaque
number.

| # | Stage | Fails how | Why it sits here |
|---|---|---|---|
| 1 | **Screen** | returns a refusal | Runs before anything is spent. A refusal costs no retrieval and no tokens. |
| 2 | **Rewrite** | falls back to the raw question | Must precede routing so the router sees a resolved question, not "what about that?". |
| 3 | **Route** | falls through to retrieval | Deterministic rules first (free, instant, cannot hallucinate); the LLM planner only for phrasing the rules do not know. |
| 4 | **Tool** | error becomes a user-visible notice | Before retrieval, so the tool's `policy_hint` can steer which policy passage is fetched alongside the record. |
| 5 | **Retrieve** | empty result → abstention | Hybrid search over the chunk corpus. |
| 6 | **Generate** | falls back to the extractive answerer | The only stage that needs a model, and the only one that can be skipped entirely. |
| 7 | **Verify** | downgrades confidence, attaches a notice | Never silently edits claims; only strips markers that point at nothing. |
| 8 | **Record** | logged, never raises | Auditing must not be able to break a chat turn. |

**Fail-soft is the rule.** A broken stage degrades the answer; it does not fail
the request. The one exception is screening, where failing closed is the point.

## 3. Retrieval

`backend/app/rag/`

**Chunking is heading-aware first, size-bounded second.** Handbook content is
structured — a heading plus its table is one semantic unit — so documents split
on H2/H3 boundaries, then pack paragraph blocks up to ~220 tokens with a
45-token overlap, never cutting a markdown table. Sections under 30 tokens are
merged into their neighbour: tiny chunks win on BM25 length normalisation while
carrying no answer. Every chunk is prefixed with its document title and heading
path, which is what makes a paragraph reading only "24 working days" findable
from "leave policy".

**Two rankers, fused by rank not by score.**

- *Lexical*: BM25, implemented directly (~50 lines) rather than pulled in as a
  dependency. The heading text is repeated twice in the indexed field — a cheap
  BM25F-style field boost. Without it, "escalation process" ranks *Severity
  levels* above *Escalation matrix*.
- *Semantic*: TF-IDF → truncated SVD (latent semantic analysis) from
  scikit-learn. No model weights, no download, no network, deterministic with a
  fixed seed. It captures term co-occurrence ("PTO" ↔ "annual leave") that BM25
  misses.

They are combined with **Reciprocal Rank Fusion** (k=10, not the literature's
60, which flattens a 20-candidate list into a tie). RRF is used because BM25
scores and cosine similarities live on different, corpus-dependent scales;
fusing ranks avoids inventing a normalisation constant that needs re-tuning per
corpus.

**Ordering and scoring are separated.** RRF decides order. The *reported* score
is a calibrated blend — `0.6 × idf-weighted query coverage + 0.4 × cosine` —
which means the same thing across queries and can therefore drive the
abstention threshold and the confidence badge. A raw BM25 score cannot do that.

**A document-level prior** (weight 0.15) lets a document's overall evidence lift
its own passages. Measured effect: the exact answer figure is present in the
model's context for 27/27 evaluation facts, up from 26/27, with no loss in
hit@1.

**MMR** (λ=0.7) removes near-duplicate passages, then results are re-sorted by
relevance so citation [1] is always the strongest evidence.

**Why not sentence-transformers?** It would add ~90 MB of weights and a
first-run download, which breaks "runs from a clean environment" on a
locked-down laptop. `EmbeddingBackend` is one isolated class with
`fit_transform`/`transform`, so swapping it for sentence-transformers or a
hosted embedding endpoint touches one file. See
[FUTURE_WORK](FUTURE_WORK.md).

## 4. Tools

`backend/app/tools/`

Five read-only tools over the synthetic HR record store: leave balances,
upcoming holidays, own leave requests, own profile, and the public directory.

Each returns three separate payloads, which is what keeps the layers honest:

- `summary` — dense text for the model,
- `data` — structured JSON the UI renders as a record card, so the user sees the
  *record* rather than the model's retelling of it,
- `policy_hint` — a query that pulls the governing policy alongside the record.
  This is what turns "you have 8.5 days left" into "you have 8.5 days left, and
  unused days above 5 expire on 31 March [2]".

**Routing** tries deterministic regex rules first, then an LLM planner that
emits JSON (`{"tool": ..., "arguments": {...}}`). JSON planning rather than
native function-calling APIs keeps every provider — including local Ollama
models — on the same code path.

**Access control is enforced in `ToolRegistry.call`, not in the prompt.** The
requester id comes from the server side; any `employee_id` the model produced is
discarded. A hallucinated `EMP-1005` cannot read that employee's balance. The
record store additionally exposes a `DIRECTORY_FIELDS` subset for colleague
lookups, so pay, level, hire date and balances are unreachable by construction.

## 5. LLM layer

`backend/app/llm/`

One interface (`health`, `complete`, `stream`), four implementations:
`ollama`, `openai_compatible` (OpenAI, Groq, LM Studio, OpenRouter, vLLM),
`gemini` (REST, no extra SDK), and `extractive`.

The **extractive** provider answers by quoting retrieved passages — no model, no
network. It is not a toy: it makes the zero-key reproduction path real, keeps
the test suite deterministic and free, and is the last rung of the degradation
ladder when a configured provider fails mid-request.

Retry policy lives in exactly one place (`with_retries`) and only retries errors
marked `retryable`. Retrying a 401 wastes the user's time and hides the real
problem, so provider exceptions are translated into a small typed taxonomy
(`LLMTimeout`, `LLMRateLimited`, `LLMUnavailable`) at the boundary.

## 6. State

| State | Where | Lifetime |
|---|---|---|
| Chunk index | process memory, built at startup (~700 ms for 117 chunks) | process |
| HR records | process memory, `lru_cache` over the CSV read | process |
| Conversations | `ConversationStore`, in-process dict with TTL | 120 min (configurable) |
| Answers + feedback | SQLite at `var/copilot.sqlite3` | durable |

`ConversationStore` is deliberately behind an async interface with TTL eviction
and bounded history — the shape a Redis implementation would have — so replacing
it is one class, not a refactor. Bounded history also keeps prompt size, latency
and cost constant however long a conversation runs.

## 7. Frontend boundary

The frontend is plain ES modules served by the same process. That is a
deliberate trade-off: the brief requires reproduction from a clean environment,
and a zero-dependency frontend removes node version drift, lockfile drift and
offline `npm install` failures from the critical path. The cost is no component
framework, no type checking and no test runner on the client — accepted for a
POC of this size, and revisited in [FUTURE_WORK](FUTURE_WORK.md).

The code is still organised the way a component tree would be: a small state
object, pure render helpers, and exactly one module (`api.js`) that knows about
HTTP. Porting to React means replacing the render helpers; the API contract does
not move. CORS already allows `localhost` origins for a separate dev server.

**Streaming contract.** `POST /api/chat/stream` emits:

| Event | Payload | When |
|---|---|---|
| `status` | `{stage, label}` | As the pipeline progresses |
| `sources` | citations + tool calls | After retrieval and verification |
| `delta` | `{text}` | Verified answer text, in word-boundary pieces |
| `done` | full response minus `answer` | End of turn |
| `error` | `{error, detail, retryable}` | Pipeline failure |

Answer text is streamed **after** verification, not token-by-token from the
model. That trades a little perceived speed for a hard guarantee: no claim is
ever displayed attributed to a source that does not exist. Stage events keep the
interface responsive in the meantime, which is what the perceived-latency win is
actually made of.

## 8. Scaling out

The design targets a laptop, but the seams are where they would need to be.

### More documents (10² → 10⁵+)

| Concern | Now | At scale |
|---|---|---|
| Index build | 700 ms in-process at startup | Offline indexing job; the API loads a prebuilt artefact |
| Vector store | numpy matrix, exhaustive cosine | pgvector / Qdrant / FAISS-IVF with ANN search |
| Embeddings | TF-IDF + SVD (`EmbeddingBackend`) | Swap the class for a transformer or hosted embedding endpoint |
| Lexical | in-memory BM25 | OpenSearch / Elasticsearch, keeping the same RRF fusion |
| Precision | RRF + document prior + MMR | Add a cross-encoder re-ranker over the top 20–50 candidates |
| Freshness | rebuild at startup | Incremental upsert on document change, with `effective_date` filters |

The retriever's public surface — `search(query, top_k, ...) -> [ScoredChunk]` —
does not change under any of these.

### More users

The app is stateless per request apart from `ConversationStore`, so it scales
horizontally behind a load balancer once sessions move to Redis. Add: per-user
and per-tenant rate limiting, a request queue with backpressure in front of the
LLM provider, a semantic cache keyed on the rewritten query (policy questions
repeat heavily — this is the single biggest cost lever), and a circuit breaker
per provider that trips to the extractive path instead of timing out.

Latency today: 30–60 ms p50 offline, ~4.5 s p50 with a hosted model, of which
retrieval is ~30 ms. The model dominates; batching and caching are where the
wins are.

### Stricter governance

Not built here, and each is a bounded addition rather than a redesign:

- **Identity**: replace the demo `X-Employee-Id` header with an SSO-verified
  JWT subject. One function (`_resolve_employee` in `api/chat.py`) changes.
- **Document-level ACLs**: attach an audience/group to each chunk's front matter
  and filter candidates by the caller's groups before ranking, so retrieval
  cannot surface a passage the user may not read.
- **Retention**: TTL and delete-by-subject on the audit tables; today the schema
  supports it but no job runs.
- **Regionalisation**: the corpus is single-locale; a real deployment needs
  jurisdiction-scoped policy variants and a locale filter in retrieval.
- **Model governance**: pin provider and model per environment, record them on
  every audit row (already done), and gate provider changes behind the same
  review as a code change.

## 9. Key design decisions

| Decision | Alternative | Why this one |
|---|---|---|
| Fixed pipeline | Autonomous agent loop | Predictable latency and cost, auditable path, no runaway tool loops. The tasks here are well-bounded. |
| Deterministic abstention | Ask the model to decline | A model asked to decline produces a fluent near-miss from whatever context it has. A fixed message is testable and cannot be confused with an answer. |
| Rules-then-LLM routing | LLM routing only | Free, instant and correct for the common phrasings; token spend goes only to genuine ambiguity. |
| Verify after generation | Trust the model | Citation markers are checkable. Checking them is cheap and turns "cited" into "verifiably cited". |
| Calibrated coverage score | Raw BM25 score | Comparable across queries, so it can drive thresholds and be shown to a user as a percentage. |
| Stream after verification | Token streaming | No unverified claim is ever displayed. Stage events preserve responsiveness. |
| Zero-build frontend | React + Vite | Removes an entire class of clean-environment failures from the reproduction path. |
| Synthetic corpus | Kaggle/GitLab dump | Gives ground truth for evaluation and lets the corpus be deliberately incomplete, so abstention is testable. |
