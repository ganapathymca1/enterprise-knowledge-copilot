# Enterprise Knowledge Copilot

An internal chat assistant that answers employee questions about HR policy and
their own HR records — grounded in a published knowledge base, cited, verified,
and honest about what it does not know.

```
┌───────────────┐   SSE    ┌──────────────────────────────────────────────┐
│  Chat UI      │ ───────► │  FastAPI orchestrator                        │
│  (no build)   │ ◄─────── │  screen → rewrite → route → tool → retrieve  │
└───────────────┘          │        → generate → verify → record          │
                           └───┬──────────────┬──────────────┬────────────┘
                               │              │              │
                     hybrid retrieval    HR record tools   LLM provider
                     BM25 + LSA + RRF    (access-scoped)   (pluggable)
```

**Everything runs locally and free.** No paid service is required, and the app
starts and answers with **no API key at all** — see [Quick start](#quick-start).

---

## Contents

| What you want | Where |
|---|---|
| Run it in two minutes | [Quick start](#quick-start) |
| Point it at a real LLM | [Configuring the LLM](#configuring-the-llm) |
| How it works and why | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Prompt design and the AI improvements | [docs/PROMPT_DESIGN.md](docs/PROMPT_DESIGN.md) |
| Measured accuracy and honest limits | [docs/ACCURACY_AND_LIMITATIONS.md](docs/ACCURACY_AND_LIMITATIONS.md) |
| Responsible AI and governance | [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) |
| Assumptions made | [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) |
| What I'd do with more time | [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) |
| Where the data came from | [data/DATA_CARD.md](data/DATA_CARD.md) |

---

## Quick start

Requirements: **Python 3.11 or newer**. Nothing else — no Node, no Docker, no
model download, no API key.

```bash
# 1. Clone and enter the repository
git clone <your-fork-url> enterprise-knowledge-copilot
cd enterprise-knowledge-copilot

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run it
uvicorn backend.app.main:app --port 8000
```

Open **<http://localhost:8000>**.

With no LLM configured the copilot runs in **offline extractive mode**: it
retrieves the right passages and answers by quoting them, with citations. The
UI says so in the status panel. Everything else — retrieval, tools, guardrails,
citations, verification, feedback — is fully live.

### Verify the install

```bash
pytest -q                 # 72 tests, offline, ~5 seconds
python eval/run_eval.py   # 49-case evaluation, offline, ~5 seconds
```

Both are network-free and deterministic.

---

## Configuring the LLM

Copy the template and edit:

```bash
cp .env.example .env
```

The `COPILOT_LLM_PROVIDER` setting selects the backend. `auto` (the default)
picks the first one that is actually usable.

### Option A — Ollama (free, local, recommended)

```bash
# install from https://ollama.com, then:
ollama pull llama3.1:8b
```

```dotenv
COPILOT_LLM_PROVIDER=ollama
COPILOT_OLLAMA_MODEL=llama3.1:8b
```

### Option B — Groq free tier (fast, hosted)

```dotenv
COPILOT_LLM_PROVIDER=openai_compatible
COPILOT_OPENAI_BASE_URL=https://api.groq.com/openai/v1
COPILOT_OPENAI_MODEL=llama-3.3-70b-versatile
COPILOT_OPENAI_API_KEY=gsk_...
```

### Option C — Google Gemini free tier

```dotenv
COPILOT_LLM_PROVIDER=gemini
COPILOT_GEMINI_MODEL=gemini-2.0-flash
COPILOT_GEMINI_API_KEY=...
```

### Option D — OpenAI, LM Studio, OpenRouter, vLLM

Any OpenAI-compatible `/v1` endpoint works through `openai_compatible`; set the
base URL, model and key. If `COPILOT_OPENAI_API_KEY` is unset the standard
`OPENAI_API_KEY` environment variable is used.

Restart the server after changing `.env`, then check the provider in the
sidebar or at `GET /api/health`.

> **Keys are never committed.** `.env` is gitignored; only `.env.example` is
> tracked, and it contains no secrets.

---

## Using it

Try these — the first three come straight from the brief:

- *Summarize our leave policy*
- *What is the escalation process for incidents?*
- *Find information about onboarding steps*
- *How many annual leave days do I have left?* → runs a record lookup
- *What are my next public holidays?* → scoped to your work location
- *What is the relocation bonus?* → **declines**, because no such policy exists

Use **Acting as** in the sidebar to switch employee identity and watch the
record answers change. Open **Sources & reasoning** under any answer to see the
retrieved passages, their relevance, the tools that ran, the search query
actually used, the stage timings and the trace id.

---

## Configuration reference

Every setting is an environment variable prefixed `COPILOT_`, or a line in
`.env`. The ones you are most likely to touch:

| Variable | Default | Purpose |
|---|---|---|
| `COPILOT_LLM_PROVIDER` | `auto` | `auto`, `ollama`, `gemini`, `openai_compatible`, `extractive` |
| `COPILOT_TOP_K` | `5` | Passages passed to the model |
| `COPILOT_CANDIDATE_K` | `20` | Candidates considered before fusion and MMR |
| `COPILOT_MIN_RETRIEVAL_SCORE` | `0.12` | Relevance floor for a passage |
| `COPILOT_CHUNK_TOKENS` | `220` | Target chunk size |
| `COPILOT_HISTORY_TURNS` | `6` | Turns of history kept for prompting |
| `COPILOT_TEMPERATURE` | `0.1` | Low, because this is a factual assistant |
| `COPILOT_ENABLE_TOOLS` | `true` | Turn off to force document-only answers |
| `COPILOT_REDACT_PII_IN_LOGS` | `true` | PII redaction on logs and audit records |
| `COPILOT_KNOWLEDGE_BASE_DIR` | `data/knowledge_base` | Point at your own corpus |

See [.env.example](.env.example) for the full list.

### Using your own documents

Drop markdown files into `data/knowledge_base/` and restart. Front matter is
optional but improves citations:

```markdown
---
doc_id: HR-POL-020
title: Travel Insurance Policy
category: Benefits
owner: People Operations
version: "1.0"
effective_date: 2026-04-01
---

# Travel Insurance Policy
...
```

---

## API

Full interactive documentation at **<http://localhost:8000/docs>**.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Ask a question, get a cited answer (JSON) |
| `POST` | `/api/chat/stream` | Same, streamed as Server-Sent Events |
| `GET` | `/api/sessions` · `DELETE /api/sessions/{id}` | Conversation management |
| `POST` | `/api/feedback` | Thumbs up/down with a reason |
| `GET` | `/api/feedback/stats` | Aggregate quality metrics |
| `GET` | `/api/feedback/failing-passages` | Passages in downvoted answers |
| `GET` | `/api/feedback/regression-candidates` | Downvoted questions for the eval set |
| `GET` | `/api/health` | Liveness, provider readiness, corpus size |
| `GET` | `/api/corpus` · `/api/documents/{id}` | What the copilot can answer from |
| `GET` | `/api/directory` | Public directory (demo identity selector) |

```bash
curl -s localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Employee-Id: EMP-1002' \
  -d '{"message":"How many annual leave days do I have left?"}' | jq
```

Every response carries `trace_id`, `answer_type`, `confidence`, `citations[]`,
`tool_calls[]`, `grounding_score`, `notices[]` and a per-stage `timings`
breakdown. The full contract is in
[`backend/app/models.py`](backend/app/models.py).

---

## How the frontend and backend fit together

The frontend is plain ES modules in `frontend/`, served by the same FastAPI
process at `/` (assets under `/assets`). One process, one command, no CORS to
configure, no build step, no `node_modules`.

It talks to the backend over exactly three calls: `POST /api/chat/stream` for a
turn, `POST /api/feedback` for a vote, and a few read-only `GET`s to populate
the sidebar. `frontend/js/api.js` is the only module that knows about HTTP;
`app.js` renders. CORS is enabled for `localhost` origins so you can point a
React or Vite dev server at this API instead — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#frontend-boundary).

**Streaming order matters:** the backend streams pipeline *stages* immediately,
then the sources, and only then the answer text — because the text is streamed
**after** citation verification has run. The copilot never shows a claim
attributed to a source that does not exist.

---

## Repository layout

```
backend/app/
  main.py            FastAPI app, lifespan, middleware, static mount
  config.py          typed settings (env / .env)
  models.py          request & response contracts
  api/               chat (JSON + SSE), feedback, system endpoints
  rag/               loading, heading-aware chunking, hybrid index
  llm/               provider abstraction, prompts, retry policy
  tools/             HR record store + tool registry with access control
  services/          orchestrator, guardrails, conversation, feedback store
frontend/            index.html + css/ + js/ (no build step)
data/
  knowledge_base/    14 synthetic policy documents
  hr_records/        synthetic employees, balances, requests, holidays
  DATA_CARD.md       provenance and limitations
eval/                golden set, harness, recorded results
tests/               72 offline tests
scripts/             seeded data generator
docs/                architecture, prompts, accuracy, responsible AI
```

---

## What this demonstrates

- **Prompt design** — three small prompts (rewrite, plan, answer) with an
  explicit abstention path and a citation contract. One worked example in the
  system prompt raised measured grounding from 0.61 to 0.89.
- **Tool / function orchestration** — provider-portable JSON planning with a
  deterministic regex router first, and access control enforced server-side.
- **Backend integration** — FastAPI, typed contracts, SSE streaming, structured
  logging, graceful degradation across four LLM providers.
- **User-centric AI workflow** — citations, confidence, record cards, follow-up
  suggestions, per-stage progress, and a copilot that says "I don't know".
- **Responsible AI** — abstention, PII redaction, audit trail, feedback routed
  to policy owners, and a published corpus boundary.

Measured results and their limits: **[docs/ACCURACY_AND_LIMITATIONS.md](docs/ACCURACY_AND_LIMITATIONS.md)**.

---

## Licence

MIT — see [LICENSE](LICENSE). All data in this repository is synthetic.
