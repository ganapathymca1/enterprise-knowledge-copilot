# Accuracy and limitations

## 1. How this was measured

49 hand-labelled cases in [`eval/golden_set.json`](../eval/golden_set.json),
run through the **real orchestrator** — screening, rewriting, routing, tools,
retrieval, generation and verification — by
[`eval/run_eval.py`](../eval/run_eval.py). No mocks, no HTTP layer, no network
in the default mode.

```bash
python eval/run_eval.py                                  # offline, deterministic
python eval/run_eval.py --provider openai_compatible     # with a real model
```

Case mix: 28 specific-fact policy questions, 3 broad "summarise this area"
questions, 1 cross-document question, 5 record-tool lookups, 3 multi-turn
follow-ups, 4 unanswerable questions, 4 guardrail probes and 1 access-control
probe.

Recorded runs are committed under [`eval/results/`](../eval/results/).

## 2. Results

| Metric | Offline (extractive) | `gpt-4o-mini` |
|---|---|---|
| Retrieval hit@1 (document level) | 0.892 | 0.892 |
| Retrieval hit@5 (document level) | **1.000** | **1.000** |
| Mean reciprocal rank | 0.927 | 0.922 |
| Key fact present in retrieved context | 0.933 | 0.933 |
| Key-fact recall **in the answer** | 0.733 | **0.933** |
| Answer-type routing accuracy | **1.000** | 0.959 |
| Tool-selection accuracy | **1.000** | **1.000** |
| Abstained or warned on unanswerable | **1.000** | **1.000** |
| Access-control leaks | **0** | **0** |
| Mean grounding score (answering turns) | 0.981 | 0.890 |
| Latency p50 | 30 ms | 4.6 s |
| Latency p95 | 44 ms | 7.6 s |

Hardware: a 2024 Windows laptop, Python 3.14, no GPU. The hosted-model latency
is almost entirely provider round-trip; retrieval is ~30 ms of it.

### Reading these numbers

**Fact-in-context (0.933) is reported separately from fact recall on purpose.**
It isolates retrieval from generation. The gap between the two columns —
0.733 offline versus 0.933 with a model — is not a retrieval difference at all;
the same context was available in both runs. It is the cost of having no model:
the extractive answerer selects sentences by term overlap and misses a figure
that sits in a table row it did not rank. That is the honest price of the
zero-key reproduction path, and it is why the UI states plainly when no model is
configured.

**Grounding is higher offline (0.981) than with a model (0.890)** because the
extractive answerer can only emit sentences it copied, each stamped with its
source. A generative model paraphrases across passages and occasionally leaves a
lead sentence uncited. Higher grounding here means *more mechanical
attribution*, not better answers.

**Grounding measures attribution, not truth.** It checks that claims carrying
numbers, dates, durations or named processes cite a marker, and that every
marker exists. A confidently wrong sentence with a valid citation scores 1.0.
The UI tooltip says so.

## 3. What the evaluation still gets wrong

Three cases fail with a real model, and each is instructive.

| Case | What happens | Diagnosis |
|---|---|---|
| `leave-04` — notice for a two-week holiday | Retrieval surfaces the **offboarding** notice-period section; the model abstains rather than answering from it | Genuine lexical clash: "notice period" is the offboarding policy's core vocabulary. The system behaved correctly — it declined instead of answering from the wrong policy. Fix is a cross-encoder re-ranker, not a prompt change. |
| `tool-03` — pending leave requests | The tool returns the requester's requests, none pending; the model leads with "I could not find this" | The abstention sentence is being used for "no matching records" as well as "no policy". Tool emptiness needs its own response path. |
| `it-01` — SLA for a P1 ticket | Answer is correct ("**30-minute** first response") but the label expects the string "30 minutes" | A labelling artefact, not a model error. Exact-substring labels are brittle for numbers; a semantic check would be better. |

I have left all three visible in the harness output rather than tuning the
labels to make the numbers look better.

## 4. Limitations

### Corpus

- **14 documents, 117 passages.** Retrieval numbers on a 100k-document corpus
  will be lower; hit@5 = 1.000 here says the pipeline is sound at this scale,
  not that it is solved.
- **Synthetic and internally consistent.** A real handbook contains
  contradictions between documents and superseded versions. The prompt tells the
  model to surface conflicts and cite both sides, but this corpus barely
  exercises that path.
- **English only, one jurisdiction.** No localised policy variants.
- **Records are a point-in-time snapshot** (`as_of 2026-08-01`), not a live HRIS
  feed. Answers state the as-of date for exactly this reason.

### Retrieval

- Lexical + LSA has no real semantic understanding. It handles synonyms it has
  seen co-occur; it will miss a paraphrase with no shared vocabulary. The
  hand-curated synonym map in `index.py` is a stopgap that would not scale — a
  transformer embedding model is the real answer.
- No cross-encoder re-ranking, so precision@1 depends on first-stage ranking.
- Table-heavy answers are the weak spot: the figure lives in a row whose wording
  matches the question less well than surrounding prose.

### Generation

- Quality tracks the model. A 7B local model produces noticeably weaker
  summaries than a hosted model on the same context.
- Verification checks attribution, not correctness. It cannot catch a
  well-cited misreading.
- No self-consistency check on numeric answers.

### Product

- Session history is in-process: restarting the server loses conversations.
- Identity is a demo header, not SSO. **Anyone can act as anyone** in this
  build — acceptable for a POC on synthetic data, unacceptable in production.
- No document-level access control: every employee can retrieve every passage.
- The frontend has no automated tests; it was verified manually and in a
  headless browser run (0 console errors, 0 failed requests).

## 5. Risks and how they are mitigated

| Risk | Mitigation here | Residual |
|---|---|---|
| **Incorrect answer stated confidently** | Deterministic abstention below a calibrated coverage floor; grounding verification; confidence badge; citations always shown | A well-cited misreading still reads as authoritative |
| **Over-reliance on the copilot** | Sources open to full policy text; corpus boundary published in the sidebar; explicit "informational only" scope; escalation to named humans | Users may still skip reading the source |
| **Answering outside the corpus** | Coverage floor plus unknown-vocabulary warning; measured 100% caution rate on unanswerable questions | Thresholds are tuned on 49 cases, not thousands |
| **Exposing another employee's data** | Access control in the tool registry from server-side identity; directory limited to public fields; 0 leaks measured | No document-level ACLs yet |
| **Prompt injection** | Pre-retrieval pattern screening; sources delivered as labelled evidence; invalid markers stripped | A poisoned *document* could still influence an answer |
| **Stale policy quoted as current** | `effective_date` and version on every citation and in the model's context | Nothing forces a rebuild when a document changes |
| **Sensitive topics handled by a bot** | Harassment, crisis, immigration and legal questions route to named humans before retrieval | Pattern matching misses paraphrases |
| **PII in logs** | Redaction at the logging handler and before every audit write | Redaction is regex-based and imperfect |

## 6. Reproducing

```bash
pytest -q                                    # 72 tests, ~5 s, offline
python eval/run_eval.py --verbose            # metrics + failing cases
python eval/run_eval.py --retrieval-only     # retrieval only, no generation
python eval/run_eval.py --json out.json      # machine-readable
```

The offline run is deterministic: fixed SVD seed, seeded data generator, no
network. Runs against a hosted model vary by a case or two between runs, which
is itself a reason the offline mode is the default for CI.
