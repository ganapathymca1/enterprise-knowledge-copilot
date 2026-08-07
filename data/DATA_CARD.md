# Data Card

## Summary

Every dataset in this repository is **synthetic** and was authored specifically
for this proof of concept. There is no real, private, company-confidential or
personal data anywhere in the repo, and no protected health information (PHI).

| Dataset | Path | Records | Nature |
|---|---|---|---|
| HR / operations knowledge base | `data/knowledge_base/*.md` | 11 policy documents | Synthetic, human-authored |
| Employee directory | `data/hr_records/employees.csv` | 24 rows | Synthetic |
| Leave balances | `data/hr_records/leave_balances.csv` | 96 rows | Synthetic |
| Leave requests | `data/hr_records/leave_requests.csv` | 30 rows | Synthetic |
| Holiday calendar | `data/hr_records/holidays.csv` | 42 rows | Synthetic, derived from public holiday names |
| Evaluation set | `eval/golden_set.json` | 32 questions | Synthetic, hand-labelled |

## Why synthetic rather than a Kaggle/GitHub dump

The assignment allows a public dataset **or** synthesized data. Synthesized data
was chosen deliberately:

1. **Governance.** The brief forbids private or confidential data. Authoring the
   corpus guarantees compliance and lets the repo be shared freely.
2. **Ground truth for evaluation.** Because the corpus is authored, every
   evaluation question has a known correct source document, which makes
   retrieval accuracy measurable instead of anecdotal (see `eval/`).
3. **Negative testing.** The corpus is deliberately *incomplete* — there is no
   relocation-bonus policy, no company car policy and no visa policy. That lets
   us test that the copilot **abstains** instead of hallucinating.
4. **Structured + unstructured mix.** Real enterprise questions ("how many leave
   days do I have left?") need a database, not a document. The CSVs exist so the
   tool-calling path is exercised, which a pure document dump cannot demonstrate.

The corpus is written in the register of a real handbook (GitLab-handbook style:
tables, thresholds, cross-references, ownership metadata) so retrieval behaviour
is representative of a real deployment.

## Fictional entities

* Company: **Northwind Systems** (fictional)
* Email domain: `@northwind-example.com` (reserved-style example domain)
* Systems: PeopleHub (HRIS), ExpenseHub, ServiceDesk — all fictional
* People: generated names not intended to refer to any real individual. Any
  resemblance is coincidental.

## Document metadata

Every knowledge base document carries YAML front matter used by the retriever
for filtering, citation rendering and freshness display:

```yaml
doc_id: HR-POL-001          # stable identifier used in citations
title: ...                  # displayed in the sources panel
category: ...               # used for coarse filtering / analytics
owner: ...                  # who to route feedback to
version: "4.2"
effective_date: 2026-01-01  # shown to the user so stale answers are visible
review_date: 2026-12-31
audience: ...
```

## Known limitations of the data

* The corpus is small (11 documents, ~250 chunks). Retrieval quality numbers in
  `docs/ACCURACY_AND_LIMITATIONS.md` will not transfer unchanged to a corpus of
  100k documents.
* Only English. No multilingual or localised policy variants.
* Policies are internally consistent by construction. A real handbook contains
  contradictions between documents and versions; the conflict-handling path is
  described in the accuracy note but only lightly exercised here.
* Leave balances are a point-in-time snapshot (`as_of_date = 2026-08-01`) rather
  than a live HRIS feed.
