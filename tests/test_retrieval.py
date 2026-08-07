"""Retrieval layer: chunking, ranking, scoring and the abstention signal."""

from __future__ import annotations

import pytest

from backend.app.rag import chunk_document, load_documents
from backend.app.rag.index import BM25Index, tokenize


def test_documents_load_with_front_matter(settings):
    documents = load_documents(settings.knowledge_base_dir)
    assert len(documents) >= 10
    leave = next(d for d in documents if d.doc_id == "HR-POL-001")
    assert leave.title == "Paid Time Off and Leave Policy"
    assert leave.owner == "People Operations"
    assert leave.effective_date == "2026-01-01"
    # Front matter must be stripped from the body, or it becomes retrievable text.
    assert "doc_id:" not in leave.body


def test_chunks_carry_heading_path_and_metadata(settings):
    document = next(
        d for d in load_documents(settings.knowledge_base_dir) if d.doc_id == "HR-POL-001"
    )
    chunks = chunk_document(document, chunk_tokens=220, overlap_tokens=45)
    assert chunks, "the leave policy must produce at least one chunk"
    assert all(chunk.doc_id == "HR-POL-001" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("HR-POL-001#") for chunk in chunks)
    assert any("Annual leave" in chunk.section for chunk in chunks)
    # The heading path is part of the retrievable text — that is what makes a
    # paragraph saying only "24 working days" findable by "leave policy".
    sample = chunks[1]
    assert sample.title in sample.embedding_text
    assert sample.header in sample.lexical_tokens_text()


def test_tiny_sections_are_merged_away(settings):
    """Sub-30-token chunks win on BM25 length normalisation and answer nothing."""
    documents = load_documents(settings.knowledge_base_dir)
    for document in documents:
        for chunk in chunk_document(document):
            assert len(chunk.text.split()) >= 12, f"{chunk.chunk_id} is too small to be useful"


def test_markdown_tables_are_not_split(settings):
    document = next(
        d for d in load_documents(settings.knowledge_base_dir) if d.doc_id == "OPS-POL-004"
    )
    chunks = chunk_document(document)
    escalation = next(c for c in chunks if "Escalation matrix" in c.section)
    # The whole matrix must survive in one chunk: half a table is unusable.
    assert "Duty Director" in escalation.text
    assert "Service on-call engineer" in escalation.text


@pytest.mark.parametrize(
    "question,expected_doc",
    [
        ("How many days of annual leave do I get?", "HR-POL-001"),
        ("What is the escalation process for incidents?", "OPS-POL-004"),
        ("What happens on my first day?", "HR-POL-003"),
        ("How do I claim an expense?", "FIN-POL-006"),
        ("When do I get paid?", "HR-POL-007"),
        ("Can I work from home?", "HR-POL-005"),
        ("How do I report a lost laptop?", "SEC-POL-011"),
        ("What is my notice period?", "HR-POL-012"),
    ],
)
def test_retrieval_finds_the_right_document(index, question, expected_doc):
    hits = index.search(question, top_k=5, min_score=0.12)
    assert expected_doc in [hit.chunk.doc_id for hit in hits], f"{question} -> {hits[:1]}"


def test_scores_are_calibrated_and_ordered(index):
    hits = index.search("how much annual leave do I get", top_k=5, min_score=0.0)
    assert hits
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
    assert hits == sorted(hits, key=lambda hit: -hit.score)
    assert hits[0].coverage > 0.3


def test_off_topic_question_retrieves_nothing(index):
    assert index.search("what is the capital of France", top_k=5, min_score=0.12) == []


def test_unknown_terms_flag_missing_topics_but_tolerate_word_forms(index):
    assert "tuition" in index.unknown_terms("what is our tuition reimbursement")
    assert "france" in index.unknown_terms("what is the capital of France")
    # Morphological variants of corpus words are not "unknown" — a warning that
    # fires on ordinary phrasing gets ignored and is then worth nothing.
    assert index.unknown_terms("what are the grievance timelines") == []
    assert index.unknown_terms("how do I carry over leave") == []
    assert index.unknown_terms("summarize our leave policy") == []
    assert index.unknown_terms("which cities count as tier-1 for hotels") == []
    # "relocate" appears in the remote-work policy, so "relocation" is known
    # vocabulary even though no relocation policy exists — the coverage gate,
    # not the vocabulary check, is what makes that question abstain.
    assert index.unknown_terms("relocation") == []


def test_mmr_diversifies_across_sections(index):
    hits = index.search("leave", top_k=5, min_score=0.0)
    sections = [hit.chunk.section for hit in hits]
    assert len(set(sections)) == len(sections), "MMR should not return duplicate sections"


def test_bm25_coverage_penalises_unseen_terms():
    corpus = [tokenize("annual leave entitlement is 24 days"), tokenize("sick leave needs a note")]
    bm25 = BM25Index(corpus)
    known = bm25.coverage(["annual", "leave"])
    unknown = bm25.coverage(["relocation", "leave"])
    # An out-of-vocabulary term must drag coverage down, not be ignored.
    assert known[0] > unknown[0]
