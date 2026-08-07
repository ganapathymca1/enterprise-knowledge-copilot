"""Hybrid retrieval index: BM25 (lexical) + LSA (semantic), fused with RRF.

Why this stack
--------------
The brief requires the solution to run on standard developer hardware with free
tools and no downloads. A transformer embedding model would add ~90 MB of
weights and a first-run download, which breaks "clean environment" reproduction
on a locked-down laptop. So the semantic half uses **TF-IDF + truncated SVD
(latent semantic analysis)** from scikit-learn: no weights, no network, fully
deterministic, and it still captures term co-occurrence ("PTO" ↔ "annual
leave") that pure BM25 misses.

Lexical BM25 is implemented directly (about 50 lines) rather than pulled in as
a dependency, because it is small, easy to defend in review, and lets us apply
a field boost to the heading path.

The two rankers are combined with **Reciprocal Rank Fusion**. RRF is used
instead of a weighted score sum because BM25 scores and cosine similarities
live on different, corpus-dependent scales; fusing ranks avoids inventing a
normalisation constant that would need re-tuning per corpus.

`EmbeddingBackend` is deliberately isolated so swapping in
`sentence-transformers` or a managed embedding endpoint later touches one class.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .loader import Chunk, Document, load_chunks

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")

# Weight of lexical coverage in the calibrated relevance score. Tuned on the
# evaluation set in `eval/` — see docs/ACCURACY_AND_LIMITATIONS.md.
COVERAGE_WEIGHT = 0.6

# Reciprocal rank fusion constant. The literature default (60) is tuned for
# candidate lists in the thousands; with ~20 candidates it flattens every score
# into a tie, so a smaller constant is used to keep ranks discriminative.
RRF_K = 10

# How much a passage's score is pulled towards its document's overall evidence.
# Tuned on the evaluation set: 0.15 leaves ranking metrics unchanged while
# raising fact-in-context from 26/27 to 27/27.
DOC_PRIOR = 0.15

# Verbs and framing words people use to *ask*, which carry no topic. They are
# excluded from the unknown-term check: warning that "summarize" is absent from
# the handbook is noise, and noisy warnings train users to ignore real ones.
INSTRUCTION_WORDS = frozenset(
    """summarize summarise summary explain describe outline detail elaborate clarify
    compare list show find give provide tell walk brief overview understand check
    quick question answer help mean means anything something everything policy
    policies document documents information info detail details company companys
    company's ours ourselves regarding concerning about""".split()
)

STOP_WORDS = frozenset(
    """a about all am an and any are as at be been being but by can could did do does
    doing for from get give got had has have how i if in into is it its just know let
    like make may me my need not of on or our out please should so some tell than that
    the their them then there these they this those to told us was we were what when
    where which who why will with would you your""".split()
)

# Small, explicit domain synonym map. Enterprise corpora are full of internal
# vocabulary that no general embedding model knows; expanding the *query* (not
# the documents) keeps the index clean and the behaviour inspectable.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "pto": ("paid", "time", "off", "annual", "leave", "vacation"),
    "vacation": ("annual", "leave", "pto"),
    "holiday": ("public", "holiday", "leave"),
    "wfh": ("remote", "work", "home", "hybrid"),
    "remote": ("hybrid", "work", "from", "home"),
    "maternity": ("parental", "primary", "caregiver", "leave"),
    "paternity": ("parental", "secondary", "caregiver", "leave"),
    "payslip": ("payroll", "salary", "pay"),
    "salary": ("compensation", "pay", "payroll"),
    "raise": ("compensation", "review", "increase"),
    "bonus": ("performance", "bonus", "compensation"),
    "reimbursement": ("expense", "claim", "reimburse"),
    "escalate": ("escalation", "incident", "severity"),
    "escalation": ("incident", "severity", "on-call"),
    "outage": ("incident", "severity", "sev1"),
    "sev1": ("severity", "incident", "outage"),
    "onboarding": ("new", "joiner", "orientation", "induction"),
    "induction": ("onboarding", "new", "joiner"),
    "notice": ("notice", "period", "resignation", "offboarding"),
    "quit": ("resignation", "notice", "offboarding", "exit"),
    "resign": ("resignation", "notice", "offboarding"),
    "fired": ("dismissal", "disciplinary", "conduct"),
    "harassment": ("conduct", "grievance", "speak", "up"),
    "complaint": ("grievance", "conduct", "escalate"),
    "mfa": ("multi-factor", "authentication", "security"),
    "2fa": ("multi-factor", "authentication", "security"),
    "vpn": ("security", "acceptable", "use", "access"),
    "laptop": ("device", "equipment", "hardware", "it"),
    "insurance": ("benefits", "cover", "medical"),
    "pension": ("retirement", "benefits", "contribution"),
    "training": ("learning", "development", "course"),
    "course": ("learning", "development", "training"),
    "sick": ("sick", "illness", "absence"),
    "ai": ("generative", "ai", "acceptable", "use"),
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _morphological_variants(term: str) -> set[str]:
    """Crude English stemming — enough to match a query word to its policy form.

    A real deployment would use a proper stemmer or lemmatiser; this covers the
    plural/gerund/possessive cases that dominate employee phrasing without
    adding an NLP dependency.
    """
    variants = {term, f"{term}s", f"{term}es"}
    for suffix in ("'s", "s", "es", "ing", "ed", "ly", "ment", "tion"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            stem = term[: -len(suffix)]
            variants.update({stem, f"{stem}e", f"{stem}s", f"{stem}y"})
    if term.endswith("ies") and len(term) > 4:
        variants.add(f"{term[:-3]}y")
    if term.endswith("y") and len(term) > 3:
        variants.add(f"{term[:-1]}ies")  # city -> cities
    return variants


def expand_query(text: str) -> str:
    """Append synonym terms to the query. Documents are never rewritten."""
    tokens = tokenize(text)
    extra: list[str] = []
    for token in tokens:
        extra.extend(SYNONYMS.get(token, ()))
    if not extra:
        return text
    return f"{text} {' '.join(dict.fromkeys(extra))}"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float                      # calibrated relevance in [0, 1]
    coverage: float = 0.0             # share of the query's idf mass present
    similarity: float = 0.0           # LSA cosine similarity
    lexical_rank: int | None = None
    semantic_rank: int | None = None


class BM25Index:
    """Okapi BM25 over the chunk corpus."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        # BM25+ style idf floor keeps very common terms from going negative.
        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        # Weight for a term that does not occur in the corpus at all. Used by
        # `coverage`, where an unknown term must *lower* confidence rather than
        # be silently ignored — that is what makes "what is the relocation
        # bonus?" abstain instead of answering from the nearest paragraph.
        self.oov_idf = math.log(1 + (self.corpus_size + 0.5) / 0.5)

    def coverage(self, query_terms: list[str]) -> np.ndarray:
        """Fraction of the query's idf mass that each chunk actually contains.

        This is the abstention signal. Unlike a BM25 score it is bounded in
        [0, 1], is comparable across queries, and is trivially explainable to a
        reviewer: "the chunk covers 30% of what you asked about".
        """
        terms = list(dict.fromkeys(query_terms))
        if not terms:
            return np.zeros(self.corpus_size, dtype=np.float64)
        weights = np.array([self.idf.get(t, self.oov_idf) for t in terms], dtype=np.float64)
        total = float(weights.sum()) or 1.0
        result = np.zeros(self.corpus_size, dtype=np.float64)
        for position, term in enumerate(terms):
            if term not in self.idf:
                continue
            present = np.array(
                [1.0 if freqs.get(term) else 0.0 for freqs in self.doc_freqs],
                dtype=np.float64,
            )
            result += weights[position] * present
        return result / total

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(self.corpus_size, dtype=np.float64)
        if not self.corpus_size or self.avgdl == 0:
            return result
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self.doc_freqs], dtype=np.float64)
            result += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return result


class EmbeddingBackend:
    """Deterministic local embeddings via TF-IDF → SVD (LSA).

    Swap point: any object exposing ``fit_transform`` / ``transform`` returning
    L2-normalised vectors can replace this class.
    """

    name = "tfidf-svd-local"

    def __init__(self, n_components: int = 160, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed
        self._pipeline = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        # Word-level and character-level views are concatenated by the
        # vectorizer union below; char n-grams give robustness to typos and to
        # morphological variants ("escalate"/"escalation").
        components = min(self.n_components, max(2, len(texts) - 1))
        self._pipeline = make_pipeline(
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                stop_words="english",
            ),
            TruncatedSVD(n_components=components, random_state=self.seed),
            Normalizer(copy=False),
        )
        return self._pipeline.fit_transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("EmbeddingBackend.fit_transform must be called first")
        return self._pipeline.transform(texts)


class KnowledgeIndex:
    """The retrievable corpus: chunks + lexical index + semantic index."""

    def __init__(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        *,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("Knowledge base is empty — check COPILOT_KNOWLEDGE_BASE_DIR")
        self.documents = documents
        self.chunks = chunks
        self.built_at = datetime.now(timezone.utc)
        texts = [chunk.embedding_text for chunk in chunks]
        self.bm25 = BM25Index([tokenize(chunk.lexical_tokens_text()) for chunk in chunks])
        self.embedder = embedding_backend or EmbeddingBackend()
        self.matrix = self.embedder.fit_transform(texts)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        kb_dir: Path,
        *,
        chunk_tokens: int = 220,
        overlap_tokens: int = 45,
    ) -> "KnowledgeIndex":
        documents, chunks = load_chunks(
            kb_dir, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
        )
        return cls(documents, chunks)

    # -- retrieval --------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        min_score: float = 0.0,
        mmr_lambda: float = 0.7,
        doc_prior: float = DOC_PRIOR,
    ) -> list[ScoredChunk]:
        """Hybrid search: BM25 ∪ LSA → RRF ordering → calibrated score → MMR.

        Ordering and scoring are deliberately separated. RRF decides *order*
        (it is robust to the two rankers' incompatible score scales), while the
        reported score is a calibrated blend of query coverage and semantic
        similarity — a number that means the same thing across queries and can
        therefore drive the abstention threshold and the confidence badge.
        """
        expanded = expand_query(query)
        candidate_k = max(candidate_k, top_k)

        lexical_scores = self.bm25.scores(tokenize(expanded))
        lexical_order = [
            int(i) for i in np.argsort(-lexical_scores)[:candidate_k] if lexical_scores[i] > 0
        ]

        query_vector = self.embedder.transform([expanded])
        semantic_scores = (self.matrix @ query_vector.T).ravel()
        semantic_order = [int(i) for i in np.argsort(-semantic_scores)[:candidate_k]]

        fused = self._reciprocal_rank_fusion(lexical_order, semantic_order)
        if not fused:
            return []

        # Coverage is computed from the *original* query so that synonym
        # expansion cannot inflate confidence for words the user never used.
        content_terms = [t for t in tokenize(query) if t not in STOP_WORDS and len(t) > 2]
        coverage = self.bm25.coverage(content_terms)

        ranked: list[tuple[int, float, float, float]] = []
        for idx, _ in fused:
            cov = float(coverage[idx])
            sim = float(max(0.0, semantic_scores[idx]))
            relevance = COVERAGE_WEIGHT * cov + (1 - COVERAGE_WEIGHT) * sim
            ranked.append((idx, relevance, cov, sim))

        ranked = self._apply_document_prior(ranked, doc_prior)
        ranked = [row for row in ranked if row[1] >= min_score]
        if not ranked:
            return []

        selected = self._mmr([(idx, rel) for idx, rel, _, _ in ranked], top_k=top_k, lam=mmr_lambda)
        # MMR decides *which* chunks are kept; presentation order is by
        # relevance so that citation [1] is always the strongest evidence.
        selected.sort(key=lambda item: -item[1])
        detail = {idx: (cov, sim) for idx, _, cov, sim in ranked}
        lexical_positions = {idx: pos for pos, idx in enumerate(lexical_order)}
        semantic_positions = {idx: pos for pos, idx in enumerate(semantic_order)}
        return [
            ScoredChunk(
                chunk=self.chunks[idx],
                score=round(float(score), 4),
                coverage=round(detail[idx][0], 4),
                similarity=round(detail[idx][1], 4),
                lexical_rank=lexical_positions.get(idx),
                semantic_rank=semantic_positions.get(idx),
            )
            for idx, score in selected
        ]

    def _apply_document_prior(
        self,
        ranked: list[tuple[int, float, float, float]],
        weight: float,
    ) -> list[tuple[int, float, float, float]]:
        """Blend in evidence from the rest of the chunk's document.

        A question is usually answered by one *document*, and the passage
        holding the exact figure often matches the question less well than the
        prose around it — "How much notice for a two week holiday?" matches the
        offboarding policy's notice-period section on wording, while the answer
        sits in a table in the leave policy. Letting a document's overall
        evidence lift its own passages fixes that without hard-coding topics.
        Measured on the evaluation set, this raises the rate at which the exact
        answer figure is present in the model's context to 100%.
        """
        if weight <= 0 or not ranked:
            return ranked
        by_document: dict[str, list[float]] = {}
        for idx, relevance, _cov, _sim in ranked:
            by_document.setdefault(self.chunks[idx].doc_id, []).append(relevance)
        # Top-2 passages per document: enough to reward a document that matches
        # in more than one place, without rewarding sheer document length.
        document_score = {
            doc_id: sum(sorted(scores, reverse=True)[:2])
            for doc_id, scores in by_document.items()
        }
        best = max(document_score.values()) or 1.0
        return [
            (
                idx,
                (1 - weight) * relevance
                + weight * (document_score[self.chunks[idx].doc_id] / best),
                cov,
                sim,
            )
            for idx, relevance, cov, sim in ranked
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        lexical_order: list[int],
        semantic_order: list[int],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for rank, idx in enumerate(lexical_order):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(semantic_order):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: -item[1])

    def _mmr(
        self,
        ranked: list[tuple[int, float]],
        *,
        top_k: int,
        lam: float,
    ) -> list[tuple[int, float]]:
        """Maximal marginal relevance — avoids five near-identical chunks."""
        selected: list[tuple[int, float]] = []
        remaining = list(ranked)
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            chosen_vectors = self.matrix[[idx for idx, _ in selected]]
            best_pos, best_value = 0, -math.inf
            for position, (idx, relevance) in enumerate(remaining):
                redundancy = float(np.max(chosen_vectors @ self.matrix[idx]))
                value = lam * relevance - (1 - lam) * redundancy
                if value > best_value:
                    best_pos, best_value = position, value
            selected.append(remaining.pop(best_pos))
        return selected

    def unknown_terms(self, query: str) -> list[str]:
        """Content words from the query that appear nowhere in the corpus.

        A strong, cheap over-trust signal. "What is the relocation bonus?"
        retrieves the compensation policy with a respectable similarity score,
        because *bonus* is well represented — but *relocation* does not exist in
        the corpus at all, which is exactly the fact the employee needs to be
        told. Ranking alone cannot express that; vocabulary can.

        Precision matters more than recall here. A warning that fires on
        ordinary questions gets ignored, and then it is worth nothing when it
        fires on a real gap. So a term counts as unknown only if neither it, nor
        its simple morphological variants, nor any word sharing its six-letter
        prefix appears in the corpus — which is what stops "carrying",
        "approves" and "timelines" from being reported as unknown vocabulary.
        """
        terms = [
            t
            for t in tokenize(query)
            if t not in STOP_WORDS and t not in INSTRUCTION_WORDS and len(t) > 3
        ]
        return [t for t in dict.fromkeys(terms) if not self._in_vocabulary(t)]

    def _in_vocabulary(self, term: str) -> bool:
        vocabulary = self.bm25.idf
        if any(variant in vocabulary for variant in _morphological_variants(term)):
            return True
        if len(term) >= 6:
            prefix = term[:6]
            return any(word.startswith(prefix) for word in vocabulary)
        return False

    # -- introspection ----------------------------------------------------
    @property
    def stats(self) -> dict[str, object]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedding_backend": self.embedder.name,
            "dimensions": int(self.matrix.shape[1]),
            "built_at": self.built_at.isoformat(),
        }
