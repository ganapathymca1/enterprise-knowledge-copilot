"""Knowledge base loading and section-aware chunking.

Design notes
------------
*Why not a fixed 512-token sliding window?* Handbook content is highly
structured: a heading plus a table is a single semantic unit, and cutting a
table in half produces chunks that retrieve well but read terribly once they
reach the model. So chunking is **heading-aware first, size-bounded second**:

1. Split the document on markdown headings (H2/H3) to get semantic sections.
2. If a section is longer than ``chunk_tokens``, split it further on paragraph
   boundaries with an overlap, never inside a markdown table.
3. Prefix every chunk with its document title and heading path. That single
   trick measurably improves lexical retrieval, because the words "leave
   policy" appear in the chunk even when the paragraph itself says only "the
   entitlement is 24 days".

Front matter is parsed with a tiny scalar YAML reader so the project does not
need a YAML dependency for six well-known keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    category: str
    owner: str
    version: str
    effective_date: str
    review_date: str
    audience: str
    path: Path
    body: str

    @property
    def source_path(self) -> str:
        return self.path.name


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    category: str = ""
    owner: str = ""
    version: str = ""
    effective_date: str = ""
    source_path: str = ""
    ordinal: int = 0
    keywords: list[str] = field(default_factory=list)

    @property
    def header(self) -> str:
        return f"{self.title} — {self.section}" if self.section else self.title

    @property
    def embedding_text(self) -> str:
        """Text handed to the retriever — heading path included on purpose."""
        return f"{self.header}\n{self.text}"

    def lexical_tokens_text(self, header_boost: int = 2) -> str:
        """Header text repeated ``header_boost`` times = cheap BM25F field boost.

        Without it, a question like "escalation process" ranks the *Severity
        levels* section above the *Escalation matrix* section, because the word
        appears more often in the prose of the former.
        """
        return "\n".join([self.header] * max(1, header_boost) + [self.text])

    def citation_snippet(self, limit: int = 420) -> str:
        flat = re.sub(r"\s+", " ", self.text).strip()
        if len(flat) <= limit:
            return flat
        return flat[: limit - 1].rsplit(" ", 1)[0] + "…"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end():]


def _approx_tokens(text: str) -> int:
    """Cheap token estimate: words × 1.3. Avoids a tokenizer dependency."""
    return int(len(text.split()) * 1.3) + 1


def _split_blocks(section_text: str) -> list[str]:
    """Split a section into blocks, keeping markdown tables in one piece."""
    blocks: list[str] = []
    buffer: list[str] = []
    in_table = False
    for line in section_text.splitlines():
        is_table_line = bool(TABLE_ROW_RE.match(line))
        if is_table_line and not in_table:
            if buffer and not buffer[-1].strip():
                blocks.append("\n".join(buffer).strip())
                buffer = []
            in_table = True
        elif in_table and not is_table_line and line.strip():
            in_table = False
        if not line.strip() and not in_table and buffer:
            blocks.append("\n".join(buffer).strip())
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        tail = "\n".join(buffer).strip()
        if tail:
            blocks.append(tail)
    return [b for b in blocks if b]


def _pack_blocks(blocks: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack blocks into chunks with a trailing-block overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = _approx_tokens(block)
        if current and current_tokens + block_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            overlap: list[str] = []
            budget = overlap_tokens
            for previous in reversed(current):
                cost = _approx_tokens(previous)
                if cost > budget:
                    break
                overlap.insert(0, previous)
                budget -= cost
            current = overlap
            current_tokens = sum(_approx_tokens(b) for b in current)
        current.append(block)
        current_tokens += block_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_documents(kb_dir: Path) -> list[Document]:
    """Read every markdown file in ``kb_dir``, sorted for deterministic ids."""
    documents: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        documents.append(
            Document(
                doc_id=meta.get("doc_id") or path.stem.upper(),
                title=meta.get("title") or path.stem.replace("-", " ").title(),
                category=meta.get("category", ""),
                owner=meta.get("owner", ""),
                version=meta.get("version", ""),
                effective_date=meta.get("effective_date", ""),
                review_date=meta.get("review_date", ""),
                audience=meta.get("audience", ""),
                path=path,
                body=body,
            )
        )
    return documents


def _merge_undersized(pieces: list[tuple[str, str]], min_tokens: int) -> list[tuple[str, str]]:
    """Fold tiny sections (e.g. a two-line "Contact" heading) into their
    neighbour. Sub-30-token chunks are retrieval noise: they win on BM25 length
    normalisation while carrying almost no answer content."""
    merged: list[tuple[str, str]] = []
    for section, text in pieces:
        if merged and _approx_tokens(text) < min_tokens:
            prev_section, prev_text = merged[-1]
            merged[-1] = (prev_section, f"{prev_text}\n\n### {section}\n{text}" if section else f"{prev_text}\n\n{text}")
            continue
        merged.append((section, text))
    if len(merged) > 1 and _approx_tokens(merged[0][1]) < min_tokens:
        section, text = merged.pop(0)
        next_section, next_text = merged[0]
        merged[0] = (next_section, f"{text}\n\n{next_text}")
    return merged


def chunk_document(
    document: Document,
    *,
    chunk_tokens: int = 220,
    overlap_tokens: int = 45,
    min_chunk_tokens: int = 30,
) -> list[Chunk]:
    """Split one document into heading-aware, size-bounded chunks."""
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[str] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in current_lines):
            sections.append((current_heading, list(current_lines)))

    for line in document.body.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            current_lines = []
            level = len(heading.group(1))
            text = heading.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(text)
            current_heading = " › ".join(part for part in heading_stack[1:] if part) or text
            continue
        current_lines.append(line)
    flush()

    pieces: list[tuple[str, str]] = []
    for section_name, lines in sections:
        blocks = _split_blocks("\n".join(lines))
        if not blocks:
            continue
        for piece in _pack_blocks(blocks, chunk_tokens, overlap_tokens):
            pieces.append((section_name, piece.strip()))

    chunks: list[Chunk] = []
    ordinal = 0
    for section_name, piece in _merge_undersized(pieces, min_chunk_tokens):
        ordinal += 1
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{ordinal:03d}",
                doc_id=document.doc_id,
                title=document.title,
                section=section_name,
                text=piece.strip(),
                category=document.category,
                owner=document.owner,
                version=document.version,
                effective_date=document.effective_date,
                source_path=document.source_path,
                ordinal=ordinal,
            )
        )
    return chunks


def load_chunks(
    kb_dir: Path,
    *,
    chunk_tokens: int = 220,
    overlap_tokens: int = 45,
) -> tuple[list[Document], list[Chunk]]:
    documents = load_documents(kb_dir)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return documents, chunks
