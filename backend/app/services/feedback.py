"""Feedback capture and the answer audit trail (SQLite, stdlib only).

Why both tables
---------------
A thumbs-down on its own is unactionable — you learn that *an* answer was wrong,
not which passage misled the model. So every answer writes an ``answers`` row
first (question, retrieved chunk ids, scores, provider, latency, grounding
score, tools used), and feedback joins to it by ``trace_id``.

That join is what makes the governance story concrete:

* **Routing.** Each retrieved chunk carries the owning team from its front
  matter, so a "this is outdated" report is routed to the policy owner rather
  than to a generic inbox.
* **Prioritisation.** ``chunk_failure_report`` ranks the passages that appear
  most often in downvoted answers — the highest-value candidates for rewriting.
* **Regression testing.** Downvoted questions are exported straight into the
  evaluation set, so a fix can be proven and cannot silently regress.

Everything written here is PII-redacted first. The database file lives under
``var/`` and is gitignored.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    trace_id       TEXT PRIMARY KEY,
    session_id     TEXT,
    created_at     TEXT NOT NULL,
    employee_id    TEXT,
    question       TEXT NOT NULL,
    rewritten      TEXT,
    answer         TEXT NOT NULL,
    answer_type    TEXT NOT NULL,
    confidence     TEXT NOT NULL,
    grounding      REAL NOT NULL,
    provider       TEXT,
    model          TEXT,
    latency_ms     INTEGER,
    chunk_ids      TEXT,
    doc_ids        TEXT,
    owners         TEXT,
    top_score      REAL,
    tools          TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id  TEXT PRIMARY KEY,
    trace_id     TEXT NOT NULL,
    session_id   TEXT,
    created_at   TEXT NOT NULL,
    rating       TEXT NOT NULL,
    reason       TEXT,
    comment      TEXT,
    routed_to    TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id);
CREATE INDEX IF NOT EXISTS idx_answers_created ON answers(created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    """Thread-safe SQLite store for answers and their feedback."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    # -- writes -----------------------------------------------------------
    def record_answer(self, record: dict[str, Any]) -> None:
        payload = {
            "trace_id": record["trace_id"],
            "session_id": record.get("session_id"),
            "created_at": _now(),
            "employee_id": record.get("employee_id"),
            "question": record.get("question", ""),
            "rewritten": record.get("rewritten"),
            "answer": record.get("answer", ""),
            "answer_type": record.get("answer_type", ""),
            "confidence": record.get("confidence", ""),
            "grounding": float(record.get("grounding", 0.0)),
            "provider": record.get("provider"),
            "model": record.get("model"),
            "latency_ms": int(record.get("latency_ms", 0)),
            "chunk_ids": json.dumps(record.get("chunk_ids", [])),
            "doc_ids": json.dumps(record.get("doc_ids", [])),
            "owners": json.dumps(record.get("owners", [])),
            "top_score": float(record.get("top_score", 0.0)),
            "tools": json.dumps(record.get("tools", [])),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{key}" for key in payload)
        with self._lock:
            self._connection.execute(
                f"INSERT OR REPLACE INTO answers ({columns}) VALUES ({placeholders})", payload
            )
            self._connection.commit()

    def record_feedback(
        self,
        *,
        trace_id: str,
        session_id: str | None,
        rating: str,
        reason: str,
        comment: str,
    ) -> tuple[str, str]:
        """Store feedback and return (feedback_id, routed_to)."""
        routed_to = self._owner_for_trace(trace_id)
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._connection.execute(
                "INSERT INTO feedback (feedback_id, trace_id, session_id, created_at, "
                "rating, reason, comment, routed_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (feedback_id, trace_id, session_id, _now(), rating, reason, comment, routed_to),
            )
            self._connection.commit()
        return feedback_id, routed_to

    # -- reads ------------------------------------------------------------
    def _owner_for_trace(self, trace_id: str) -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT owners FROM answers WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        if not row or not row["owners"]:
            return "People Operations"
        owners = json.loads(row["owners"])
        return owners[0] if owners else "People Operations"

    def stats(self) -> dict[str, Any]:
        with self._lock:
            answers = self._connection.execute(
                "SELECT COUNT(*) AS n, AVG(grounding) AS grounding, AVG(latency_ms) AS latency "
                "FROM answers"
            ).fetchone()
            by_type = self._connection.execute(
                "SELECT answer_type, COUNT(*) AS n FROM answers GROUP BY answer_type"
            ).fetchall()
            votes = self._connection.execute(
                "SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating"
            ).fetchall()
            reasons = self._connection.execute(
                "SELECT reason, COUNT(*) AS n FROM feedback WHERE rating = 'down' "
                "GROUP BY reason ORDER BY n DESC"
            ).fetchall()
        up = next((row["n"] for row in votes if row["rating"] == "up"), 0)
        down = next((row["n"] for row in votes if row["rating"] == "down"), 0)
        return {
            "answers": answers["n"] or 0,
            "avg_grounding": round(answers["grounding"] or 0.0, 3),
            "avg_latency_ms": int(answers["latency"] or 0),
            "answers_by_type": {row["answer_type"]: row["n"] for row in by_type},
            "feedback": {"up": up, "down": down, "total": up + down},
            "satisfaction": round(up / (up + down), 3) if (up + down) else None,
            "downvote_reasons": {row["reason"]: row["n"] for row in reasons},
        }

    def chunk_failure_report(self, limit: int = 10) -> list[dict[str, Any]]:
        """Passages most often present in a downvoted answer — the rewrite queue."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT a.chunk_ids, a.doc_ids FROM feedback f "
                "JOIN answers a ON a.trace_id = f.trace_id WHERE f.rating = 'down'"
            ).fetchall()
        tally: dict[str, int] = {}
        docs: dict[str, str] = {}
        for row in rows:
            chunk_ids = json.loads(row["chunk_ids"] or "[]")
            doc_ids = json.loads(row["doc_ids"] or "[]")
            for position, chunk_id in enumerate(chunk_ids):
                tally[chunk_id] = tally.get(chunk_id, 0) + 1
                if position < len(doc_ids):
                    docs[chunk_id] = doc_ids[position]
        ranked = sorted(tally.items(), key=lambda item: -item[1])[:limit]
        return [
            {"chunk_id": chunk_id, "doc_id": docs.get(chunk_id, ""), "downvotes": count}
            for chunk_id, count in ranked
        ]

    def export_downvoted_questions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Downvoted turns, formatted for pasting into the evaluation set."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT a.question, a.answer_type, a.doc_ids, f.reason, f.comment, f.created_at "
                "FROM feedback f JOIN answers a ON a.trace_id = f.trace_id "
                "WHERE f.rating = 'down' ORDER BY f.created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "question": row["question"],
                "answer_type": row["answer_type"],
                "retrieved_docs": json.loads(row["doc_ids"] or "[]"),
                "reason": row["reason"],
                "comment": row["comment"],
                "reported_at": row["created_at"],
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
