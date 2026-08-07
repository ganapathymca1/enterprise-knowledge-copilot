"""In-process conversation store.

Deliberately simple, and deliberately behind an interface. A POC does not need
Redis, but the shape here (async methods, TTL, bounded history, explicit
eviction) is the shape a Redis or Postgres implementation would have, so
swapping it is a single class change rather than a refactor. That trade-off is
recorded in docs/ARCHITECTURE.md under scale-out.

Two properties matter for correctness, not just tidiness:

* **Bounded history.** Only the last N turns are kept for prompting, so prompt
  size — and therefore latency and cost — stays constant no matter how long the
  conversation runs.
* **TTL eviction.** Sessions expire, so a long-running process does not grow
  without limit, and stale personal context does not linger in memory.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..models import ChatMessage, Role, SessionSummary


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    session_id: str
    employee_id: str
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    messages: list[ChatMessage] = field(default_factory=list)

    @property
    def title(self) -> str:
        for message in self.messages:
            if message.role == Role.user:
                text = message.content.strip().splitlines()[0]
                return text[:60] + ("…" if len(text) > 60 else "")
        return "New conversation"

    def summary(self) -> SessionSummary:
        return SessionSummary(
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            turns=sum(1 for m in self.messages if m.role == Role.user),
            title=self.title,
        )


class ConversationStore:
    def __init__(self, *, ttl_minutes: int = 120, max_messages: int = 60) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_messages = max_messages
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str | None, employee_id: str) -> Session:
        async with self._lock:
            self._evict_expired()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.updated_at = _now()
                return session
            new_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
            session = Session(session_id=new_id, employee_id=employee_id)
            self._sessions[new_id] = session
            return session

    async def append(self, session_id: str, role: Role, content: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.messages.append(ChatMessage(role=role, content=content))
            if len(session.messages) > self.max_messages:
                del session.messages[: len(session.messages) - self.max_messages]
            session.updated_at = _now()

    async def history(self, session_id: str, turns: int) -> list[ChatMessage]:
        """Return the last ``turns`` exchanges (a user + assistant pair each)."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return list(session.messages[-(turns * 2) :])

    async def list_sessions(self) -> list[SessionSummary]:
        async with self._lock:
            self._evict_expired()
            sessions = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
            return [session.summary() for session in sessions]

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict_expired(self) -> None:
        cutoff = _now() - self.ttl
        for session_id in [sid for sid, s in self._sessions.items() if s.updated_at < cutoff]:
            del self._sessions[session_id]
