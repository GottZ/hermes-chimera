"""Fake ``hermes_state.SessionDB`` (hermes_state.py:584, hermes_state_compression.py:916-940): only
``get_compression_lineage``. ``SESSIONS`` maps a session id to ``(parent_session_id, end_reason)``; a session
continues its parent when the parent ended with ``"compression"``. ``OPENED`` counts the stores opened (the plugin
reads the store only when a job does not match the caller directly)."""

from __future__ import annotations

SESSIONS: dict[str, tuple[str | None, str | None]] = {}
OPENED: list[bool] = []


class SessionDB:
    def __init__(self, db_path=None, read_only: bool = False) -> None:
        OPENED.append(bool(read_only))
        self.closed = False

    def _continues(self, sid: str) -> bool:
        parent = SESSIONS.get(sid, (None, None))[0]
        return parent is not None and SESSIONS.get(parent, (None, None))[1] == "compression"

    def get_compression_lineage(self, session_id: str) -> list[str]:
        if session_id not in SESSIONS:
            return []
        root = session_id
        while self._continues(root):
            root = SESSIONS[root][0]
        chain = [root]
        while SESSIONS.get(chain[-1], (None, None))[1] == "compression":
            child = next((s for s, (p, _) in SESSIONS.items() if p == chain[-1] and self._continues(s)), None)
            if child is None or child in chain:
                break
            chain.append(child)
        return chain

    def close(self) -> None:
        self.closed = True
