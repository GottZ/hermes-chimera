"""Which Hermes sessions a render job belongs to, and stopping a session's jobs when the session ends.

A job is bound to the ids of the tool call that started it: ``session_id`` (every supported Hermes passes it) and
``conversation_id`` (passed by newer Hermes; it stays the same through context compression). Context compression
continues a conversation in a new session whose parent is the compressed one; Hermes' session store returns that
chain (``SessionDB.get_compression_lineage``: compression only - ``/new``, ``/branch`` and delegated tasks start
their own chain). A call whose ids match none of a job's ids is matched once more with the ids of its compression
chain, so a job started before a compression stays reachable afterwards and every other session still gets
"unknown job".

Session end: Hermes calls ``on_session_finalize`` with the ending session (CLI exit, ``/new``, gateway shutdown,
desktop session close) and ``on_session_reset`` with ``old_session_id`` after ``/new``; both stop the unfinished jobs
of that session and of its compression chain. ``on_session_end`` is not used: it fires after every turn.
"""

from __future__ import annotations

import logging
from typing import Any

from . import _vocal

logger = logging.getLogger(__name__)


def owner_ids(kwargs: dict) -> frozenset | None:
    """The ids a tool call is known by (``conversation_id:<id>``, ``session_id:<id>``); None outside a session."""
    ids = frozenset(f"{key}:{v}" for key in ("conversation_id", "session_id")
                    if isinstance(v := kwargs.get(key), str) and v)
    return ids or None


def compression_lineage(session_id: Any) -> list[str]:
    """Session ids of the compression chain through ``session_id``, oldest first, from the session store of the
    active Hermes profile; [] when the session or the store is not available."""
    if not isinstance(session_id, str) or not session_id:
        return []
    try:
        from hermes_state import SessionDB
    except Exception:  # noqa: BLE001 - not inside Hermes
        return []
    db = None
    try:
        db = SessionDB(read_only=True)
        walk = getattr(db, "get_compression_lineage", None)
        chain = walk(session_id) if callable(walk) else []
        return [s for s in chain or [] if isinstance(s, str) and s]
    except Exception as e:  # noqa: BLE001 - an unreadable store only means: no compression chain known
        logger.debug("chimera: compression chain of %s unavailable: %s", session_id, e)
        return []
    finally:
        close = getattr(db, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


def related_ids(session_id: Any) -> frozenset:
    """Ids a job started anywhere in the compression chain of ``session_id`` may carry: each session of the chain
    and the conversation id newer Hermes derives from the chain's first session."""
    chain = compression_lineage(session_id)
    ids = {f"session_id:{s}" for s in chain}
    if chain:
        ids.add(f"conversation_id:{chain[0]}")
    return frozenset(ids)


def related_lookup(kwargs: dict) -> Any:
    """For JobRegistry.wait/cancel: computes ``related_ids`` of the caller's session only when a job does not match
    the caller's own ids (it reads Hermes' session store), at most once per tool call."""
    session_id = kwargs.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    found: list = []

    def lookup() -> frozenset:
        if not found:
            found.append(related_ids(session_id))
        return found[0]

    return lookup


def cancel_session(session_id: Any) -> list[str]:
    """Stop the unfinished jobs of ``session_id`` and of its compression chain; returns their ids."""
    if not isinstance(session_id, str) or not session_id:
        return []
    reg = _vocal.jobs_if_any()
    if reg is None or not reg.has_unfinished():
        return []
    return reg.cancel_owned({f"session_id:{session_id}"} | related_ids(session_id))


def on_session_finalize(session_id: Any = None, **_: Any) -> None:
    """Hermes hook: the session ends."""
    try:
        stopped = cancel_session(session_id)
        if stopped:
            logger.info("chimera: session %s ended; stopped %d render(s)", session_id, len(stopped))
    except Exception as e:  # noqa: BLE001 - a hook never breaks Hermes' session handling
        logger.warning("chimera: stopping the renders of session %s failed: %s", session_id, e)


def on_session_reset(old_session_id: Any = None, **_: Any) -> None:
    """Hermes hook after ``/new``: ``old_session_id`` ended. ``session_id`` names the session that starts now (and
    in the desktop backend a session that is only being rebuilt), so it is left alone."""
    on_session_finalize(old_session_id)
