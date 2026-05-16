"""Simple in-memory session manager for local raw ASL inference."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.runtime_state import RuntimeSessionState
from app.ml.runtime_config import INPUT_DIM, SEQUENCE_LENGTH


@dataclass
class SessionManager:
    """Store runtime session buffers by session id."""

    default_session_id: str = "default"
    sequence_length: int = SEQUENCE_LENGTH
    input_dim: int = INPUT_DIM
    _sessions: dict[str, RuntimeSessionState] = field(default_factory=dict, init=False)

    def get_session(self, session_id: str | None = None) -> RuntimeSessionState:
        resolved_session_id = session_id or self.default_session_id
        if resolved_session_id not in self._sessions:
            self._sessions[resolved_session_id] = RuntimeSessionState(
                session_id=resolved_session_id,
                sequence_length=self.sequence_length,
                input_dim=self.input_dim,
            )
        return self._sessions[resolved_session_id]

    def reset_session(self, session_id: str | None = None) -> RuntimeSessionState:
        session = self.get_session(session_id)
        session.reset()
        return session

    def get_session_metadata(self, session_id: str | None = None) -> dict[str, int | str]:
        session = self.get_session(session_id)
        return {
            "session_id": session.session_id,
            "frames_collected": session.valid_frames_collected,
            "sequence_length": session.sequence_length,
            "total_frames_received": session.total_frames_received,
            "last_status": session.last_status,
        }
