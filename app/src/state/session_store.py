from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from PIL import Image


@dataclass
class SessionState:
    session_id: str
    image_bytes: bytes
    history: List[Tuple[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionStore:
    """In-memory helper that keeps track of conversational VQA context."""

    def __init__(self, max_sessions: int = 32) -> None:
        self._max_sessions = max_sessions
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _evict_if_needed(self) -> None:
        if len(self._sessions) < self._max_sessions:
            return
        oldest_key = min(self._sessions.values(), key=lambda state: state.updated_at).session_id
        self._sessions.pop(oldest_key, None)

    def create_session(self, image: Image.Image) -> str:
        with self._lock:
            self._evict_if_needed()
            session_id = uuid.uuid4().hex
            self._sessions[session_id] = SessionState(
                session_id=session_id,
                image_bytes=self._image_to_bytes(image),
            )
            return session_id

    def update_image(self, session_id: str, image: Image.Image) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    image_bytes=self._image_to_bytes(image),
                )
                return
            state.image_bytes = self._image_to_bytes(image)
            state.touch()

    def append_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                return
            state.history.append((user_text, assistant_text))
            state.touch()

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        state = self._sessions.get(session_id)
        return state.history[:] if state else []

    def get_image(self, session_id: str) -> Image.Image | None:
        state = self._sessions.get(session_id)
        if not state:
            return None
        return Image.open(io.BytesIO(state.image_bytes)).convert("RGB")

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()
