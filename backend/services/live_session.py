from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

import numpy as np

from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.virtual_mic import VirtualMicRouter


@dataclass
class LiveSession:
    session_id: str
    task: str
    options: dict[str, object]
    route_to_virtual_mic: bool
    router: VirtualMicRouter | None


class LiveSessionManager:
    def __init__(self, pipeline: VoiceConversionPipeline, sample_rate: int) -> None:
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        self._sessions: dict[str, LiveSession] = {}
        self._lock = Lock()

    def list_virtual_mics(self) -> list[str]:
        return VirtualMicRouter(sample_rate=self.sample_rate).list_candidate_devices()

    def start_session(
        self,
        task: str,
        options: dict[str, object],
        route_to_virtual_mic: bool,
        virtual_mic_device: str | None,
    ) -> LiveSession:
        session_id = str(uuid4())
        router = None
        if route_to_virtual_mic:
            router = VirtualMicRouter(sample_rate=self.sample_rate)
            router.open(preferred_device=virtual_mic_device)

        session = LiveSession(
            session_id=session_id,
            task=task,
            options=options,
            route_to_virtual_mic=route_to_virtual_mic,
            router=router,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def process_chunk(self, session_id: str, chunk: np.ndarray) -> np.ndarray:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown live session: {session_id}")

        processed = self.pipeline.process_live_chunk(
            chunk=np.asarray(chunk, dtype=np.float32),
            task=session.task,
            options=session.options,
        )

        if session.route_to_virtual_mic and session.router is not None:
            session.router.write(processed)

        return processed

    def stop_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session and session.router:
            session.router.close()

    def shutdown(self) -> None:
        with self._lock:
            session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            self.stop_session(session_id)
