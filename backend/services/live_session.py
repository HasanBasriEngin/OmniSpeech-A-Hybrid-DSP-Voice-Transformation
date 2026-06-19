from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

import numpy as np

from backend.audio.filtering import LiveVoicePostFilter
from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.virtual_mic import VirtualMicRouter


class LiveOverlapCrossfader:
    """Smooth boundaries between independently processed live chunks."""

    def __init__(self, sample_rate: int, *, overlap_ms: float = 24.0, max_overlap_ratio: float = 0.25) -> None:
        self.sample_rate = sample_rate
        self.overlap_samples = max(1, int(round(sample_rate * overlap_ms / 1000.0)))
        self.max_overlap_ratio = float(np.clip(max_overlap_ratio, 0.02, 0.5))
        self._previous_tail: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32)
        if x.size == 0:
            return x
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        out = np.asarray(x, dtype=np.float32).copy()
        overlap = min(self.overlap_samples, max(1, int(out.size * self.max_overlap_ratio)))

        if self._previous_tail is None:
            fade = self._smooth_fade(min(overlap, out.size))
            out[: fade.size] *= fade
        else:
            overlap = min(overlap, out.size, self._previous_tail.size)
            if overlap > 1:
                fade_in = self._smooth_fade(overlap)
                previous_context = self._previous_tail[-overlap:][::-1]
                out[:overlap] = previous_context * (1.0 - fade_in) + out[:overlap] * fade_in
            elif overlap == 1:
                out[0] = 0.5 * float(self._previous_tail[-1]) + 0.5 * float(out[0])

        tail_size = min(self.overlap_samples, out.size)
        self._previous_tail = out[-tail_size:].copy()
        return out.astype(np.float32)

    @staticmethod
    def _smooth_fade(length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros(0, dtype=np.float32)
        positions = np.linspace(0.0, 1.0, length, dtype=np.float32)
        return (positions * positions * (3.0 - 2.0 * positions)).astype(np.float32)


@dataclass
class LiveSession:
    session_id: str
    task: str
    options: dict[str, object]
    route_to_virtual_mic: bool
    router: VirtualMicRouter | None
    post_filter: LiveVoicePostFilter
    crossfader: LiveOverlapCrossfader


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
        self.pipeline.apply_system_clownfish_for_task(task, options)
        if route_to_virtual_mic:
            router = VirtualMicRouter(sample_rate=self.sample_rate)
            router.open(preferred_device=virtual_mic_device)

        session = LiveSession(
            session_id=session_id,
            task=task,
            options=options,
            route_to_virtual_mic=route_to_virtual_mic,
            router=router,
            post_filter=LiveVoicePostFilter(self.sample_rate),
            crossfader=LiveOverlapCrossfader(self.sample_rate),
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
        processed = session.post_filter.process(processed)
        processed = session.crossfader.process(processed)

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
