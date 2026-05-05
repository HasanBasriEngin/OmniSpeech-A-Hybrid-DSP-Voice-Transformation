from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import load_settings


class BackendRuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._ready = False
        self._error: str | None = None
        self._live_manager = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def _mark_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _attach_runtime(self, app: FastAPI, router, live_manager) -> None:
        router.routes = [route for route in router.routes if getattr(route, "path", None) != "/health"]
        app.include_router(router)
        with self._lock:
            self._ready = True
            self._error = None
            self._live_manager = live_manager

    def start_background_init(self, app: FastAPI) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                settings = load_settings()
                from backend.api.routes import build_router
                from backend.pipeline.processor import VoiceConversionPipeline
                from backend.services.live_session import LiveSessionManager

                pipeline = VoiceConversionPipeline(sample_rate=settings.model_sample_rate)
                live_manager = LiveSessionManager(pipeline=pipeline, sample_rate=settings.live_sample_rate)
                router = build_router(pipeline, live_manager)
                loop.call_soon_threadsafe(self._attach_runtime, app, router, live_manager)
            except Exception as exc:  # pragma: no cover - startup path only
                self._mark_error(str(exc))

        threading.Thread(target=worker, name="omnispeech-backend-init", daemon=True).start()

    def shutdown(self) -> None:
        with self._lock:
            live_manager = self._live_manager
        if live_manager is not None:
            live_manager.shutdown()


RUNTIME_STATE = BackendRuntimeState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNTIME_STATE.start_background_init(app)
    yield
    RUNTIME_STATE.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="OmniSpeech Backend", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["tauri://localhost", "http://tauri.localhost", "http://localhost:1420"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        ready = RUNTIME_STATE.ready
        error = RUNTIME_STATE.error
        return {
            "status": "ok" if ready and not error else "starting" if not error else "error",
            "backend": "omnispeech-python",
            "ready": ready,
            "error": error,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = load_settings()
    uvicorn.run("backend.server:app", host=settings.host, port=settings.port, reload=False)
