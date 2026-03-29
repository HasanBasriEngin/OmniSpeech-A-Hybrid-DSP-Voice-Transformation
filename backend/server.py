from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import build_router
from backend.config import SETTINGS
from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.live_session import LiveSessionManager


def create_app() -> FastAPI:
    pipeline = VoiceConversionPipeline(sample_rate=SETTINGS.model_sample_rate)
    live_manager = LiveSessionManager(pipeline=pipeline, sample_rate=SETTINGS.live_sample_rate)

    app = FastAPI(title="OmniSpeech Backend", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["tauri://localhost", "http://tauri.localhost", "http://localhost:1420"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_router(pipeline, live_manager))

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        live_manager.shutdown()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host=SETTINGS.host, port=SETTINGS.port, reload=False)
