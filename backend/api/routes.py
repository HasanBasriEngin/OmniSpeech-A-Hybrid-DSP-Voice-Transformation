from __future__ import annotations

from fastapi import APIRouter, HTTPException
import numpy as np

from backend.api.schemas import (
    CelebrityRequest,
    ConversionResponse,
    DSPFeedbackRequest,
    DSPFeedbackResponse,
    EmotionRequest,
    GenderAgeRequest,
    HealthResponse,
    LiveChunkRequest,
    LiveChunkResponse,
    LiveSessionStartRequest,
    LiveSessionStartResponse,
    LiveSessionStopRequest,
    SingingRequest,
    SpeakerCloneRequest,
    VirtualMicDevicesResponse,
)
from backend.audio.dsp_profiles import apply_user_feedback
from backend.pipeline.processor import VoiceConversionPipeline
from backend.services.live_session import LiveSessionManager


def build_router(pipeline: VoiceConversionPipeline, live_manager: LiveSessionManager) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", backend="omnispeech-python")

    @router.post("/api/convert/emotion", response_model=ConversionResponse)
    async def convert_emotion(payload: EmotionRequest) -> ConversionResponse:
        try:
            result = pipeline.convert_emotion_file(
                input_path=payload.input_path,
                emotion=payload.emotion,
                pitch_override=payload.pitch_override,
                rate_override=payload.rate_override,
                energy_override=payload.energy_override,
                use_ai_engines=payload.use_ai_engines,
                output_path=payload.output_path,
            )
            return ConversionResponse(output_path=result.output_path, metrics=result.metrics)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/convert/gender-age", response_model=ConversionResponse)
    async def convert_gender_age(payload: GenderAgeRequest) -> ConversionResponse:
        try:
            result = pipeline.convert_gender_age_file(
                input_path=payload.input_path,
                mode=payload.mode,
                use_ai_engines=payload.use_ai_engines,
                output_path=payload.output_path,
            )
            return ConversionResponse(output_path=result.output_path, metrics=result.metrics)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/convert/speaker-clone", response_model=ConversionResponse)
    async def convert_speaker_clone(payload: SpeakerCloneRequest) -> ConversionResponse:
        try:
            result = pipeline.convert_speaker_clone_file(
                input_path=payload.input_path,
                reference_paths=payload.reference_paths,
                use_ai_engines=payload.use_ai_engines,
                output_path=payload.output_path,
            )
            return ConversionResponse(output_path=result.output_path, metrics=result.metrics)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/convert/singing", response_model=ConversionResponse)
    async def convert_singing(payload: SingingRequest) -> ConversionResponse:
        try:
            result = pipeline.convert_singing_file(
                input_path=payload.input_path,
                midi_path=payload.midi_path,
                pitch_contour=payload.pitch_contour,
                use_ai_engines=payload.use_ai_engines,
                output_path=payload.output_path,
            )
            return ConversionResponse(output_path=result.output_path, metrics=result.metrics)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/convert/celebrity", response_model=ConversionResponse)
    async def convert_celebrity(payload: CelebrityRequest) -> ConversionResponse:
        try:
            result = pipeline.convert_celebrity_file(
                input_path=payload.input_path,
                celebrity=payload.celebrity,
                use_ai_engines=payload.use_ai_engines,
                output_path=payload.output_path,
            )
            return ConversionResponse(output_path=result.output_path, metrics=result.metrics)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/live/virtual-mics", response_model=VirtualMicDevicesResponse)
    async def list_virtual_mics() -> VirtualMicDevicesResponse:
        return VirtualMicDevicesResponse(devices=live_manager.list_virtual_mics())

    @router.post("/api/dsp/feedback", response_model=DSPFeedbackResponse)
    async def send_dsp_feedback(payload: DSPFeedbackRequest) -> DSPFeedbackResponse:
        try:
            settings = apply_user_feedback(
                payload.profile_name,
                payload.feedback,
                profiles_dir=pipeline.dsp_profiles_dir,
            )
            return DSPFeedbackResponse(
                profile_name=payload.profile_name,
                feedback=payload.feedback,
                settings=settings.as_filter_settings(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/live/start", response_model=LiveSessionStartResponse)
    async def start_live(payload: LiveSessionStartRequest) -> LiveSessionStartResponse:
        try:
            session = live_manager.start_session(
                task=payload.task,
                options=payload.options,
                route_to_virtual_mic=payload.route_to_virtual_mic,
                virtual_mic_device=payload.virtual_mic_device,
            )
            return LiveSessionStartResponse(session_id=session.session_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/live/chunk", response_model=LiveChunkResponse)
    async def push_live_chunk(payload: LiveChunkRequest) -> LiveChunkResponse:
        if not payload.chunk:
            return LiveChunkResponse(chunk=[])

        try:
            processed = live_manager.process_chunk(
                payload.session_id,
                np.asarray(payload.chunk, dtype=np.float32),
            )
            return LiveChunkResponse(chunk=np.asarray(processed, dtype=np.float32).tolist())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/live/stop")
    async def stop_live(payload: LiveSessionStopRequest) -> dict[str, str]:
        live_manager.stop_session(payload.session_id)
        return {"status": "stopped"}

    return router
