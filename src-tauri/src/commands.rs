use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::AppHandle;
use tauri_plugin_dialog::{DialogExt, FilePath};

use crate::backend;
use crate::types::{ConversionResponse, LiveChunkResponse, LiveDevicesResponse, LiveSessionResponse};

fn map_file_path(path: FilePath) -> String {
    match path {
        FilePath::Path(path_buf) => path_buf.to_string_lossy().to_string(),
        FilePath::Url(url) => url.to_string(),
    }
}

#[tauri::command]
pub async fn start_backend() -> Result<String, String> {
    backend::start_backend().await
}

#[tauri::command]
pub fn stop_backend() -> Result<String, String> {
    backend::stop_backend()
}

#[tauri::command]
pub async fn backend_health() -> bool {
    backend::health().await
}

#[tauri::command]
pub async fn pick_audio_file(app: AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    app.dialog()
        .file()
        .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a"])
        .pick_file(move |path| {
            let mapped = path.map(map_file_path);
            let _ = tx.send(mapped);
        });

    rx.await
        .map_err(|err| format!("Failed to receive file dialog response: {err}"))
}

#[tauri::command]
pub async fn pick_reference_files(app: AppHandle) -> Result<Vec<String>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    app.dialog()
        .file()
        .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a"])
        .pick_files(move |paths| {
            let mapped = paths
                .unwrap_or_default()
                .into_iter()
                .map(map_file_path)
                .collect::<Vec<_>>();
            let _ = tx.send(mapped);
        });

    rx.await
        .map_err(|err| format!("Failed to receive file dialog response: {err}"))
}

#[tauri::command]
pub async fn pick_midi_file(app: AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    app.dialog()
        .file()
        .add_filter("MIDI", &["mid", "midi"])
        .pick_file(move |path| {
            let mapped = path.map(map_file_path);
            let _ = tx.send(mapped);
        });

    rx.await
        .map_err(|err| format!("Failed to receive file dialog response: {err}"))
}

#[tauri::command]
pub fn save_recording_wav(bytes: Vec<u8>) -> Result<String, String> {
    if bytes.is_empty() {
        return Err("Recording is empty".to_string());
    }

    let mut dir = std::env::temp_dir();
    dir.push("omnispeech-recordings");
    fs::create_dir_all(&dir).map_err(|err| format!("Failed to create recording directory: {err}"))?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("System clock error: {err}"))?
        .as_millis();
    let path = dir.join(format!("mic_recording_{timestamp}.wav"));
    fs::write(&path, bytes).map_err(|err| format!("Failed to save recording: {err}"))?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn export_audio_file(app: AppHandle, source_path: String) -> Result<Option<String>, String> {
    let source = PathBuf::from(&source_path);
    if !source.exists() {
        return Err(format!("Output file does not exist: {source_path}"));
    }
    if !source.is_file() {
        return Err(format!("Output path is not a file: {source_path}"));
    }

    let suggested_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or("omnispeech_output.wav")
        .to_string();

    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .add_filter("Audio", &["wav", "flac", "mp3", "ogg", "m4a"])
        .set_file_name(&suggested_name)
        .save_file(move |path| {
            let mapped = path.map(map_file_path);
            let _ = tx.send(mapped);
        });

    let Some(destination) = rx
        .await
        .map_err(|err| format!("Failed to receive save dialog response: {err}"))?
    else {
        return Ok(None);
    };

    let destination_path = Path::new(&destination);
    if let Some(parent) = destination_path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("Failed to create export directory: {err}"))?;
    }
    if source == destination_path {
        return Ok(Some(destination));
    }

    fs::copy(&source, destination_path).map_err(|err| format!("Failed to export audio: {err}"))?;
    Ok(Some(destination))
}

#[tauri::command]
pub async fn convert_emotion(
    input_path: String,
    emotion: String,
    pitch_override: Option<f64>,
    rate_override: Option<f64>,
    energy_override: Option<f64>,
    use_ai_engines: Option<bool>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "emotion": emotion,
        "pitch_override": pitch_override,
        "rate_override": rate_override,
        "energy_override": energy_override,
        "use_ai_engines": use_ai_engines.unwrap_or(true),
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/emotion", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn convert_gender_age(
    input_path: String,
    mode: String,
    use_ai_engines: Option<bool>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "mode": mode,
        "use_ai_engines": use_ai_engines.unwrap_or(true),
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/gender-age", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn convert_speaker_clone(
    input_path: String,
    reference_paths: Vec<String>,
    use_ai_engines: Option<bool>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "reference_paths": reference_paths,
        "use_ai_engines": use_ai_engines.unwrap_or(true),
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/speaker-clone", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn convert_singing(
    input_path: String,
    midi_path: Option<String>,
    pitch_contour: Option<Vec<f32>>,
    use_ai_engines: Option<bool>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "midi_path": midi_path,
        "pitch_contour": pitch_contour,
        "use_ai_engines": use_ai_engines.unwrap_or(true),
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/singing", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn convert_celebrity(
    input_path: String,
    celebrity: String,
    use_ai_engines: Option<bool>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "celebrity": celebrity,
        "use_ai_engines": use_ai_engines.unwrap_or(true),
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/celebrity", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn convert_voice_clone(
    input_path: String,
    reference_paths: Vec<String>,
    celebrity: Option<String>,
    use_ai_engines: Option<bool>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "reference_paths": reference_paths,
        "celebrity": celebrity,
        "use_ai_engines": use_ai_engines.unwrap_or(true),
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/voice-clone", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn send_dsp_feedback(profile_name: String, feedback: String) -> Result<Value, String> {
    let payload = json!({
        "profile_name": profile_name,
        "feedback": feedback,
    });
    backend::post("/api/dsp/feedback", payload).await
}

#[tauri::command]
pub async fn list_virtual_mic_devices() -> Result<Vec<String>, String> {
    let response = backend::get("/api/live/virtual-mics").await?;
    let parsed: LiveDevicesResponse =
        serde_json::from_value(response).map_err(|err| format!("Invalid devices response: {err}"))?;
    Ok(parsed.devices)
}

#[tauri::command]
pub async fn start_live_session(
    task: String,
    options: Value,
    route_to_virtual_mic: bool,
    virtual_mic_device: Option<String>,
) -> Result<String, String> {
    let payload = json!({
        "task": task,
        "options": options,
        "route_to_virtual_mic": route_to_virtual_mic,
        "virtual_mic_device": virtual_mic_device,
    });
    let response = backend::post("/api/live/start", payload).await?;
    let parsed: LiveSessionResponse =
        serde_json::from_value(response).map_err(|err| format!("Invalid session response: {err}"))?;
    Ok(parsed.session_id)
}

#[tauri::command]
pub async fn process_live_chunk(session_id: String, chunk: Vec<f32>) -> Result<Vec<f32>, String> {
    let payload = json!({
        "session_id": session_id,
        "chunk": chunk,
    });
    let response = backend::post("/api/live/chunk", payload).await?;
    let parsed: LiveChunkResponse =
        serde_json::from_value(response).map_err(|err| format!("Invalid live chunk response: {err}"))?;
    Ok(parsed.chunk)
}

#[tauri::command]
pub async fn stop_live_session(session_id: String) -> Result<String, String> {
    let payload = json!({ "session_id": session_id });
    let response = backend::post("/api/live/stop", payload).await?;

    match response.get("status").and_then(|status| status.as_str()) {
        Some(status) => Ok(status.to_string()),
        None => Ok("stopped".to_string()),
    }
}
