use serde_json::{json, Value};
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
pub async fn convert_gender_age(
    input_path: String,
    mode: String,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "mode": mode,
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/gender-age", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
}

#[tauri::command]
pub async fn convert_speaker_clone(
    input_path: String,
    reference_paths: Vec<String>,
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "reference_paths": reference_paths,
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
    output_path: Option<String>,
) -> Result<ConversionResponse, String> {
    let payload = json!({
        "input_path": input_path,
        "midi_path": midi_path,
        "pitch_contour": pitch_contour,
        "output_path": output_path,
    });
    let response = backend::post("/api/convert/singing", payload).await?;
    serde_json::from_value(response).map_err(|err| format!("Invalid conversion response: {err}"))
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
