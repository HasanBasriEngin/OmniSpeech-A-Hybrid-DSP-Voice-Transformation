#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod backend;
mod commands;
mod types;

use tauri::Manager;

use commands::{
    backend_health, convert_celebrity, convert_emotion, convert_gender_age, convert_singing, convert_speaker_clone,
    export_audio_file, list_virtual_mic_devices,
    pick_audio_file, pick_midi_file, pick_reference_files, process_live_chunk, save_recording_wav, start_backend,
    send_dsp_feedback, start_live_session, stop_backend, stop_live_session,
};

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
            }
            tauri::async_runtime::spawn(async {
                let _ = backend::ensure_backend().await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            backend_health,
            pick_audio_file,
            pick_reference_files,
            pick_midi_file,
            save_recording_wav,
            export_audio_file,
            convert_emotion,
            convert_gender_age,
            convert_speaker_clone,
            convert_singing,
            convert_celebrity,
            send_dsp_feedback,
            list_virtual_mic_devices,
            start_live_session,
            process_live_chunk,
            stop_live_session,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    let _ = backend::stop_backend();
}
