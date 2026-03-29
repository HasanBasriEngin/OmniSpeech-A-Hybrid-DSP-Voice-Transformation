mod backend;
mod commands;
mod types;

use commands::{
    backend_health, convert_gender_age, convert_singing, convert_speaker_clone, list_virtual_mic_devices,
    pick_audio_file, pick_midi_file, pick_reference_files, process_live_chunk, start_backend,
    start_live_session, stop_backend, stop_live_session,
};

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_backend,
            stop_backend,
            backend_health,
            pick_audio_file,
            pick_reference_files,
            pick_midi_file,
            convert_gender_age,
            convert_speaker_clone,
            convert_singing,
            list_virtual_mic_devices,
            start_live_session,
            process_live_chunk,
            stop_live_session,
        ])
        .on_page_load(|window, _payload| {
            let _ = window.show();
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    let _ = backend::stop_backend();
}
