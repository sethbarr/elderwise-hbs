// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use tauri::Manager;

mod ai;
mod assessment;
mod export;
mod history;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init());

    builder
        .manage(ai::AiState::new())
        .manage(history::HistoryState::new())
        .setup(|app| {
            // Version from Cargo (bumped in lockstep with the other manifests
            // at release) so the window title shows the running build.
            if let Some(window) = app.get_webview_window("main") {
                window.set_title(&format!("Elderwise v{}", env!("CARGO_PKG_VERSION")))?;
            }
            let config = assessment::process::Config::for_app()
                .map_err(|e| std::io::Error::other(e.message))?;
            let engine = std::sync::Arc::new(assessment::process::Engine::new(config));
            app.manage(engine.clone());
            tauri::async_runtime::spawn_blocking(move || {
                let _ = engine.ensure_ready();
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            assessment::assess_session,
            assessment::assessment_engine_status,
            ai::ai_get_settings,
            ai::ai_set_settings,
            ai::ai_set_key,
            ai::ai_clear_key,
            ai::ai_status,
            ai::ai_test,
            ai::transcribe_audio,
            ai::synthesize_speech,
            ai::summarize_assessment,
            export::export_default_dir,
            export::export_dialog_mode,
            export::export_save_pdf,
            history::history_append,
            history::history_read,
            history::history_clear,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                app.state::<std::sync::Arc<assessment::process::Engine>>()
                    .shutdown();
            }
        })
}
