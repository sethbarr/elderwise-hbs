pub mod process;
mod request;
mod response;
#[cfg(test)]
mod tests;

use process::{Engine, EngineStatus};
use response::AssessmentDecision;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use tauri::Manager;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineError {
    pub code: String,
    pub message: String,
}
impl EngineError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}
#[tauri::command]
pub async fn assess_session(
    app: tauri::AppHandle,
    session: Value,
) -> Result<AssessmentDecision, EngineError> {
    let engine = app.state::<Arc<Engine>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || engine.assess(&session))
        .await
        .map_err(|_| EngineError::new("internal", "Assessment worker failed"))?
}
#[tauri::command]
pub fn assessment_engine_status(app: tauri::AppHandle) -> EngineStatus {
    app.state::<Arc<Engine>>().status()
}
