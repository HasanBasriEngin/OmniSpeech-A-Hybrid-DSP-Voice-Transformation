use once_cell::sync::Lazy;
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tokio::time::{sleep, Duration};

const BASE_URL: &str = "http://127.0.0.1:8765";

#[derive(Default)]
struct BackendProcessState {
    child: Option<Child>,
}

static BACKEND_STATE: Lazy<Mutex<BackendProcessState>> = Lazy::new(|| Mutex::new(BackendProcessState::default()));

fn find_backend_script() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("backend").join("server.py"));
        candidates.push(cwd.join("../backend/server.py"));
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            candidates.push(parent.join("backend").join("server.py"));
            candidates.push(parent.join("../../backend/server.py"));
            candidates.push(parent.join("../../../backend/server.py"));
        }
    }

    candidates
        .into_iter()
        .map(|path| path.canonicalize().unwrap_or(path))
        .find(|path| path.exists())
}

fn python_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    if let Ok(custom) = std::env::var("OMNISPEECH_PYTHON") {
        if !custom.trim().is_empty() {
            candidates.push(custom);
        }
    }
    candidates.push("python".to_string());
    candidates.push("python3".to_string());
    candidates
}

fn pick_python() -> Option<String> {
    python_candidates().into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    })
}

pub async fn start_backend() -> Result<String, String> {
    let script = find_backend_script()
        .ok_or_else(|| "Python backend script not found at backend/server.py".to_string())?;

    let python = pick_python().ok_or_else(|| {
        "Python runtime not found. Set OMNISPEECH_PYTHON or install python/python3".to_string()
    })?;

    {
        let mut state = BACKEND_STATE
            .lock()
            .map_err(|_| "Failed to lock backend process state".to_string())?;

        if let Some(child) = state.child.as_mut() {
            match child.try_wait() {
                Ok(None) => return Ok("Backend already running".to_string()),
                Ok(Some(_)) | Err(_) => {
                    state.child = None;
                }
            }
        }

        let mut cmd = Command::new(&python);
        cmd.arg(script)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .stdin(Stdio::null());

        let child = cmd
            .spawn()
            .map_err(|err| format!("Failed to spawn backend process: {err}"))?;

        state.child = Some(child);
    }

    wait_for_health().await?;
    Ok("Backend started".to_string())
}

pub fn stop_backend() -> Result<String, String> {
    let mut state = BACKEND_STATE
        .lock()
        .map_err(|_| "Failed to lock backend process state".to_string())?;

    if let Some(mut child) = state.child.take() {
        let _ = child.kill();
        let _ = child.wait();
        return Ok("Backend stopped".to_string());
    }

    Ok("Backend already stopped".to_string())
}

pub async fn ensure_backend() -> Result<(), String> {
    if health().await {
        return Ok(());
    }
    let _ = start_backend().await?;
    Ok(())
}

pub async fn health() -> bool {
    let url = format!("{BASE_URL}/health");
    match reqwest::Client::new().get(url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

async fn wait_for_health() -> Result<(), String> {
    for _ in 0..40 {
        if health().await {
            return Ok(());
        }
        sleep(Duration::from_millis(250)).await;
    }
    Err("Backend did not become healthy in time".to_string())
}

pub async fn post(path: &str, payload: Value) -> Result<Value, String> {
    ensure_backend().await?;

    let response = reqwest::Client::new()
        .post(format!("{BASE_URL}{path}"))
        .json(&payload)
        .send()
        .await
        .map_err(|err| format!("Backend request failed: {err}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_else(|_| "".to_string());
        return Err(format!("Backend request failed ({status}): {text}"));
    }

    response
        .json::<Value>()
        .await
        .map_err(|err| format!("Invalid backend response JSON: {err}"))
}

pub async fn get(path: &str) -> Result<Value, String> {
    ensure_backend().await?;

    let response = reqwest::Client::new()
        .get(format!("{BASE_URL}{path}"))
        .send()
        .await
        .map_err(|err| format!("Backend request failed: {err}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_else(|_| "".to_string());
        return Err(format!("Backend request failed ({status}): {text}"));
    }

    response
        .json::<Value>()
        .await
        .map_err(|err| format!("Invalid backend response JSON: {err}"))
}
