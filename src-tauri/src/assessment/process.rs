//! Native-only process ownership. Tauri bundles externalBin; no frontend shell permission.
use super::{
    request::{build_request, MODEL},
    response::{normalize, AssessmentDecision},
    EngineError,
};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

pub struct Config {
    pub executable: PathBuf,
    pub startup_timeout: Duration,
}
impl Config {
    pub fn for_app() -> Result<Self, EngineError> {
        // Overrides are native developer configuration, never frontend arguments.
        #[cfg(debug_assertions)]
        if let Some(path) = std::env::var_os("ELDERWISE_LLAMA_SERVER") {
            return Ok(Self {
                executable: path.into(),
                startup_timeout: Duration::from_secs(1200),
            });
        }
        let executable = std::env::current_exe()
            .map_err(|e| EngineError::new("startup", e.to_string()))?
            .parent()
            .ok_or_else(|| EngineError::new("startup", "Cannot locate sidecar directory"))?
            .join(if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            });
        Ok(Self {
            executable,
            startup_timeout: Duration::from_secs(1200),
        })
    }
}
struct ManagedChild {
    child: Child,
    base_url: String,
    key: String,
}
impl Drop for ManagedChild {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        #[cfg(unix)]
        {
            // SAFETY: pid belongs to this unreaped, owned child; no shell is involved.
            unsafe {
                libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineStatus {
    pub state: &'static str,
    pub error: Option<EngineError>,
}
pub struct Engine {
    config: Config,
    child: Mutex<Option<ManagedChild>>,
    gate: Mutex<()>,
    stopped: AtomicBool,
    status: Mutex<EngineStatus>,
}
impl Engine {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            child: Mutex::new(None),
            gate: Mutex::new(()),
            stopped: AtomicBool::new(false),
            status: Mutex::new(EngineStatus {
                state: "idle",
                error: None,
            }),
        }
    }
    pub fn status(&self) -> EngineStatus {
        self.status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    fn set_status(&self, state: &'static str, error: Option<EngineError>) {
        let mut status = self.status.lock().unwrap_or_else(|e| e.into_inner());
        *status = if self.stopped.load(Ordering::SeqCst) {
            EngineStatus {
                state: "stopped",
                error: None,
            }
        } else {
            EngineStatus { state, error }
        };
    }
    fn client() -> Result<Client, EngineError> {
        Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| EngineError::new("transport", e.to_string()))
    }
    pub fn shutdown(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.child.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.set_status("stopped", None);
    }
    fn start(&self) -> Result<(String, String), EngineError> {
        let mut slot = self.child.lock().unwrap_or_else(|e| e.into_inner());
        if self.stopped.load(Ordering::SeqCst) {
            return Err(EngineError::new(
                "stopped",
                "Assessment engine is shutting down",
            ));
        }
        if let Some(child) = slot.as_mut() {
            if child
                .child
                .try_wait()
                .map_err(|e| EngineError::new("startup", e.to_string()))?
                .is_none()
            {
                return Ok((child.base_url.clone(), child.key.clone()));
            }
            slot.take();
        }
        // Reserve an ephemeral loopback port; API key protects against the bind handoff race.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")
            .map_err(|e| EngineError::new("startup", e.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|e| EngineError::new("startup", e.to_string()))?
            .port()
            .to_string();
        let key = uuid::Uuid::new_v4().to_string();
        drop(listener);
        let child = Command::new(&self.config.executable)
            .args(["-hf",MODEL,"--n-gpu-layers","99","--ctx-size","2048","--batch-size","2048","--ubatch-size","2048","--flash-attn","on","--host","127.0.0.1","--port",&port,"--parallel","1"])
            .env("LLAMA_API_KEY", &key)
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
            .spawn().map_err(|e| EngineError::new("startup",format!("Cannot start bundled llama-server: {e}. Provision the sidecar for this platform.")))?;
        let base_url = format!("http://127.0.0.1:{port}");
        *slot = Some(ManagedChild {
            child,
            base_url: base_url.clone(),
            key: key.clone(),
        });
        Ok((base_url, key))
    }
    fn ready(&self, client: &Client) -> Result<(String, String), EngineError> {
        self.set_status("starting", None);
        let (base, key) = self.start()?;
        let deadline = Instant::now() + self.config.startup_timeout;
        loop {
            if self.stopped.load(Ordering::SeqCst) {
                return Err(EngineError::new(
                    "stopped",
                    "Assessment engine is shutting down",
                ));
            }
            {
                let mut slot = self.child.lock().unwrap_or_else(|e| e.into_inner());
                let child = slot
                    .as_mut()
                    .ok_or_else(|| EngineError::new("stopped", "Assessment engine stopped"))?;
                if let Some(exit) = child
                    .child
                    .try_wait()
                    .map_err(|e| EngineError::new("startup", e.to_string()))?
                {
                    slot.take();
                    return Err(EngineError::new("startup",format!("llama-server exited ({exit}). Check first-run network access, disk space, available RAM/VRAM and sidecar compatibility.")));
                }
            }
            if client
                .get(format!("{base}/health"))
                .timeout(Duration::from_secs(2))
                .send()
                .ok()
                .filter(|r| r.status().is_success())
                .and_then(|r| r.json::<Value>().ok())
                .is_some_and(|v| v["status"] == "ok")
            {
                // /health is public; authenticated properties verifies this is OUR server.
                if client
                    .get(format!("{base}/props"))
                    .bearer_auth(&key)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .is_ok_and(|r| r.status().is_success())
                {
                    self.set_status("ready", None);
                    return Ok((base, key));
                }
            }
            if Instant::now() >= deadline {
                self.child.lock().unwrap_or_else(|e| e.into_inner()).take();
                return Err(EngineError::new("startup_timeout","Local model startup timed out. First use downloads about 6.5 GB into llama.cpp's local cache; check network/disk capacity and retry by restarting Elderwise."));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    pub fn ensure_ready(&self) -> Result<(), EngineError> {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let result = Self::client().and_then(|c| self.ready(&c).map(|_| ()));
        if let Err(e) = &result {
            self.set_status("unavailable", Some(e.clone()));
        }
        result
    }
    pub fn assess(&self, session: &Value) -> Result<AssessmentDecision, EngineError> {
        let request = build_request(session)?;
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let result =
            (|| {
                let client = Self::client()?;
                let (base, key) = self.ready(&client)?;
                let response = client
                    .post(format!("{base}/v1/systemone"))
                    .bearer_auth(key)
                    .json(&request)
                    .send()
                    .map_err(|_| {
                        EngineError::new("transport", "Local model request failed or timed out")
                    })?;
                if !response.status().is_success() {
                    return Err(EngineError::new(
                        "transport",
                        format!("Local model returned HTTP {}", response.status()),
                    ));
                }
                // Bound response memory and never expose server prompts/raw response in errors.
                use std::io::Read;
                let mut bytes = Vec::new();
                response
                    .take(65537)
                    .read_to_end(&mut bytes)
                    .map_err(|_| EngineError::new("transport", "Cannot read local decision"))?;
                if bytes.len() > 65536 {
                    return Err(EngineError::new(
                        "invalid_response",
                        "Local decision exceeded response limit",
                    ));
                }
                normalize(serde_json::from_slice(&bytes).map_err(|_| {
                    EngineError::new("invalid_response", "Invalid local model JSON")
                })?)
            })();
        if let Err(e) = &result {
            self.set_status("unavailable", Some(e.clone()));
        }
        result
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_executable_is_typed_and_shutdown_prevents_restart() {
        let engine = Engine::new(Config {
            executable: PathBuf::from("/nonexistent/elderwise-llama-server"),
            startup_timeout: Duration::from_millis(10),
        });
        assert_eq!(engine.ensure_ready().unwrap_err().code, "startup");
        assert_eq!(engine.status().state, "unavailable");
        engine.shutdown();
        assert_eq!(engine.ensure_ready().unwrap_err().code, "stopped");
        assert_eq!(engine.status().state, "stopped");
    }
    #[test]
    #[ignore = "Requires provisioned Clef Q4 and local llama-server; downloads model on first use"]
    fn managed_clef_smoke() {
        let executable =
            std::env::var_os("ELDERWISE_LLAMA_SERVER").expect("set ELDERWISE_LLAMA_SERVER");
        let engine = Engine::new(Config {
            executable: executable.into(),
            startup_timeout: Duration::from_secs(1200),
        });
        engine.ensure_ready().unwrap();
        let pid = engine.child.lock().unwrap().as_ref().unwrap().child.id();
        engine.ensure_ready().unwrap();
        assert_eq!(
            pid,
            engine.child.lock().unwrap().as_ref().unwrap().child.id()
        );
        let session: Value =
            serde_json::from_str(include_str!("fixtures/full-session.json")).unwrap();
        let decision = engine.assess(&session).unwrap();
        println!(
            "Managed Clef decision: {}",
            serde_json::to_string(&decision).unwrap()
        );
        let base = engine
            .child
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .base_url
            .clone();
        engine.shutdown();
        assert!(engine.child.lock().unwrap().is_none());
        assert!(std::net::TcpStream::connect(base.strip_prefix("http://").unwrap()).is_err());
    }
}

#[cfg(all(test, unix))]
mod lifecycle_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    fn fake() -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("elderwise-clef-test-{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, "#!/bin/sh\nexec sleep 30\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }
    #[test]
    fn startup_timeout_reaps_owned_child() {
        let executable = fake();
        let engine = Engine::new(Config {
            executable: executable.clone(),
            startup_timeout: Duration::from_millis(5),
        });
        assert_eq!(engine.ensure_ready().unwrap_err().code, "startup_timeout");
        assert!(engine.child.lock().unwrap().is_none());
        std::fs::remove_file(executable).unwrap();
    }
    #[test]
    fn shutdown_cancels_inflight_startup_and_reaps_child() {
        let executable = fake();
        let engine = std::sync::Arc::new(Engine::new(Config {
            executable: executable.clone(),
            startup_timeout: Duration::from_secs(30),
        }));
        let worker = engine.clone();
        let task = std::thread::spawn(move || worker.ensure_ready());
        let deadline = Instant::now() + Duration::from_secs(2);
        while engine.child.lock().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(engine.child.lock().unwrap().is_some());
        engine.shutdown();
        assert_eq!(task.join().unwrap().unwrap_err().code, "stopped");
        assert!(engine.child.lock().unwrap().is_none());
        std::fs::remove_file(executable).unwrap();
    }
}
