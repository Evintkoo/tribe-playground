//! TRIBE v2 — pure-Rust inference server (tribe-server).
//!
//! On startup:
//!   1. Ensures `best.safetensors` exists (run `python3 convert_ckpt.py` if not).
//!   2. Loads FmriEncoder (candle, CPU/Metal).
//!   3. Tries to load Wav2Vec2Bert, LlamaTextEncoder, ClipVisualEncoder (all optional).
//!   4. Spawns a background download worker (processes job queue one-at-a-time).
//!   5. Starts Axum HTTP server on port 8081 with WebSocket at /ws.
//!
//! No Python process is spawned at runtime.

use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tracing::{info, warn, error};

use anyhow::{Context, Result};
use axum::{
    http::Method,
    routing::{get, post},
    Router,
};
use candle_core::Device;
use tokio::sync::{mpsc, RwLock};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use uuid::Uuid;

mod audio;
mod clip_encoder;
mod downloader;
mod features;
mod fmri_encoder;
mod handlers;
mod jobs;
mod llama_encoder;
mod wav2vec2bert;

use audio::MelSpec;
use clip_encoder::ClipVisualEncoder;
use fmri_encoder::FmriEncoder;
use jobs::JobStore;
use llama_encoder::LlamaTextEncoder;
use wav2vec2bert::Wav2Vec2Bert;

// ── Config ────────────────────────────────────────────────────────────────────

const SERVER_PORT: u16 = 8081;

// ── Shared application state ──────────────────────────────────────────────────

pub struct AppState {
    pub fmri:      FmriEncoder,
    pub text_enc:  Arc<RwLock<Option<LlamaTextEncoder>>>,
    pub audio_enc: Arc<RwLock<Option<Wav2Vec2Bert>>>,
    pub mel_spec:  Option<MelSpec>,
    pub clip_enc:  Arc<RwLock<Option<ClipVisualEncoder>>>,
    pub root:      PathBuf,
    pub device:    Device,
    pub job_store: JobStore,
    pub job_tx:    mpsc::Sender<Uuid>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn find_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap();
    loop {
        if dir.join("tribe-v2-weights").exists() {
            return dir;
        }
        if !dir.pop() {
            return std::env::current_dir().unwrap();
        }
    }
}

fn pick_device() -> Device {
    if std::env::var("TRIBE_DEVICE").as_deref() == Ok("cpu") {
        info!("using CPU (forced via TRIBE_DEVICE)");
        return Device::Cpu;
    }
    if let Ok(dev) = Device::new_metal(0) {
        info!("using Metal GPU");
        return dev;
    }
    info!("using CPU");
    Device::Cpu
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let root        = find_root();
    let weights_dir = root.join("tribe-v2-weights");
    let st_path     = weights_dir.join("best.safetensors");

    // ── Convert checkpoint if needed ──────────────────────────────────────────
    if !st_path.exists() {
        warn!("best.safetensors not found — running convert_ckpt.py");
        let status = std::process::Command::new("python3")
            .arg(root.join("convert_ckpt.py"))
            .current_dir(&root)
            .status()
            .context("failed to spawn python3 convert_ckpt.py")?;
        if !status.success() {
            anyhow::bail!("convert_ckpt.py failed; run it manually first");
        }
    }

    let device = pick_device();

    // ── Load FmriEncoder ─────────────────────────────────────────────────────
    info!("loading FmriEncoder");
    let fmri = FmriEncoder::load(
        st_path.to_str().context("invalid path")?,
        &device,
    ).context("failed to load FmriEncoder")?;
    info!(params_m = format!("{:.1}", fmri.n_params() as f64 / 1e6), "FmriEncoder loaded");

    // ── MelSpec ───────────────────────────────────────────────────────────────
    let mel_spec = match MelSpec::load(&weights_dir) {
        Ok(m)  => { info!("MelSpec loaded"); Some(m) }
        Err(e) => { warn!(error = %e, "MelSpec not loaded"); None }
    };

    // ── Wav2Vec2Bert ──────────────────────────────────────────────────────────
    let w2v_path = weights_dir.join("w2v-bert-2.0.safetensors");
    let audio_enc: Option<Wav2Vec2Bert> = if w2v_path.exists() {
        let p = w2v_path.to_str().context("invalid w2v path")?;
        info!("loading Wav2Vec2Bert");
        match Wav2Vec2Bert::load(p, &device) {
            Ok(enc) => { info!("Wav2Vec2Bert loaded"); Some(enc) }
            Err(e)  => { error!(error = %e, "Wav2Vec2Bert failed"); None }
        }
    } else {
        warn!("w2v-bert-2.0.safetensors not found — run POST /api/download/wav2vec");
        None
    };

    // ── LlamaTextEncoder ─────────────────────────────────────────────────────
    let llama_dir = weights_dir.join("llama");
    let text_enc: Option<LlamaTextEncoder> = if llama_dir.exists() {
        info!("loading LLaMA text encoder");
        match LlamaTextEncoder::load(&llama_dir, &device) {
            Ok(enc) => { info!("LLaMA text encoder loaded"); Some(enc) }
            Err(e)  => { error!(error = %e, "LLaMA text encoder failed"); None }
        }
    } else {
        warn!("tribe-v2-weights/llama/ not found — run POST /api/download/llama");
        None
    };

    // ── CLIP ViT-L/14 ────────────────────────────────────────────────────────
    let clip_path = weights_dir.join("clip").join("model.safetensors");
    let clip_enc: Option<ClipVisualEncoder> = if clip_path.exists() {
        let p = clip_path.to_str().context("invalid clip path")?;
        info!("loading CLIP ViT-L/14");
        match ClipVisualEncoder::load(p, &device) {
            Ok(enc) => { info!("CLIP encoder loaded"); Some(enc) }
            Err(e)  => { error!(error = %e, "CLIP encoder failed"); None }
        }
    } else {
        warn!("tribe-v2-weights/clip/model.safetensors not found — run POST /api/download/clip");
        None
    };

    // ── Metal JIT warmup ─────────────────────────────────────────────────────
    info!("warming up Metal kernels");
    {
        use candle_core::Tensor;
        let dummy = Tensor::zeros((1, 4, 6144), candle_core::DType::F32, &device)?;
        let _ = fmri.forward(Some(&dummy), None, None);
    }
    info!("warmup done");

    // ── Job infrastructure ────────────────────────────────────────────────────
    let job_store = jobs::new_job_store();
    let (job_tx, mut job_rx) = mpsc::channel::<Uuid>(64);

    let state = Arc::new(AppState {
        fmri,
        text_enc:  Arc::new(RwLock::new(text_enc)),
        audio_enc: Arc::new(RwLock::new(audio_enc)),
        mel_spec,
        clip_enc:  Arc::new(RwLock::new(clip_enc)),
        root:      root.clone(),
        device,
        job_store: job_store.clone(),
        job_tx:    job_tx.clone(),
    });

    // ── Background download worker ────────────────────────────────────────────
    let worker_state = state.clone();
    let worker_wdir  = weights_dir.clone();
    tokio::spawn(async move {
        while let Some(job_id) = job_rx.recv().await {
            let model = {
                let guard = worker_state.job_store.read().await;
                guard.get(&job_id).map(|j| j.model)
            };
            if let Some(model) = model {
                downloader::run(
                    job_id,
                    model,
                    worker_state.job_store.clone(),
                    worker_wdir.clone(),
                    worker_state.clone(),
                ).await;
            }
        }
    });

    // ── Axum router ──────────────────────────────────────────────────────────
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/",                    get(serve_html))
        .route("/api/predict",         post(handlers::predict))
        .route("/api/info",            get(handlers::info))
        .route("/health",              get(handlers::health))
        .route("/ws",                  get(handlers::ws_handler))
        .route("/brain.obj",           get(serve_brain_mesh))
        .route("/api/download/:model", post(handlers::download_handler))
        .route("/api/jobs/:job_id",    get(handlers::job_poll_handler))
        .route("/ws/jobs/:job_id",     get(handlers::ws_job_handler))
        .nest_service("/static", ServeDir::new(&root))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{SERVER_PORT}").parse()?;
    info!(addr = %format!("http://localhost:{SERVER_PORT}"), "TRIBE server listening");
    info!(addr = %format!("ws://localhost:{SERVER_PORT}/ws"), "WebSocket ready");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── /brain.obj ────────────────────────────────────────────────────────────────

async fn serve_brain_mesh(
    axum::extract::State(st): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    use axum::{body::Body, http::StatusCode, response::IntoResponse};
    let gz_path = st.root.join("brain-surface.obj.gz");
    match tokio::fs::read(&gz_path).await {
        Ok(bytes) => axum::response::Response::builder()
            .header("content-type", "text/plain; charset=utf-8")
            .header("content-encoding", "gzip")
            .header("cache-control", "public, max-age=86400")
            .body(Body::from(bytes))
            .unwrap(),
        Err(_) => {
            warn!("brain-surface.obj.gz not found — trying brain.obj");
            let obj_path = st.root.join("brain.obj");
            match tokio::fs::read(&obj_path).await {
                Ok(bytes) => axum::response::Response::builder()
                    .header("content-type", "text/plain; charset=utf-8")
                    .body(Body::from(bytes))
                    .unwrap(),
                Err(_) => (StatusCode::NOT_FOUND, "brain mesh not found").into_response(),
            }
        }
    }
}

// ── / ─────────────────────────────────────────────────────────────────────────

async fn serve_html(
    axum::extract::State(st): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    use axum::{body::Body, http::StatusCode, response::IntoResponse};
    let path = st.root.join("tribe-v2-playground.html");
    match tokio::fs::read_to_string(&path).await {
        Ok(html) => axum::response::Response::builder()
            .header("content-type", "text/html; charset=utf-8")
            .body(Body::from(html))
            .unwrap(),
        Err(_) => (StatusCode::NOT_FOUND, "tribe-v2-playground.html not found").into_response(),
    }
}
