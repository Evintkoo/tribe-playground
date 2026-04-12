//! TRIBE v2 — pure-Rust inference server.
//!
//! On startup:
//!   1. Ensures `best.safetensors` exists (run `python3 convert_ckpt.py` if not).
//!   2. Loads FmriEncoder (candle, CPU/Metal).
//!   3. Tries to load Wav2Vec2Bert from tribe-v2-weights/ (optional).
//!   4. Tries to load LlamaTextEncoder from tribe-v2-weights/llama/ (optional).
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
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

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
use llama_encoder::LlamaTextEncoder;
use wav2vec2bert::Wav2Vec2Bert;

// ── Config ────────────────────────────────────────────────────────────────────

const SERVER_PORT: u16 = 8081;

// ── Shared application state ──────────────────────────────────────────────────

pub struct AppState {
    pub fmri:      FmriEncoder,
    pub text_enc:  Option<LlamaTextEncoder>,
    pub audio_enc: Option<Wav2Vec2Bert>,
    pub mel_spec:  Option<MelSpec>,
    pub clip_enc:  Option<ClipVisualEncoder>,
    pub root:      PathBuf,
    pub device:    Device,
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

    let root = find_root();
    let weights_dir = root.join("tribe-v2-weights");
    let st_path = weights_dir.join("best.safetensors");

    // ── Ensure converted artifacts exist ──────────────────────────────────────
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

    // ── Mel spectrogram extractor ─────────────────────────────────────────────
    let mel_spec = match MelSpec::load(&weights_dir) {
        Ok(m)  => { info!("MelSpec loaded"); Some(m) }
        Err(e) => { warn!(error = %e, "MelSpec not loaded"); None }
    };

    // ── Wav2Vec2Bert (audio encoder) ─────────────────────────────────────────
    let w2v_path = weights_dir.join("w2v-bert-2.0.safetensors");
    let audio_enc = if w2v_path.exists() {
        let p = w2v_path.to_str().context("invalid w2v path")?;
        info!("loading Wav2Vec2Bert");
        match Wav2Vec2Bert::load(p, &device) {
            Ok(enc) => { info!("Wav2Vec2Bert loaded"); Some(enc) }
            Err(e)  => { error!(error = %e, "Wav2Vec2Bert failed to load"); None }
        }
    } else {
        warn!("w2v-bert-2.0.safetensors not found — audio disabled");
        None
    };

    // ── LLaMA-3.2-3B (text encoder) ──────────────────────────────────────────
    let llama_dir = weights_dir.join("llama");
    let text_enc = if llama_dir.exists() {
        info!("loading LLaMA text encoder");
        match LlamaTextEncoder::load(&llama_dir, &device) {
            Ok(enc) => { info!("LLaMA text encoder loaded"); Some(enc) }
            Err(e)  => { error!(error = %e, "LLaMA text encoder failed to load"); None }
        }
    } else {
        warn!("tribe-v2-weights/llama/ not found — using hash-based text features");
        None
    };

    // ── CLIP ViT-L/14 (image encoder) ────────────────────────────────────────
    let clip_path = weights_dir.join("clip").join("model.safetensors");
    let clip_enc = if clip_path.exists() {
        let p = clip_path.to_str().context("invalid clip path")?;
        info!("loading CLIP ViT-L/14 image encoder");
        match ClipVisualEncoder::load(p, &device) {
            Ok(enc) => { info!("CLIP image encoder loaded"); Some(enc) }
            Err(e)  => { error!(error = %e, "CLIP image encoder failed to load"); None }
        }
    } else {
        warn!("tribe-v2-weights/clip/model.safetensors not found — image uses demo mode");
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

    let state = Arc::new(AppState {
        fmri, text_enc, audio_enc, mel_spec, clip_enc,
        root: root.clone(), device,
    });

    // ── Axum router ──────────────────────────────────────────────────────────
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);

    let app = Router::new()
        .route("/",            get(serve_html))
        .route("/api/predict", post(handlers::predict))
        .route("/api/info",    get(handlers::info))
        .route("/health",      get(handlers::health))
        .route("/ws",          get(handlers::ws_handler))
        .route("/brain.obj",   get(serve_brain_mesh))
        .nest_service("/static", ServeDir::new(&root))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("0.0.0.0:{SERVER_PORT}").parse()?;
    info!(addr = %format!("http://localhost:{SERVER_PORT}"), "listening");
    info!(addr = %format!("ws://localhost:{SERVER_PORT}/ws"), "WebSocket ready");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── /brain.obj — serves brain-surface.obj.gz with gzip encoding ──────────────
// The browser decompresses transparently; Three.js OBJLoader sees plain text.

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
            warn!("brain-surface.obj.gz not found — brain will use procedural fallback");
            (StatusCode::NOT_FOUND, "brain mesh not found").into_response()
        }
    }
}

// ── Static HTML handler ───────────────────────────────────────────────────────

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
