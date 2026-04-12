# TRIBE v2 — Rust Server Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `inference_server.py` (FastAPI, port 8081) with the existing Rust crate (`tribe-downloader/`) as the sole server, fix the response shape to return full vertex data, and add a download job queue with hot-reload so model weights can be fetched at runtime.

**Architecture:** The Axum server gains a `JobStore` (`Arc<RwLock<HashMap<Uuid,Job>>>`) and a background worker that processes download jobs sequentially. Encoder fields in `AppState` change from `Option<Enc>` to `Arc<RwLock<Option<Enc>>>` so the worker can swap them in after a download completes. Three new HTTP routes handle job creation, polling, and WebSocket streaming.

**Tech Stack:** Rust, Axum 0.7, Tokio, candle-core/candle-nn, reqwest 0.12 (streaming), uuid 1, futures-util 0.3, serde/serde_json.

---

## Context for implementers

The `tribe-downloader/` crate already compiles cleanly and implements the full TRIBE v2 pipeline in pure Rust:
- `main.rs` — Axum app, `AppState`, startup
- `handlers.rs` — HTTP + WS predict handlers
- `fmri_encoder.rs` — FmriEncoder (candle)
- `features.rs` — region_stats, temporal_pool, tribe_group_mean, demo fallbacks
- `clip_encoder.rs` — CLIP ViT-L/14
- `llama_encoder.rs` — LLaMA-3.2-3B
- `wav2vec2bert.rs` — Wav2Vec-BERT 2.0
- `audio.rs` — MelSpec + audio decoding

**Current gaps being fixed:**
1. Crate is named `tribe-downloader`, port is 8080 (HTML expects 8081)
2. `PredictResp` has `vertex_sample: Vec<f32>` (512 pts) — frontend needs `vertex_acts` (20484) + `temporal_acts` ([T][6])
3. No download endpoints — currently handled by Python scripts
4. No hot-reload — encoders are `Option<Enc>` loaded once at startup

**Working directory for all commands:** `/Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `tribe-downloader/Cargo.toml` | Modify | Rename package; add uuid + futures-util deps |
| `tribe-downloader/src/main.rs` | Modify | Port → 8081; AppState encoder fields → Arc<RwLock>; add job_store/job_tx; spawn worker; add new routes; declare new modules |
| `tribe-downloader/src/handlers.rs` | Modify | Fix PredictResp shape; update encoder reads with RwLock guards; add download_handler, job_poll_handler, ws_job_handler |
| `tribe-downloader/src/jobs.rs` | Create | ModelKind, JobStatus, Job, JobStore, enqueue() |
| `tribe-downloader/src/downloader.rs` | Create | file_list() per model; streaming download; hot-swap via AppState write locks |
| `inference_server.py` | Delete | Replaced by Rust server |
| `download_clip.py` | Delete | Replaced by POST /api/download/clip |
| `download_llama.py` | Delete | Replaced by POST /api/download/llama |
| `download_w2v.py` | Delete | Replaced by POST /api/download/wav2vec |
| `tribe-v2-playground.html` | Modify | Add downloadModel() function and per-encoder download buttons |

---

## Task 1: Rename crate, fix port, add dependencies

**Files:**
- Modify: `tribe-downloader/Cargo.toml`
- Modify: `tribe-downloader/src/main.rs:41`

- [ ] **Step 1: Update Cargo.toml**

Replace the `[package]` section and add two dependencies:

```toml
[package]
name = "tribe-server"
version = "0.1.0"
edition = "2024"

[dependencies]
# ML framework (pure Rust, no libtorch required)
candle-core = { version = "0.8", features = ["metal"] }
candle-nn   = "0.8"
safetensors = "0.4"

# HuggingFace model hub
hf-hub = { version = "0.3", features = ["tokio"] }

# Async runtime + HTTP
tokio        = { version = "1",    features = ["full"] }
axum         = { version = "0.7", features = ["ws"] }
tower-http   = { version = "0.5",  features = ["fs", "cors"] }
reqwest      = { version = "0.12", features = ["json", "stream"] }

# Serialisation
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
anyhow       = "1"

# Job IDs
uuid         = { version = "1", features = ["v4", "serde"] }

# Streaming download helper
futures-util = "0.3"

# Audio decoding (MP3, WAV, FLAC, OGG, M4A …)
symphonia = { version = "0.5", features = ["all-codecs", "all-formats"] }

# FFT for mel-spectrogram
rustfft = "6"

# Base64 (audio upload)
base64 = "0.22"

# Hash-based demo text features (fallback)
sha2 = "0.10"

# LLaMA-3.2-3B text encoder
tokenizers = { version = "0.19", default-features = false, features = ["onig"] }

# Image decoding for CLIP encoder
image = { version = "0.25", default-features = false, features = ["jpeg", "png", "webp", "bmp", "gif"] }

# Logging
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

- [ ] **Step 2: Change port constant in main.rs**

Find line 41 in `src/main.rs`:
```rust
const SERVER_PORT: u16 = 8080;
```

Change to:
```rust
const SERVER_PORT: u16 = 8081;
```

- [ ] **Step 3: Verify it builds**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader
cargo build 2>&1 | tail -5
```

Expected:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in Xs
```

- [ ] **Step 4: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-downloader/Cargo.toml tribe-downloader/src/main.rs
git commit -m "chore: rename crate to tribe-server, port 8081, add uuid + futures-util"
```

---

## Task 2: Create `src/jobs.rs`

**Files:**
- Create: `tribe-downloader/src/jobs.rs`
- Modify: `tribe-downloader/src/main.rs` (add `mod jobs;`)

- [ ] **Step 1: Write the test**

At the bottom of `tribe-downloader/src/jobs.rs` (the new file), after the implementation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_returns_uuid_and_rejects_duplicate() {
        let store = new_job_store();
        let (tx, _rx) = tokio::sync::mpsc::channel::<uuid::Uuid>(8);

        let id = enqueue(&store, &tx, ModelKind::Clip).await.unwrap();
        assert!(!id.is_nil());

        // Second enqueue for same model while first is Queued → Err
        let err = enqueue(&store, &tx, ModelKind::Clip).await;
        assert!(err.is_err());

        // Different model is fine
        let id2 = enqueue(&store, &tx, ModelKind::Llama).await.unwrap();
        assert_ne!(id, id2);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader
cargo test jobs 2>&1 | tail -10
```

Expected: FAIL — `jobs` module doesn't exist yet.

- [ ] **Step 3: Implement `src/jobs.rs`**

Create `tribe-downloader/src/jobs.rs` with full contents:

```rust
//! jobs.rs — Download job queue for TRIBE v2.
use std::{collections::HashMap, sync::Arc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

// ── ModelKind ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Clip,
    Llama,
    Wav2Vec,
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Clip    => "clip",
            Self::Llama   => "llama",
            Self::Wav2Vec => "wav2vec",
        })
    }
}

impl std::str::FromStr for ModelKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "clip"    => Ok(Self::Clip),
            "llama"   => Ok(Self::Llama),
            "wav2vec" => Ok(Self::Wav2Vec),
            other     => Err(format!("unknown model '{other}'; use clip, llama, or wav2vec")),
        }
    }
}

// ── JobStatus ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "PascalCase")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
}

// ── Job ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Job {
    pub id:          Uuid,
    pub model:       ModelKind,
    pub status:      JobStatus,
    pub pct:         f32,
    pub bytes_done:  u64,
    pub total_bytes: Option<u64>,
    pub error:       Option<String>,
}

// ── JobStore ──────────────────────────────────────────────────────────────────

pub type JobStore = Arc<RwLock<HashMap<Uuid, Job>>>;

pub fn new_job_store() -> JobStore {
    Arc::new(RwLock::new(HashMap::new()))
}

// ── enqueue ───────────────────────────────────────────────────────────────────

/// Insert a new job for `model` and send its ID to the worker channel.
/// Returns `Err` if a Queued or Running job already exists for that model.
pub async fn enqueue(
    store: &JobStore,
    tx:    &tokio::sync::mpsc::Sender<Uuid>,
    model: ModelKind,
) -> Result<Uuid, String> {
    let mut guard = store.write().await;

    for job in guard.values() {
        if job.model == model && matches!(job.status, JobStatus::Queued | JobStatus::Running) {
            return Err(format!("job for {model} already active (id={})", job.id));
        }
    }

    let id = Uuid::new_v4();
    guard.insert(id, Job {
        id,
        model,
        status:      JobStatus::Queued,
        pct:         0.0,
        bytes_done:  0,
        total_bytes: None,
        error:       None,
    });
    drop(guard);

    tx.send(id).await.map_err(|e| e.to_string())?;
    Ok(id)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_returns_uuid_and_rejects_duplicate() {
        let store = new_job_store();
        let (tx, _rx) = tokio::sync::mpsc::channel::<Uuid>(8);

        let id = enqueue(&store, &tx, ModelKind::Clip).await.unwrap();
        assert!(!id.is_nil());

        // Second enqueue for same model while first is Queued → Err
        let err = enqueue(&store, &tx, ModelKind::Clip).await;
        assert!(err.is_err());

        // Different model is fine
        let id2 = enqueue(&store, &tx, ModelKind::Llama).await.unwrap();
        assert_ne!(id, id2);
    }
}
```

- [ ] **Step 4: Add `mod jobs;` to main.rs**

In `tribe-downloader/src/main.rs`, after the existing `mod` declarations (around line 26):

```rust
mod audio;
mod clip_encoder;
mod features;
mod fmri_encoder;
mod handlers;
mod jobs;          // ← add this
mod llama_encoder;
mod wav2vec2bert;
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader
cargo test jobs 2>&1 | tail -10
```

Expected:
```
test jobs::tests::enqueue_returns_uuid_and_rejects_duplicate ... ok
test result: ok. 1 passed; 0 failed
```

- [ ] **Step 6: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-downloader/src/jobs.rs tribe-downloader/src/main.rs
git commit -m "feat: jobs.rs — ModelKind, JobStatus, Job, JobStore, enqueue()"
```

---

## Task 3: Create `src/downloader.rs`

**Files:**
- Create: `tribe-downloader/src/downloader.rs`
- Modify: `tribe-downloader/src/main.rs` (add `mod downloader;`)

`AppState` will be updated in Task 4. For now the `run()` function signature references `AppState` but `AppState` fields are still `Option<Enc>`. Task 3 only compiles because `run()` is written to accept the final `AppState` shape — the compiler won't see it until Task 4 wires everything together. Write the file in full here; it will compile cleanly after Task 4.

- [ ] **Step 1: Create `src/downloader.rs`**

```rust
//! downloader.rs — Streaming HuggingFace downloads with progress + hot-swap.
use std::{path::{Path, PathBuf}, sync::Arc};
use anyhow::{Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    jobs::{Job, JobStatus, JobStore, ModelKind},
    AppState,
};

// ── File lists per model ──────────────────────────────────────────────────────

/// (path relative to tribe-v2-weights/, HuggingFace URL)
fn file_list(model: ModelKind) -> Vec<(&'static str, &'static str)> {
    match model {
        ModelKind::Clip => vec![
            (
                "clip/model.safetensors",
                "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors",
            ),
        ],
        ModelKind::Llama => vec![
            (
                "llama/tokenizer.json",
                "https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B/resolve/main/tokenizer.json",
            ),
            (
                "llama/model-00001-of-00002.safetensors",
                "https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B/resolve/main/model-00001-of-00002.safetensors",
            ),
            (
                "llama/model-00002-of-00002.safetensors",
                "https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B/resolve/main/model-00002-of-00002.safetensors",
            ),
        ],
        ModelKind::Wav2Vec => vec![
            (
                "w2v-bert-2.0.safetensors",
                "https://huggingface.co/facebook/w2v-bert-2.0/resolve/main/model.safetensors",
            ),
        ],
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

/// Run one download job. Called by the background worker (Task 4).
/// Updates job progress in `store` as bytes arrive.
/// After all files land, hot-swaps the encoder into `state`.
pub async fn run(
    job_id:      Uuid,
    model:       ModelKind,
    store:       JobStore,
    weights_dir: PathBuf,
    state:       Arc<AppState>,
) {
    set_status(&store, job_id, JobStatus::Running, 0.0, 0, None).await;

    let client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .timeout(std::time::Duration::from_secs(7200))
        .build()
    {
        Ok(c)  => c,
        Err(e) => { mark_failed(&store, job_id, format!("build client: {e}")).await; return; }
    };

    let files = file_list(model);
    let mut overall_total: u64 = 0;
    let mut overall_done:  u64 = 0;

    for (rel_path, url) in &files {
        let dest = weights_dir.join(rel_path);

        // Skip already-complete files
        let existing_size = dest.metadata().map(|m| m.len()).unwrap_or(0);
        if dest.exists() && existing_size > 1024 {
            info!(%rel_path, bytes = existing_size, "already downloaded, skipping");
            continue;
        }

        // Create parent directory
        if let Some(parent) = dest.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                mark_failed(&store, job_id, format!("mkdir {parent:?}: {e}")).await;
                return;
            }
        }

        // Start streaming GET
        let resp = match client.get(*url).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                mark_failed(&store, job_id, format!("HTTP {} for {url}", r.status())).await;
                return;
            }
            Err(e) => {
                mark_failed(&store, job_id, format!("GET {url}: {e}")).await;
                return;
            }
        };

        if let Some(n) = resp.content_length() {
            overall_total += n;
        }

        // Write to temp file, then rename (atomic on same filesystem)
        let tmp = dest.with_extension("_downloading");
        let mut file = match tokio::fs::File::create(&tmp).await {
            Ok(f)  => f,
            Err(e) => { mark_failed(&store, job_id, format!("create {tmp:?}: {e}")).await; return; }
        };

        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c)  => c,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    mark_failed(&store, job_id, format!("read stream: {e}")).await;
                    return;
                }
            };
            if let Err(e) = file.write_all(&chunk).await {
                let _ = tokio::fs::remove_file(&tmp).await;
                mark_failed(&store, job_id, format!("write to disk: {e}")).await;
                return;
            }
            overall_done += chunk.len() as u64;
            let pct = if overall_total > 0 {
                (overall_done as f32 / overall_total as f32 * 99.0).min(99.0)
            } else {
                0.0
            };
            let total_opt = if overall_total > 0 { Some(overall_total) } else { None };
            set_status(&store, job_id, JobStatus::Running, pct, overall_done, total_opt).await;
        }

        drop(file); // flush + close before rename
        if let Err(e) = tokio::fs::rename(&tmp, &dest).await {
            mark_failed(&store, job_id, format!("rename {tmp:?} → {dest:?}: {e}")).await;
            return;
        }
        info!(%rel_path, "file complete");
    }

    // ── Hot-swap encoder ──────────────────────────────────────────────────────
    if let Err(e) = hot_swap(model, &weights_dir, &state).await {
        warn!(error = %e, "hot-swap failed — encoder unavailable until restart");
    }

    // ── Mark done ─────────────────────────────────────────────────────────────
    {
        let mut guard = store.write().await;
        if let Some(job) = guard.get_mut(&job_id) {
            job.status      = JobStatus::Done;
            job.pct         = 100.0;
            job.bytes_done  = overall_done;
        }
    }
    info!(%model, "download job complete");
}

// ── Hot-swap helpers ──────────────────────────────────────────────────────────

async fn hot_swap(model: ModelKind, weights_dir: &Path, state: &AppState) -> Result<()> {
    match model {
        ModelKind::Clip => {
            let path = weights_dir.join("clip/model.safetensors");
            let enc = crate::clip_encoder::ClipVisualEncoder::load(
                path.to_str().context("invalid clip path")?,
                &state.device,
            )?;
            *state.clip_enc.write().await = Some(enc);
            info!("CLIP encoder hot-swapped ✓");
        }
        ModelKind::Llama => {
            let dir = weights_dir.join("llama");
            let enc = crate::llama_encoder::LlamaTextEncoder::load(&dir, &state.device)?;
            *state.text_enc.write().await = Some(enc);
            info!("LLaMA text encoder hot-swapped ✓");
        }
        ModelKind::Wav2Vec => {
            let path = weights_dir.join("w2v-bert-2.0.safetensors");
            let enc = crate::wav2vec2bert::Wav2Vec2Bert::load(
                path.to_str().context("invalid wav2vec path")?,
                &state.device,
            )?;
            *state.audio_enc.write().await = Some(enc);
            info!("Wav2Vec2Bert hot-swapped ✓");
        }
    }
    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

async fn set_status(
    store:       &JobStore,
    id:          Uuid,
    status:      JobStatus,
    pct:         f32,
    bytes_done:  u64,
    total_bytes: Option<u64>,
) {
    let mut guard = store.write().await;
    if let Some(job) = guard.get_mut(&id) {
        job.status      = status;
        job.pct         = pct;
        job.bytes_done  = bytes_done;
        job.total_bytes = total_bytes;
    }
}

async fn mark_failed(store: &JobStore, id: Uuid, msg: String) {
    error!(%msg, "download job failed");
    let mut guard = store.write().await;
    if let Some(job) = guard.get_mut(&id) {
        job.status = JobStatus::Failed;
        job.error  = Some(msg);
    }
}
```

- [ ] **Step 2: Add `mod downloader;` to main.rs**

In `tribe-downloader/src/main.rs` after the existing mod declarations:

```rust
mod audio;
mod clip_encoder;
mod downloader;    // ← add this
mod features;
mod fmri_encoder;
mod handlers;
mod jobs;
mod llama_encoder;
mod wav2vec2bert;
```

- [ ] **Step 3: Commit (will compile fully after Task 4)**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-downloader/src/downloader.rs tribe-downloader/src/main.rs
git commit -m "feat: downloader.rs — streaming HuggingFace download + hot-swap"
```

---

## Task 4: Update `AppState`, `handlers.rs`, and wire everything in `main.rs`

**Files:**
- Modify: `tribe-downloader/src/main.rs`
- Modify: `tribe-downloader/src/handlers.rs`

This task changes `AppState` encoder fields to `Arc<RwLock<Option<Enc>>>`, adds job infrastructure, fixes the response shape, updates all handler code to use read locks, adds three new handlers, and spawns the background worker. All changes must land together — the compiler will reject a partial state.

### Section A: Fix `handlers.rs` — response shape + RwLock reads + new handlers

- [ ] **Step 1: Replace `handlers.rs` completely**

Replace the full contents of `tribe-downloader/src/handlers.rs` with:

```rust
//! handlers.rs — Axum HTTP + WebSocket handlers for TRIBE v2.
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use candle_core::{DType, Device, Tensor};

use crate::{
    audio::{decode_audio, MelSpec},
    features::{
        region_stats, temporal_pool, text_to_features_demo, tribe_group_mean,
        visual_features_clip, visual_features_demo, REGION_RANGES,
    },
    jobs::{self, JobStatus},
    wav2vec2bert::Wav2Vec2Bert,
    AppState,
};

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct PredictReq {
    pub text:       Option<String>,
    pub audio_b64:  Option<String>,
    pub image_b64:  Option<String>,
    pub video_b64:  Option<String>,
    pub seq_len:    Option<usize>,
}

#[derive(Serialize, Clone)]
pub struct PredictResp {
    pub region_stats:  std::collections::HashMap<String, crate::features::RegionStat>,
    pub global_stats:  crate::features::GlobalStat,
    pub vertex_acts:   Vec<f32>,        // 20484 floats — mean over T
    pub temporal_acts: Vec<Vec<f32>>,   // [T][6] regional means per timepoint
    pub seq_len:       usize,
    pub modality:      String,
    pub elapsed_ms:    f64,
    pub demo_mode:     bool,
}

// ── Shared prediction logic ───────────────────────────────────────────────────

async fn run_prediction<F>(
    req: PredictReq,
    st:  Arc<AppState>,
    mut progress: F,
) -> Result<PredictResp, String>
where
    F: FnMut(u8, &str),
{
    let t0 = Instant::now();

    let text      = req.text.as_deref().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
    let audio_b64 = req.audio_b64.filter(|s| !s.is_empty());
    let image_b64 = req.image_b64.filter(|s| !s.is_empty());
    let video_b64 = req.video_b64.filter(|s| !s.is_empty());

    if text.is_none() && audio_b64.is_none() && image_b64.is_none() && video_b64.is_none() {
        return Err("Provide 'text', 'audio_b64', 'image_b64', or 'video_b64'".into());
    }

    let seq    = req.seq_len.unwrap_or(16).clamp(1, 100);
    let device = &st.device;
    let mut demo_mode  = false;
    let mut modalities: Vec<&str> = Vec::new();

    // ── Text features ─────────────────────────────────────────────────────────
    let text_feat: Option<Tensor> = if let Some(ref txt) = text {
        modalities.push("text");
        progress(15, "Encoding text…");
        let guard = st.text_enc.read().await;
        if let Some(ref enc) = *guard {
            enc.encode(txt, seq).map(Some).map_err(|e| format!("LLaMA: {e}"))?
        } else {
            demo_mode = true;
            drop(guard);
            text_to_features_demo(txt, seq, device).map(Some).map_err(|e| e.to_string())?
        }
    } else {
        None
    };

    // ── Audio features ────────────────────────────────────────────────────────
    let audio_feat: Option<Tensor> = if let Some(ref b64) = audio_b64 {
        modalities.push("audio");
        progress(25, "Encoding audio…");
        let guard_enc = st.audio_enc.read().await;
        match encode_audio(b64, seq, &*guard_enc, &st.mel_spec, device) {
            Ok(Some(t)) => Some(t),
            Ok(None)    => { demo_mode = true; None }
            Err(e)      => return Err(format!("audio: {e}")),
        }
    } else {
        None
    };

    // ── Visual features (image / video) ──────────────────────────────────────
    let is_video_modality = video_b64.is_some();
    let visual_feat: Option<Tensor> = if let Some(ref b64) = image_b64.or(video_b64) {
        modalities.push(if is_video_modality { "video" } else { "image" });
        progress(35, if is_video_modality { "Encoding video…" } else { "Encoding image…" });
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
            .map_err(|e| format!("base64 decode: {e}"))?;

        if !is_video_modality {
            let guard_clip = st.clip_enc.read().await;
            if let Some(ref clip) = *guard_clip {
                visual_features_clip(&bytes, seq, clip).map(Some).map_err(|e| format!("CLIP: {e}"))?
            } else {
                demo_mode = true;
                drop(guard_clip);
                visual_features_demo(&bytes, seq, device).map(Some).map_err(|e| format!("visual demo: {e}"))?
            }
        } else {
            demo_mode = true;
            visual_features_demo(&bytes, seq, device).map(Some).map_err(|e| format!("visual demo: {e}"))?
        }
    } else {
        None
    };

    if text_feat.is_none() && audio_feat.is_none() && visual_feat.is_none() {
        return Err("Feature extraction produced no tensors".into());
    }

    // ── FmriEncoder forward ───────────────────────────────────────────────────
    progress(55, "Running FmriEncoder…");
    let out = st.fmri.forward(text_feat.as_ref(), audio_feat.as_ref(), visual_feat.as_ref())
        .map_err(|e| e.to_string())?;

    // out: [1, T, N_VERT]
    progress(85, "Computing region statistics…");
    let act_flat: Vec<f32> = out.squeeze(0)
        .and_then(|o| o.to_vec2::<f32>())
        .map(|rows| rows.into_iter().flatten().collect())
        .map_err(|e| e.to_string())?;

    let n_vert = crate::fmri_encoder::N_VERT;
    let n_time = act_flat.len() / n_vert;

    let (rstats, gstats) = region_stats(&act_flat, n_vert);

    // Mean over T for per-vertex BOLD coloring
    let vertex_acts: Vec<f32> = (0..n_vert)
        .map(|vi| act_flat.iter().skip(vi).step_by(n_vert).sum::<f32>() / n_time as f32)
        .collect();

    // [T][6] regional means — order matches REGION_RANGES
    let temporal_acts: Vec<Vec<f32>> = (0..n_time)
        .map(|ti| {
            REGION_RANGES.iter()
                .map(|&(_, lo, hi)| {
                    let hi = hi.min(n_vert);
                    let s  = &act_flat[ti * n_vert + lo..ti * n_vert + hi];
                    s.iter().sum::<f32>() / (hi - lo) as f32
                })
                .collect()
        })
        .collect();

    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    progress(100, "Done");

    Ok(PredictResp {
        region_stats: rstats,
        global_stats: gstats,
        vertex_acts,
        temporal_acts,
        seq_len:    seq,
        modality:   modalities.join("+"),
        elapsed_ms: (elapsed * 10.0).round() / 10.0,
        demo_mode,
    })
}

// ── /api/predict (HTTP) ───────────────────────────────────────────────────────

pub async fn predict(
    State(st): State<Arc<AppState>>,
    Json(req): Json<PredictReq>,
) -> Response {
    match run_prediction(req, st, |_, _| {}).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e)   => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

// ── /ws (WebSocket predict) ───────────────────────────────────────────────────

pub async fn ws_handler(
    ws:           WebSocketUpgrade,
    State(st):    State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, st))
}

async fn handle_ws(mut socket: WebSocket, st: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
                let req: PredictReq = match serde_json::from_str(&text) {
                    Ok(r)  => r,
                    Err(e) => {
                        let _ = socket.send(Message::Text(
                            json!({"type":"error","message": e.to_string()}).to_string()
                        )).await;
                        continue;
                    }
                };

                let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
                let st2  = st.clone();
                let tx2  = tx.clone();

                tokio::spawn(async move {
                    let result = run_prediction(req, st2, move |pct, msg| {
                        let _ = tx2.try_send(
                            json!({"type":"progress","pct":pct,"msg":msg}).to_string()
                        );
                    }).await;
                    match result {
                        Ok(resp) => {
                            let mut obj = serde_json::to_value(resp).unwrap_or_default();
                            if let Some(m) = obj.as_object_mut() {
                                m.insert("type".into(), json!("result"));
                            }
                            let _ = tx.send(obj.to_string()).await;
                        }
                        Err(e) => {
                            let _ = tx.send(
                                json!({"type":"error","message":e}).to_string()
                            ).await;
                        }
                    }
                });

                while let Some(msg) = rx.recv().await {
                    if socket.send(Message::Text(msg)).await.is_err() {
                        return;
                    }
                }
            }
            Message::Close(_) => break,
            Message::Ping(d)  => { let _ = socket.send(Message::Pong(d)).await; }
            _ => {}
        }
    }
}

// ── /health ───────────────────────────────────────────────────────────────────

pub async fn health(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    let text_loaded  = st.text_enc.read().await.is_some();
    let audio_loaded = st.audio_enc.read().await.is_some();
    let clip_loaded  = st.clip_enc.read().await.is_some();
    Json(json!({
        "ready":     true,
        "text_enc":  text_loaded,
        "audio_enc": audio_loaded,
        "clip_enc":  clip_loaded,
    }))
}

// ── /api/info ─────────────────────────────────────────────────────────────────

pub async fn info(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    let text_loaded  = st.text_enc.read().await.is_some();
    let audio_loaded = st.audio_enc.read().await.is_some();
    let clip_loaded  = st.clip_enc.read().await.is_some();
    Json(json!({
        "model_class": "FmriEncoder",
        "n_params":    st.fmri.n_params(),
        "checkpoint":  "facebook/tribev2 · best.safetensors",
        "architecture": {
            "hidden_dim":  1152,
            "n_heads":     8,
            "n_layers":    16,
            "ff_mult":     4,
            "n_vertices":  20484,
            "max_seq_len": 1024,
            "surface":     "fsaverage5"
        },
        "feature_dims": {
            "text":  { "total": 6144, "encoder": "LLaMA-3.2-3B" },
            "audio": { "total": 2048, "encoder": "Wav2Vec-BERT 2.0" },
            "video": { "total": 2816, "encoder": "V-JEPA2 ViT-G" }
        },
        "encoders": {
            "text":  if text_loaded  { "LLaMA-3.2-3B · loaded"    } else { "not loaded" },
            "audio": if audio_loaded { "Wav2Vec-BERT 2.0 · loaded" } else { "not loaded" },
            "image": if clip_loaded  { "CLIP ViT-L/14 · loaded"    } else { "not loaded" },
            "video": "not loaded (V-JEPA2 not integrated)",
        },
        "runtime":   "Rust/candle (no Python)",
        "demo_mode": !text_loaded || !clip_loaded,
    }))
}

// ── POST /api/download/:model ─────────────────────────────────────────────────

pub async fn download_handler(
    Path(model_str): Path<String>,
    State(st):       State<Arc<AppState>>,
) -> Response {
    let model = match model_str.parse::<crate::jobs::ModelKind>() {
        Ok(m)  => m,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    match jobs::enqueue(&st.job_store, &st.job_tx, model).await {
        Ok(id)  => Json(json!({ "job_id": id })).into_response(),
        Err(e)  => (StatusCode::CONFLICT, e).into_response(),
    }
}

// ── GET /api/jobs/:job_id ─────────────────────────────────────────────────────

pub async fn job_poll_handler(
    Path(job_id): Path<Uuid>,
    State(st):    State<Arc<AppState>>,
) -> Response {
    let guard = st.job_store.read().await;
    match guard.get(&job_id) {
        Some(job) => Json(job.clone()).into_response(),
        None      => (StatusCode::NOT_FOUND, "job not found").into_response(),
    }
}

// ── GET /ws/jobs/:job_id ──────────────────────────────────────────────────────

pub async fn ws_job_handler(
    ws:           WebSocketUpgrade,
    Path(job_id): Path<Uuid>,
    State(st):    State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_job(socket, job_id, st))
}

async fn handle_ws_job(mut socket: WebSocket, job_id: Uuid, st: Arc<AppState>) {
    loop {
        let job = {
            let guard = st.job_store.read().await;
            guard.get(&job_id).cloned()
        };

        match job {
            None => {
                let _ = socket.send(Message::Text(
                    json!({"type":"error","message":"job not found"}).to_string()
                )).await;
                break;
            }
            Some(j) => {
                let progress_msg = json!({
                    "type":        "progress",
                    "pct":         j.pct,
                    "bytes_done":  j.bytes_done,
                    "total_bytes": j.total_bytes,
                });
                if socket.send(Message::Text(progress_msg.to_string())).await.is_err() {
                    break;
                }

                match j.status {
                    JobStatus::Done => {
                        let _ = socket.send(Message::Text(
                            json!({"type":"done","model": j.model}).to_string()
                        )).await;
                        break;
                    }
                    JobStatus::Failed => {
                        let _ = socket.send(Message::Text(
                            json!({"type":"error","message": j.error.unwrap_or_default()}).to_string()
                        )).await;
                        break;
                    }
                    _ => {}
                }
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

// ── Internal: audio encoding ──────────────────────────────────────────────────

fn encode_audio(
    b64:     &str,
    seq_len: usize,
    enc:     &Option<crate::wav2vec2bert::Wav2Vec2Bert>,
    mel:     &Option<crate::audio::MelSpec>,
    device:  &Device,
) -> anyhow::Result<Option<Tensor>> {
    let audio_enc = match enc { Some(e) => e, None => return Ok(None) };
    let mel_spec  = match mel { Some(m) => m, None => return Ok(None) };

    let bytes    = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)?;
    let waveform = decode_audio(&bytes)?;
    let mel_frames = mel_spec.compute(&waveform);
    if mel_frames.is_empty() { return Ok(None); }

    let t_mel = mel_frames.len();
    let flat: Vec<f32> = mel_frames.into_iter().flatten().collect();
    let feats = Tensor::from_vec(flat, (1, t_mel, 160), device)?.to_dtype(DType::F32)?;

    let [h12, h18, h24] = audio_enc.encode(&feats)?;
    let h12 = h12.squeeze(0)?;
    let h18 = h18.squeeze(0)?;
    let h24 = h24.squeeze(0)?;
    let feats_2048 = tribe_group_mean(&h12, &h18, &h24)?;
    let pooled = temporal_pool(&feats_2048, seq_len)?;
    Ok(Some(pooled.unsqueeze(0)?))
}
```

### Section B: Update `main.rs` — AppState + worker + new routes

- [ ] **Step 2: Replace `src/main.rs` completely**

Replace the full contents of `tribe-downloader/src/main.rs` with:

```rust
//! TRIBE v2 — pure-Rust inference server (tribe-server).
//!
//! On startup:
//!   1. Ensures `best.safetensors` exists (run `python3 convert_ckpt.py` if not).
//!   2. Loads FmriEncoder (candle, CPU/Metal).
//!   3. Tries to load Wav2Vec2Bert, LlamaTextEncoder, ClipVisualEncoder (all optional).
//!   4. Spawns a background download worker (processes job queue one-at-a-time).
//!   5. Starts Axum HTTP server on port 8081 with WebSocket at /ws.

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
use jobs::{JobStore, ModelKind};
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
```

- [ ] **Step 3: Build release**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader
cargo build --release 2>&1 | tail -8
```

Expected:
```
Compiling tribe-server v0.1.0 (…/tribe-downloader)
Finished `release` profile [optimized] target(s) in Xs
```

If there are compile errors, fix them before committing. Common issues:
- Missing import: add `use crate::features::REGION_RANGES;` to handlers.rs imports
- `encode_audio` parameter type mismatch: the second encoder parameter is now `&Option<Wav2Vec2Bert>` (not `&Option<Arc<RwLock<…>>>`); the read guard dereferences to `Option<…>` already via `&*guard_enc`

- [ ] **Step 4: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-downloader/src/main.rs tribe-downloader/src/handlers.rs
git commit -m "feat: Arc<RwLock> hot-reload, vertex_acts+temporal_acts response, download job routes"
```

---

## Task 5: Delete Python files

**Files:**
- Delete: `inference_server.py`
- Delete: `download_clip.py`
- Delete: `download_llama.py`
- Delete: `download_w2v.py`

- [ ] **Step 1: Kill the Python server if running**

```bash
pkill -f inference_server.py 2>/dev/null; echo "stopped"
```

- [ ] **Step 2: Delete the files**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
rm inference_server.py download_clip.py download_llama.py download_w2v.py
```

- [ ] **Step 3: Verify deletion**

```bash
ls *.py
```

Expected: only `convert_ckpt.py` and `convert_brain.py` remain (these are still needed by the Rust server's startup check).

- [ ] **Step 4: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add -u
git commit -m "chore: delete Python server and download scripts (replaced by Rust)"
```

---

## Task 6: Frontend download UI

**Files:**
- Modify: `tribe-v2-playground.html`

Add a `downloadModel(model)` JavaScript function and per-encoder download buttons that appear near the encoder status dots. Each button POSTs to `/api/download/{model}`, then opens a WebSocket to `/ws/jobs/{job_id}` and shows a small progress bar.

- [ ] **Step 1: Add CSS for the download progress bar**

In the `<style>` block, before `</style>`, add:

```css
/* ── DOWNLOAD PROGRESS ───────────────────────────────────────── */
.dl-btn {
  font-size: 9px; padding: 2px 6px; border-radius: 3px;
  border: 1px solid var(--b1); background: var(--s1); color: var(--t2);
  cursor: pointer; margin-left: 6px; vertical-align: middle;
}
.dl-btn:hover { background: var(--ac); color: #fff; }
.dl-btn:disabled { opacity: 0.45; cursor: default; }
.dl-bar-wrap {
  display: inline-block; width: 70px; height: 4px;
  background: var(--b1); border-radius: 2px; margin-left: 6px;
  vertical-align: middle; overflow: hidden;
}
.dl-bar { height: 100%; width: 0%; background: var(--ac); transition: width 0.3s; }
```

- [ ] **Step 2: Add download buttons to the image panel meta bar**

Find in the image panel (search for `id="imageEncoderLabel"`):

```html
<span class="meta-item">encoder: <span id="imageEncoderLabel">CLIP ViT-L/14</span><span id="imgEncDot" title="checking…" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--t3);margin-left:5px;vertical-align:middle"></span></span>
```

Replace with:

```html
<span class="meta-item">encoder: <span id="imageEncoderLabel">CLIP ViT-L/14</span><span id="imgEncDot" title="checking…" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--t3);margin-left:5px;vertical-align:middle"></span><button class="dl-btn" id="dlBtn-clip" onclick="downloadModel('clip')" title="Download CLIP weights (~1.7 GB)">↓ Download</button><span class="dl-bar-wrap" id="dlBar-clip" style="display:none"><span class="dl-bar" id="dlBarFill-clip"></span></span></span>
```

- [ ] **Step 3: Add download buttons for text and audio encoders**

Find in the main `<script>` block, inside `fetchModelInfo()`, the section that updates encoder labels. After the existing encoder dot/label updates, add the following to show/hide the download button based on encoder state. But first, we need to add the download buttons to the HTML. Find the audio panel meta bar (search for `meta-item` near audio) or add a dedicated "Encoders" row in the status panel.

Add to the info panel (wherever `#imageEncoderLabel` is rendered, look for the `.info-table` or similar structure) these two rows. Search for `<div class="info-row">` patterns and insert after the last one visible in the server info section:

```html
<div class="info-row" id="rowDlLlama" style="display:none">
  <span class="info-lbl">LLaMA encoder</span>
  <span class="info-val">not loaded
    <button class="dl-btn" id="dlBtn-llama" onclick="downloadModel('llama')" title="Download LLaMA weights (~6 GB)">↓ Download</button>
    <span class="dl-bar-wrap" id="dlBar-llama" style="display:none"><span class="dl-bar" id="dlBarFill-llama"></span></span>
  </span>
</div>
<div class="info-row" id="rowDlWav2vec" style="display:none">
  <span class="info-lbl">Wav2Vec encoder</span>
  <span class="info-val">not loaded
    <button class="dl-btn" id="dlBtn-wav2vec" onclick="downloadModel('wav2vec')" title="Download Wav2Vec weights (~0.6 GB)">↓ Download</button>
    <span class="dl-bar-wrap" id="dlBar-wav2vec" style="display:none"><span class="dl-bar" id="dlBarFill-wav2vec"></span></span>
  </span>
</div>
```

- [ ] **Step 4: Add `downloadModel()` to the main script**

In the main `<script>` block, after `showToast()`, add:

```javascript
async function downloadModel(model) {
  const btn     = document.getElementById('dlBtn-' + model);
  const barWrap = document.getElementById('dlBar-' + model);
  const barFill = document.getElementById('dlBarFill-' + model);

  if (btn) btn.disabled = true;
  if (barWrap) barWrap.style.display = 'inline-block';
  showToast('Starting download: ' + model + '…', 4000);

  let jobId;
  try {
    const r = await fetch('/api/download/' + model, { method: 'POST' });
    if (r.status === 409) {
      showToast(model + ' download already in progress');
      if (btn) btn.disabled = false;
      return;
    }
    if (!r.ok) { throw new Error(await r.text()); }
    const d = await r.json();
    jobId = d.job_id;
  } catch (e) {
    showToast('Download failed: ' + e.message, 5000);
    if (btn) btn.disabled = false;
    if (barWrap) barWrap.style.display = 'none';
    return;
  }

  // Stream progress via WebSocket
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(proto + '://' + location.host + '/ws/jobs/' + jobId);

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'progress') {
      if (barFill) barFill.style.width = (msg.pct || 0).toFixed(1) + '%';
    } else if (msg.type === 'done') {
      showToast('✓ ' + model + ' downloaded — encoder now active', 5000);
      if (barFill) barFill.style.width = '100%';
      setTimeout(() => {
        if (barWrap) barWrap.style.display = 'none';
        if (btn) { btn.textContent = '✓ loaded'; btn.disabled = true; }
        fetchModelInfo();   // refresh encoder status dots
      }, 1200);
    } else if (msg.type === 'error') {
      showToast('Download error: ' + msg.message, 6000);
      if (btn) btn.disabled = false;
      if (barWrap) barWrap.style.display = 'none';
    }
  };

  ws.onerror = () => {
    showToast('WebSocket error tracking ' + model + ' download', 4000);
    if (btn) btn.disabled = false;
  };
}
```

- [ ] **Step 5: Show/hide download buttons in `fetchModelInfo()`**

In `fetchModelInfo()`, after the existing encoder dot update logic, add:

```javascript
// Show download buttons for encoders that are not loaded
const showDl = (model, rowId, loaded) => {
  const btn = document.getElementById('dlBtn-' + model);
  const row = document.getElementById(rowId);
  if (btn) btn.style.display = loaded ? 'none' : '';
  if (row) row.style.display = loaded ? 'none' : '';
};

const clipLoaded = d.encoders?.image?.includes('loaded') || false;
const textLoaded = d.encoders?.text?.includes('loaded')  || false;
const audLoaded  = d.encoders?.audio?.includes('loaded') || false;

showDl('clip',    null,          clipLoaded);
showDl('llama',   'rowDlLlama',   textLoaded);
showDl('wav2vec', 'rowDlWav2vec', audLoaded);
```

- [ ] **Step 6: Commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add tribe-v2-playground.html
git commit -m "feat: download buttons + progress bar UI for clip, llama, wav2vec encoders"
```

---

## Task 7: End-to-end verification

- [ ] **Step 1: Start the Rust server**

```bash
pkill -f tribe-server 2>/dev/null
cd /Users/evintleovonzko/Documents/research/tribe-playground/tribe-downloader
./target/release/tribe-server &
```

Wait for the log line: `TRIBE server listening` (typically 3–10 seconds for model load).

- [ ] **Step 2: Smoke-test core routes**

```bash
curl -s http://127.0.0.1:8081/health | python3 -m json.tool
curl -s http://127.0.0.1:8081/ | head -3
curl -s -o /dev/null -w "brain.obj: %{http_code}\n" http://127.0.0.1:8081/brain.obj
curl -s http://127.0.0.1:8081/api/info | python3 -c "import sys,json; d=json.load(sys.stdin); print('runtime:', d.get('runtime')); print('encoders:', d.get('encoders'))"
```

Expected:
```
{"ready": true, ...}
<!DOCTYPE html>
<html lang="en">
brain.obj: 200
runtime: Rust/candle (no Python)
encoders: {'text': '...', 'audio': '...', 'image': '...', 'video': '...'}
```

- [ ] **Step 3: Test WebSocket prediction**

```bash
pip3 install websockets --quiet
python3 - << 'EOF'
import asyncio, websockets, json

async def run():
    async with websockets.connect('ws://127.0.0.1:8081/ws') as ws:
        await ws.send(json.dumps({'text': 'visual cortex activates to bright red', 'seq_len': 4}))
        for _ in range(10):
            msg = json.loads(await ws.recv())
            t = msg.get('type')
            print(t, msg.get('pct',''), msg.get('msg', msg.get('modality','')))
            if t in ('result', 'error'):
                if t == 'result':
                    print('vertex_acts len:', len(msg.get('vertex_acts', [])))
                    print('temporal_acts frames:', len(msg.get('temporal_acts', [])))
                    print('demo_mode:', msg.get('demo_mode'))
                break

asyncio.run(run())
EOF
```

Expected:
```
progress 15 Encoding text…
progress 55 Running FmriEncoder…
progress 85 Computing region statistics…
progress 100 Done
result  text
vertex_acts len: 20484
temporal_acts frames: 4
demo_mode: True
```

- [ ] **Step 4: Test download job queue**

```bash
# Submit a clip download job
curl -s -X POST http://127.0.0.1:8081/api/download/clip | python3 -m json.tool
# → {"job_id": "<uuid>"}

# Capture the job_id and poll it
JOB_ID=$(curl -s -X POST http://127.0.0.1:8081/api/download/clip 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))")
# Second POST for same model → 409
curl -s -o /dev/null -w "duplicate → %{http_code}\n" -X POST http://127.0.0.1:8081/api/download/clip

# Poll job
curl -s "http://127.0.0.1:8081/api/jobs/$JOB_ID" | python3 -m json.tool

# Unknown model → 400
curl -s -o /dev/null -w "bad model → %{http_code}\n" -X POST http://127.0.0.1:8081/api/download/gpt4
```

Expected:
```
{"job_id": "..."}
duplicate → 409
{"id": "...", "model": "clip", "status": "Queued", "pct": 0.0, ...}
bad model → 400
```

- [ ] **Step 5: Confirm Python is gone**

```bash
ls /Users/evintleovonzko/Documents/research/tribe-playground/*.py
```

Expected: only `convert_ckpt.py` and `convert_brain.py`. No `inference_server.py`, no `download_*.py`.

- [ ] **Step 6: Final commit**

```bash
cd /Users/evintleovonzko/Documents/research/tribe-playground
git add -A
git commit -m "chore: end-to-end verification — Rust server fully operational on port 8081"
```

---

## Self-Review

**Spec coverage:**
- ✅ Port changed to 8081
- ✅ `AppState` encoder fields → `Arc<RwLock<Option<Enc>>>`
- ✅ `job_store: JobStore` + `job_tx: Sender<Uuid>` added to AppState
- ✅ Background worker spawned in main
- ✅ `POST /api/download/{model}` → job_id (409 on duplicate)
- ✅ `GET /api/jobs/{job_id}` polling
- ✅ `GET /ws/jobs/{job_id}` streaming (progress / done / error)
- ✅ `vertex_acts: Vec<f32>` (20484) replacing `vertex_sample` (512)
- ✅ `temporal_acts: Vec<Vec<f32>>` ([T][6]) added
- ✅ Hot-swap: `downloader::hot_swap` writes to `state.{text,audio,clip}_enc`
- ✅ Python files deleted
- ✅ Frontend download buttons with progress bars

**Type consistency check:**
- `ModelKind` defined in `jobs.rs`, used in `downloader.rs`, `handlers.rs` via `crate::jobs::ModelKind` ✓
- `JobStore = Arc<RwLock<HashMap<Uuid, Job>>>` used identically across all files ✓
- `job_tx: mpsc::Sender<Uuid>` matches `mpsc::Receiver<Uuid>` in worker ✓
- `vertex_acts` named consistently in `PredictResp` struct and JavaScript `data.vertex_acts` ✓
- `temporal_acts` named consistently in struct and JS `data.temporal_acts` ✓
