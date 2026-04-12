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

/// Run one download job. Called by the background worker.
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
            let _ = tokio::fs::remove_file(&tmp).await;
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
