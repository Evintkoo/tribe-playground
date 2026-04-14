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
    audio::decode_audio,
    features::{
        region_stats, temporal_pool, text_to_features_demo, tribe_group_mean,
        visual_features_clip, visual_features_demo, REGION_RANGES,
    },
    jobs::{self, JobStatus},
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
            let cache_key = (txt.clone(), seq);
            // Check cache first
            let cached = st.text_feat_cache.read().await.get(&cache_key).cloned();
            let feat_vec = if let Some(v) = cached {
                v
            } else {
                let t = enc.encode(txt, seq).map_err(|e| format!("LLaMA: {e}"))?;
                let v = t.flatten_all()
                    .and_then(|f| f.to_vec1::<f32>())
                    .map_err(|e| format!("LLaMA to_vec: {e}"))?;
                // Cache up to 32 entries (drop all on overflow)
                let mut cache = st.text_feat_cache.write().await;
                if cache.len() >= 32 { cache.clear(); }
                cache.insert(cache_key, v.clone());
                v
            };
            drop(guard);
            let shape = (1usize, seq, feat_vec.len() / seq);
            Tensor::from_vec(feat_vec, shape, device)
                .and_then(|t| t.to_device(device))
                .map(Some)
                .map_err(|e| format!("LLaMA tensor: {e}"))?
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
    if n_time == 0 {
        return Err(format!(
            "unexpected FmriEncoder output length: {} (expected >= {n_vert})",
            act_flat.len()
        ));
    }

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
    ws:        WebSocketUpgrade,
    State(st): State<Arc<AppState>>,
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
                let st2 = st.clone();
                let tx2 = tx.clone();

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
                // Check terminal status FIRST before sending progress
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
                    _ => {
                        // Job still running — send progress update
                        let progress_msg = json!({
                            "type":        "progress",
                            "pct":         j.pct,
                            "bytes_done":  j.bytes_done,
                            "total_bytes": j.total_bytes,
                        });
                        if socket.send(Message::Text(progress_msg.to_string())).await.is_err() {
                            break;
                        }
                    }
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
