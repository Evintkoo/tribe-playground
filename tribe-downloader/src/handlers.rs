/// handlers.rs — Axum HTTP + WebSocket handlers for TRIBE v2.
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use crate::{
    audio::{decode_audio, MelSpec},
    features::{
        region_stats, temporal_pool, text_to_features_demo, tribe_group_mean,
        visual_features_clip, visual_features_demo,
    },
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
    pub vertex_sample: Vec<f32>,
    pub seq_len:       usize,
    pub modality:      String,
    pub elapsed_ms:    f64,
    pub demo_mode:     bool,
}

// ── Shared prediction logic ───────────────────────────────────────────────────

/// Run a single prediction and call `progress(pct, msg)` at each step.
/// Returns PredictResp or an error string.
async fn run_prediction<F>(
    req: PredictReq,
    st: Arc<AppState>,
    mut progress: F,
) -> Result<PredictResp, String>
where
    F: FnMut(u8, &str),
{
    let t0 = Instant::now();

    let text       = req.text.as_deref().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
    let audio_b64  = req.audio_b64.filter(|s| !s.is_empty());
    let image_b64  = req.image_b64.filter(|s| !s.is_empty());
    let video_b64  = req.video_b64.filter(|s| !s.is_empty());

    if text.is_none() && audio_b64.is_none() && image_b64.is_none() && video_b64.is_none() {
        return Err("Provide 'text', 'audio_b64', 'image_b64', or 'video_b64'".into());
    }

    let seq = req.seq_len.unwrap_or(16).clamp(1, 100);
    let device = &st.device;
    let mut demo_mode = false;
    let mut modalities: Vec<&str> = Vec::new();

    // ── Text features ─────────────────────────────────────────────────────────
    let text_feat: Option<Tensor> = if let Some(ref txt) = text {
        modalities.push("text");
        progress(15, "Encoding text…");

        if let Some(ref enc) = st.text_enc {
            // Real LLaMA-3.2-3B encoder
            let txt_owned = txt.clone();
            let enc_ref: &crate::llama_encoder::LlamaTextEncoder = enc;
            match enc_ref.encode(&txt_owned, seq) {
                Ok(t)  => Some(t),
                Err(e) => return Err(format!("LLaMA encode: {e}")),
            }
        } else {
            // Hash-based fallback (demo)
            demo_mode = true;
            match text_to_features_demo(txt, seq, device) {
                Ok(t)  => Some(t),
                Err(e) => return Err(e.to_string()),
            }
        }
    } else {
        None
    };

    // ── Audio features ────────────────────────────────────────────────────────
    let audio_feat: Option<Tensor> = if let Some(ref b64) = audio_b64 {
        modalities.push("audio");
        progress(25, "Encoding audio…");
        match encode_audio(b64, seq, &st.audio_enc, &st.mel_spec, device) {
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

        // Use CLIP encoder for images when available; fall back to hash-demo otherwise.
        // Video always uses hash-demo (no frame extractor yet).
        if !is_video_modality {
            if let Some(ref clip) = st.clip_enc {
                match visual_features_clip(&bytes, seq, clip) {
                    Ok(t)  => Some(t),
                    Err(e) => return Err(format!("CLIP encode: {e}")),
                }
            } else {
                demo_mode = true;
                match visual_features_demo(&bytes, seq, device) {
                    Ok(t)  => Some(t),
                    Err(e) => return Err(format!("visual demo: {e}")),
                }
            }
        } else {
            demo_mode = true;
            match visual_features_demo(&bytes, seq, device) {
                Ok(t)  => Some(t),
                Err(e) => return Err(format!("visual demo: {e}")),
            }
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

    // out: [1, seq, 20484]
    progress(85, "Computing region statistics…");
    let act_flat: Vec<f32> = out.squeeze(0)
        .and_then(|o| o.to_vec2::<f32>())
        .map(|rows| rows.into_iter().flatten().collect())
        .map_err(|e| e.to_string())?;

    let n_vert = crate::fmri_encoder::N_VERT;
    let (rstats, gstats) = region_stats(&act_flat, n_vert);

    let n_time = act_flat.len() / n_vert;
    let mean_v: Vec<f32> = (0..n_vert)
        .map(|vi| act_flat.iter().skip(vi).step_by(n_vert).sum::<f32>() / n_time as f32)
        .collect();
    let sample: Vec<f32> = (0..512usize)
        .map(|i| mean_v[(i * (n_vert - 1) / 511).min(n_vert - 1)])
        .collect();

    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    progress(100, "Done");

    Ok(PredictResp {
        region_stats: rstats,
        global_stats: gstats,
        vertex_sample: sample,
        seq_len: seq,
        modality: modalities.join("+"),
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
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

// ── /ws (WebSocket) ───────────────────────────────────────────────────────────

pub async fn ws_handler(
    ws: WebSocketUpgrade,
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

                // Stream progress events through the socket
                let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
                let st2 = st.clone();

                // Spawn blocking prediction on a thread so progress sends can interleave
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

                // Forward all messages from the channel to the WebSocket
                while let Some(msg) = rx.recv().await {
                    if socket.send(Message::Text(msg)).await.is_err() {
                        return;  // client disconnected
                    }
                }
            }
            Message::Close(_) => break,
            Message::Ping(d) => {
                let _ = socket.send(Message::Pong(d)).await;
            }
            _ => {}
        }
    }
}

// ── /health ───────────────────────────────────────────────────────────────────

pub async fn health(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "ready": true,
        "audio_enc": st.audio_enc.is_some(),
        "text_enc": st.text_enc.is_some(),
        "clip_enc": st.clip_enc.is_some(),
    }))
}

// ── /api/info ─────────────────────────────────────────────────────────────────

pub async fn info(State(st): State<Arc<AppState>>) -> impl IntoResponse {
    let text_status = if st.text_enc.is_some() {
        "LLaMA-3.2-3B · loaded"
    } else {
        "hash-based (place weights in tribe-v2-weights/llama/)"
    };
    Json(json!({
        "model_class": "FmriEncoder",
        "n_params": st.fmri.n_params(),
        "checkpoint": "facebook/tribev2 · best.safetensors",
        "architecture": {
            "hidden_dim": 1152,
            "n_heads":    8,
            "n_layers":   16,
            "ff_mult":    4,
            "n_vertices": 20484,
            "max_seq_len":1024,
            "surface":    "fsaverage5"
        },
        "feature_dims": {
            "text":  { "total": 6144, "encoder": "LLaMA-3.2-3B" },
            "audio": { "total": 2048, "encoder": "Wav2Vec-BERT 2.0" },
            "video": { "total": 2816, "encoder": "V-JEPA2 ViT-G" }
        },
        "encoders": {
            "text":  text_status,
            "audio": if st.audio_enc.is_some() { "Wav2Vec-BERT 2.0 · loaded" } else { "not loaded" },
            "image": if st.clip_enc.is_some() { "CLIP ViT-L/14 · loaded" } else { "not loaded" },
            "video": "not loaded"
        },
        "runtime": "Rust/candle (no Python)",
        "demo_mode": st.text_enc.is_none() || st.clip_enc.is_none()
    }))
}

// ── Internal: audio encoding ──────────────────────────────────────────────────

fn encode_audio(
    b64: &str,
    seq_len: usize,
    enc: &Option<Wav2Vec2Bert>,
    mel: &Option<MelSpec>,
    device: &Device,
) -> anyhow::Result<Option<Tensor>> {
    let audio_enc = match enc  { Some(e) => e, None => return Ok(None) };
    let mel_spec  = match mel  { Some(m) => m, None => return Ok(None) };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)?;
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
