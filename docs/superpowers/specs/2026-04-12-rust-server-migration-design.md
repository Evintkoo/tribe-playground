# TRIBE v2 — Rust Server Migration Design

**Date:** 2026-04-12

## Goal

Replace `inference_server.py` (FastAPI/Python) with the existing Rust crate (`tribe-downloader/`) as the sole TRIBE server. Add a download job queue with hot-reload so model weights can be fetched and activated without restarting the server. Delete all Python download scripts.

---

## Current State

| Component | Current | After |
|-----------|---------|-------|
| Server | `inference_server.py` (FastAPI, port 8081) | `tribe-server` (Axum, port 8081) |
| Download | `download_clip.py`, `download_llama.py`, `download_w2v.py` | `POST /api/download/{model}` job queue |
| Response | `vertex_sample: Vec<f32>` (512 points) | `vertex_acts: Vec<f32>` (20484) + `temporal_acts: Vec<Vec<f32>>` ([T][6]) |
| Encoder state | `Option<Enc>` loaded at startup | `Arc<RwLock<Option<Enc>>>` hot-swappable |

The Rust crate already builds cleanly and implements all encoders (FmriEncoder, LLaMA, Wav2Vec2-Bert, CLIP). The gaps being filled here are: correct response shape, download job queue, hot-reload, and Python retirement.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `tribe-downloader/Cargo.toml` | Modify | Rename package to `tribe-server`; add `uuid` dep |
| `tribe-downloader/src/main.rs` | Modify | Change port to 8081; change encoder fields to `Arc<RwLock<Option<…>>>`; add job store + download worker to `AppState`; wire new routes |
| `tribe-downloader/src/handlers.rs` | Modify | Fix response shape (`vertex_acts` + `temporal_acts`); add download/job/ws-job handlers; update read-lock usage for encoders |
| `tribe-downloader/src/jobs.rs` | Create | `Job` struct, `JobStore`, `JobQueue` (mpsc sender), background worker task |
| `tribe-downloader/src/downloader.rs` | Create | Per-model file list + HuggingFace URLs; streaming download with `bytes_done` updates; post-download encoder load + hot-swap |
| `inference_server.py` | Delete | Replaced by Rust server |
| `download_clip.py` | Delete | Replaced by `POST /api/download/clip` |
| `download_llama.py` | Delete | Replaced by `POST /api/download/llama` |
| `download_w2v.py` | Delete | Replaced by `POST /api/download/wav2vec` |
| `tribe-v2-playground.html` | Modify | Update WebSocket URL from port 8081 → 8081 (no change needed); update `fetchModelInfo` to show download buttons |

---

## Architecture

### Job System (`src/jobs.rs`)

```
POST /api/download/{model}
  └─ JobStore::enqueue(model) → Uuid
       └─ mpsc::Sender<JobId> → background worker
            └─ downloader::run(job_id, model, store, state)
```

**`Job` struct:**
```rust
pub struct Job {
    pub id:          Uuid,
    pub model:       ModelKind,   // Clip | Llama | Wav2Vec
    pub status:      JobStatus,   // Queued | Running | Done | Failed
    pub pct:         f32,
    pub bytes_done:  u64,
    pub total_bytes: Option<u64>,
    pub error:       Option<String>,
}
```

**`JobStore`:** `Arc<RwLock<HashMap<Uuid, Job>>>` — all handlers and the worker share one instance.

**Worker:** single Tokio task spawned at startup, reads from `mpsc::Receiver<Uuid>`, processes jobs one at a time. Concurrent POSTs queue up safely.

### Download Logic (`src/downloader.rs`)

Each `ModelKind` maps to a list of `(filename, HuggingFace URL)` pairs:

| Model | Files |
|-------|-------|
| `clip` | `tribe-v2-weights/clip/model.safetensors` from `openai/clip-vit-large-patch14` |
| `llama` | `tribe-v2-weights/llama/tokenizer.json`, `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` from `NousResearch/Hermes-3-Llama-3.2-3B` |
| `wav2vec` | `tribe-v2-weights/w2v-bert-2.0.safetensors` from `facebook/w2v-bert-2.0` |

Download uses `reqwest` streaming: reads chunks of 64 KiB, writes to file, increments `bytes_done` on the job. `total_bytes` is set from the `Content-Length` header if present. Supports resume: if the destination file already exists and is non-empty, the download is skipped.

After all files land, the worker loads the encoder and swaps it into `AppState` via a write lock on the appropriate `Arc<RwLock<Option<Enc>>>` field.

### Hot-Reload State

`AppState` encoder fields change from:
```rust
pub text_enc:  Option<LlamaTextEncoder>,
pub audio_enc: Option<Wav2Vec2Bert>,
pub clip_enc:  Option<ClipVisualEncoder>,
```
to:
```rust
pub text_enc:  Arc<RwLock<Option<LlamaTextEncoder>>>,
pub audio_enc: Arc<RwLock<Option<Wav2Vec2Bert>>>,
pub clip_enc:  Arc<RwLock<Option<ClipVisualEncoder>>>,
```

Handlers call `.read().await` to borrow the encoder for the duration of a prediction. The download worker calls `.write().await` only during the swap (milliseconds). No restart required.

---

## API

### Download endpoints

**`POST /api/download/{model}`** — `model` is `clip`, `llama`, or `wav2vec`

Response `200 OK`:
```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

Response `400` if model name unknown. Response `409` if a job for that model is already Queued or Running.

---

**`GET /api/jobs/{job_id}`** — poll job state

Response `200 OK`:
```json
{
  "id": "550e8400…",
  "model": "clip",
  "status": "Running",
  "pct": 42.7,
  "bytes_done": 731906048,
  "total_bytes": 1716987224,
  "error": null
}
```

Response `404` if job ID unknown.

---

**`GET /ws/jobs/{job_id}`** — stream job progress

Upgrades to WebSocket. Server sends one JSON message per progress update (chunk boundary), then a terminal message:

Progress: `{"type":"progress","pct":42.7,"bytes_done":731906048,"total_bytes":1716987224}`

Done: `{"type":"done","model":"clip"}`

Error: `{"type":"error","message":"…"}`

The WebSocket closes after the terminal message. If the job is already Done/Failed when the client connects, the terminal message is sent immediately.

---

### Updated predict response

`vertex_sample` removed. Two new fields added:

```json
{
  "vertex_acts":   [/* 20484 f32 — mean activation over T */],
  "temporal_acts": [[/* 6 regional means */], …]  /* T rows */
}
```

`temporal_acts` row order: `["visual","auditory","language","prefrontal","motor","parietal"]` — same as Python server.

---

## Frontend Changes

`tribe-v2-playground.html` needs minor additions:

1. **Download buttons** in the `#panel-image` meta bar and in a new settings/download section: one button per model (`clip`, `llama`, `wav2vec`). Each button POSTs to `/api/download/{model}`, then opens a WS to `/ws/jobs/{job_id}` and shows a progress bar.

2. **No port change needed** — Rust server runs on 8081.

3. **No WebSocket predict change needed** — `/ws` endpoint is unchanged.

---

## Deletion List

- `inference_server.py`
- `download_clip.py`
- `download_llama.py`
- `download_w2v.py`

`convert_ckpt.py` and `convert_brain.py` are conversion utilities; they are kept (still referenced by `main.rs`).

---

## Non-Goals

- V-JEPA2 video encoder (not integrated in Python either; stays as demo)
- GPU download acceleration
- Parallel file downloads within a job (sequential is simpler and avoids disk contention)
- Authentication / rate limiting on download endpoints
