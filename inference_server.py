"""
TRIBE v2 inference server.
Implements FmriEncoder from the checkpoint's state_dict structure and serves
predictions via FastAPI on port 8081.

Feature encoders (loaded lazily):
  - AudioEncoder : facebook/w2v-bert-2.0   → [T, 2048]
  - TextEncoder  : meta-llama/Llama-3.2-3B → [T, 6144]  (requires HF login + licence)
  - VideoEncoder : not implemented (stays zero / demo)
"""
from __future__ import annotations

# ── Must be set before ANY other imports to prevent TF mutex crash on macOS ──
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys, math, hashlib, base64, time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import json as _json

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH  = os.path.join(SCRIPT_DIR, "tribe-v2-weights", "best.ckpt")
PORT       = 8081

# ── Model architecture ────────────────────────────────────────────────────────

class ScaleNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8) * self.g


class RotaryEmbedding(nn.Module):
    """Partial rotary PE: rotates the first 72 of 144 head-dims."""
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.zeros(36))

    def forward(self, seq_len: int, device: torch.device):
        t    = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def _apply_rotary(q, k, cos, sin):
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    def _rot(x):
        x1, x2 = x[..., :rot_dim//2], x[..., rot_dim//2:]
        return torch.cat([-x2, x1], dim=-1)
    return (torch.cat([q_rot * cos + _rot(q_rot) * sin, q_pass], dim=-1),
            torch.cat([k_rot * cos + _rot(k_rot) * sin, k_pass], dim=-1))


class Attention(nn.Module):
    def __init__(self, dim: int = 1152, heads: int = 8):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.scale    = self.head_dim ** -0.5
        self.to_q  = nn.Linear(dim, dim, bias=False)
        self.to_k  = nn.Linear(dim, dim, bias=False)
        self.to_v  = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        h, hd   = self.heads, self.head_dim
        q = self.to_q(x).view(B,T,h,hd).transpose(1,2)
        k = self.to_k(x).view(B,T,h,hd).transpose(1,2)
        v = self.to_v(x).view(B,T,h,hd).transpose(1,2)
        q, k = _apply_rotary(q, k, cos, sin)
        attn = torch.softmax(q @ k.transpose(-2,-1) * self.scale, dim=-1)
        return self.to_out((attn @ v).transpose(1,2).reshape(B,T,D))


class FeedForward(nn.Module):
    def __init__(self, dim: int = 1152, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner)),
            nn.GELU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x): return self.ff(x)


class ResidualScale(nn.Module):
    def __init__(self, dim: int = 1152):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim))


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int = 1152, heads: int = 8, depth: int = 8):
        super().__init__()
        self.rotary_pos_emb = RotaryEmbedding()
        self.layers = nn.ModuleList()
        for i in range(depth * 2):
            self.layers.append(nn.ModuleList([
                nn.ModuleList([ScaleNorm()]),
                Attention(dim, heads) if i % 2 == 0 else FeedForward(dim),
                ResidualScale(dim),
            ]))
        self.final_norm = ScaleNorm()

    def forward(self, x):
        B, T, _ = x.shape
        cos, sin = self.rotary_pos_emb(T, x.device)
        for i, layer in enumerate(self.layers):
            normed = layer[0][0](x)
            out = layer[1](normed, cos, sin) if i % 2 == 0 else layer[1](normed)
            x = x + out * layer[2].residual_scale
        return self.final_norm(x)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, 2048, 20484))
        self.bias    = nn.Parameter(torch.zeros(1, 20484))

    def forward(self, x):
        return x @ self.weights[0] + self.bias[0]


class FmriEncoder(nn.Module):
    DIM = 1152
    def __init__(self):
        super().__init__()
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 1024, self.DIM))
        self.projectors = nn.ModuleDict({
            "text":  nn.Linear(6144, 384),
            "audio": nn.Linear(2048, 384),
            "video": nn.Linear(2816, 384),
        })
        self.encoder       = TransformerEncoder(self.DIM, heads=8, depth=8)
        self.low_rank_head = nn.Linear(self.DIM, 2048, bias=False)
        self.predictor     = Predictor()

    def forward(self,
                text:  Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                video: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        ref = next(f for f in (text, audio, video) if f is not None)
        B, T = ref.shape[0], ref.shape[1]
        z = lambda n: torch.zeros(B, T, n, device=device)
        t = self.projectors["text" ](text  if text  is not None else z(6144))
        a = self.projectors["audio"](audio if audio is not None else z(2048))
        v = self.projectors["video"](video if video is not None else z(2816))
        x = torch.cat([t, a, v], dim=-1) + self.time_pos_embed[:, :T]
        return self.predictor(self.low_rank_head(self.encoder(x)))


# ── Load FmriEncoder ─────────────────────────────────────────────────────────

def load_fmri_encoder() -> FmriEncoder:
    print(f"[tribe] Loading checkpoint: {CKPT_PATH} …", flush=True)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items()
          if k.startswith("model.")}
    model = FmriEncoder()
    model.load_state_dict(sd, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[tribe] FmriEncoder loaded ✓  ({n/1e6:.1f}M params)", flush=True)
    return model


# ── Feature helpers ───────────────────────────────────────────────────────────

def temporal_pool(feats: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly interpolate [T_src, D] → [target_len, D]."""
    T = feats.shape[0]
    if T == target_len:
        return feats
    x = feats.float().T.unsqueeze(0)   # [1, D, T_src]
    x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
    return x.squeeze(0).T              # [target_len, D]


def tribe_group_mean(h0: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """
    TRIBE group_mean over layers [0.5, 0.75, 1.0]:
      group0 = h0          (layers at relative pos 0.5)
      group1 = mean(h1,h2) (layers at relative pos 0.75, 1.0)
      output = cat([group0, group1], dim=-1)  →  2× the feature dim
    """
    return torch.cat([h0, (h1 + h2) / 2], dim=-1)


# ── Audio encoder (Wav2Vec-BERT 2.0) ─────────────────────────────────────────

class AudioEncoder:
    """
    Extracts audio features using facebook/w2v-bert-2.0 (580 M params).
    Returns [1, seq_len, 2048] tensors matching FmriEncoder's audio projector.
    """
    MODEL_ID = "facebook/w2v-bert-2.0"

    def __init__(self):
        from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        print(f"[tribe] Loading AudioEncoder ({self.MODEL_ID}) …", flush=True)
        # Use AutoFeatureExtractor (not AutoProcessor) — w2v-bert-2.0 needs audio FE only
        self.processor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = Wav2Vec2BertModel.from_pretrained(self.MODEL_ID)
        self.model.eval()
        n_layers = self.model.config.num_hidden_layers   # 24
        self.li = [
            round(0.50 * n_layers),   # 12
            round(0.75 * n_layers),   # 18
            n_layers,                  # 24
        ]
        print(f"[tribe] AudioEncoder ready ✓  (layers {self.li})", flush=True)

    def encode(self, audio_bytes: bytes, seq_len: int) -> torch.Tensor:
        import io
        import librosa
        audio_io = io.BytesIO(audio_bytes)
        waveform, _sr = librosa.load(audio_io, sr=16000, mono=True, dtype=np.float32)
        # norm_audio: true (from config)
        waveform = waveform / (np.abs(waveform).max() + 1e-8)

        inputs = self.processor(waveform, return_tensors="pt",
                                sampling_rate=16000, padding=True)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        # hidden_states[0] = after feat extractor; [1..24] = conformer layers
        h = [out.hidden_states[i].squeeze(0) for i in self.li]  # each [T_a, 1024]
        feats = tribe_group_mean(*h)                              # [T_a, 2048]
        return temporal_pool(feats, seq_len).unsqueeze(0)         # [1, seq_len, 2048]


# ── Text encoder (LLaMA-3.2-3B) ──────────────────────────────────────────────

class TextEncoder:
    """
    Extracts text features using meta-llama/Llama-3.2-3B (3 B params).
    Requires: huggingface-cli login + Meta licence acceptance.
    Returns [1, seq_len, 6144] tensors matching FmriEncoder's text projector.
    """
    MODEL_ID = "meta-llama/Llama-3.2-3B"

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        print(f"[tribe] Loading TextEncoder ({self.MODEL_ID}) …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        # Load in bfloat16 to halve memory; inference stays accurate
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        n_layers = self.model.config.num_hidden_layers  # 28
        self.li = [
            round(0.50 * n_layers),   # 14
            round(0.75 * n_layers),   # 21
            n_layers,                  # 28
        ]
        print(f"[tribe] TextEncoder ready ✓  (layers {self.li})", flush=True)

    def encode(self, text: str, seq_len: int) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        # hidden_states[0] = embedding; [1..28] = transformer layers
        h = [out.hidden_states[i].squeeze(0).float() for i in self.li]  # [T_tok, 3072]
        feats = tribe_group_mean(*h)                                      # [T_tok, 6144]
        return temporal_pool(feats, seq_len).unsqueeze(0)                 # [1, seq_len, 6144]


# ── Image encoder (CLIP ViT-L/14) ────────────────────────────────────────────

class ImageEncoder:
    """
    Encodes images using CLIP ViT-L/14 vision transformer.
    Extracts patch hidden states at layers 50%, 75%, 100% (12, 18, 24).
    tribe_group_mean -> [n_patches, 2048].
    Temporal pool to seq_len then zero-pad to 2816 (video projector slot).
    Returns [1, seq_len, 2816].
    """
    def __init__(self, weights_path: str):
        from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
        from safetensors.torch import load_file

        print("[tribe] Loading ImageEncoder (CLIP ViT-L/14)\u2026", flush=True)
        cfg = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPVisionModel(cfg)
        self.model.eval()

        sd_full   = load_file(weights_path, device="cpu")
        prefix    = "vision_model."
        sd_vision = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
        if sd_vision:
            self.model.vision_model.load_state_dict(sd_vision, strict=False)
        else:
            # weights file may have no prefix (pure vision model save)
            self.model.load_state_dict(sd_full, strict=False)

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        n_layers = cfg.num_hidden_layers   # 24
        self.li  = [round(0.50 * n_layers), round(0.75 * n_layers), n_layers]
        print(f"[tribe] ImageEncoder ready \u2713  (layers {self.li})", flush=True)

    def encode(self, image_bytes: bytes, seq_len: int) -> torch.Tensor:
        from PIL import Image
        import io
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # hidden_states: list of [1, n_patches+1, 1024]
        # index 0 = embedding layer, 1..24 = transformer layers
        # Drop CLS token (position 0), keep patch tokens
        h      = [out.hidden_states[i].squeeze(0)[1:].float() for i in self.li]
        feats  = tribe_group_mean(*h)                           # [n_patches, 2048]
        pooled = temporal_pool(feats, seq_len)                  # [seq_len, 2048]
        pad    = torch.zeros(seq_len, 768, dtype=pooled.dtype)
        return torch.cat([pooled, pad], dim=-1).unsqueeze(0)    # [1, seq_len, 2816]


# ── Hash-based demo text features (fallback) ─────────────────────────────────

def text_to_features_demo(text: str, seq_len: int) -> torch.Tensor:
    """
    Deterministic pseudo-LLaMA features for demo/offline mode.
    Returns [1, seq_len, 6144].
    """
    words  = (text.lower().split() or [text])[:seq_len]
    proj_w = np.random.default_rng(42).standard_normal((768, 6144)) / math.sqrt(768)
    vecs   = []
    for w in words:
        ws = int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % (2**31)
        v  = np.random.default_rng(ws).standard_normal(768)
        vecs.append(np.clip(v, -3.0, 3.0))
    sent = np.stack(vecs)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        raw = sent.astype(np.float64) @ proj_w
    raw   = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)
    feats = torch.tensor(np.clip(raw, -5.0, 5.0), dtype=torch.float32)  # [words, 6144]
    return temporal_pool(feats, seq_len).unsqueeze(0)                     # [1, seq_len, 6144]


# ── Region statistics ─────────────────────────────────────────────────────────

REGION_RANGES = {
    "visual":     (0,     3600),
    "auditory":   (3600,  6800),
    "language":   (6800,  10500),
    "prefrontal": (10500, 14000),
    "motor":      (14000, 17200),
    "parietal":   (17200, 20484),
}


def region_stats(activations: np.ndarray):
    mean_t = activations.mean(axis=0)  # [20484]
    g_mean = float(mean_t.mean())
    g_std  = float(mean_t.std()) + 1e-8
    stats  = {}
    for region, (lo, hi) in REGION_RANGES.items():
        v = mean_t[lo:hi]
        rm = float(v.mean())
        stats[region] = {
            "mean":           rm,
            "std":            float(v.std()),
            "rel_activation": float((rm - g_mean) / g_std),
            "peak":           float(np.percentile(np.abs(v), 95)),
            "n_vertices":     hi - lo,
        }
    return stats, {
        "global_mean": g_mean,
        "global_std":  g_std - 1e-8,
        "global_min":  float(mean_t.min()),
        "global_max":  float(mean_t.max()),
    }


# ── Global model handles ──────────────────────────────────────────────────────

MODEL:         Optional[FmriEncoder] = None
AUDIO_ENCODER: Optional[AudioEncoder] = None
TEXT_ENCODER:  Optional[TextEncoder]  = None
IMAGE_ENCODER: Optional["ImageEncoder"] = None


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app):
    global MODEL, AUDIO_ENCODER, TEXT_ENCODER, IMAGE_ENCODER

    MODEL = load_fmri_encoder()

    # Try AudioEncoder
    try:
        AUDIO_ENCODER = AudioEncoder()
    except Exception as e:
        print(f"[tribe] AudioEncoder not loaded: {e}", flush=True)
        AUDIO_ENCODER = None

    # Try TextEncoder (requires HF login + Meta licence)
    try:
        TEXT_ENCODER = TextEncoder()
    except Exception as e:
        print(f"[tribe] TextEncoder not loaded (demo mode for text): {e}", flush=True)
        TEXT_ENCODER = None

    # Try ImageEncoder (CLIP ViT-L/14)
    clip_path = os.path.join(SCRIPT_DIR, "tribe-v2-weights", "clip", "model.safetensors")
    try:
        if os.path.exists(clip_path):
            IMAGE_ENCODER = ImageEncoder(clip_path)
        else:
            print(f"[tribe] CLIP weights not found at {clip_path} — run: python3 download_clip.py", flush=True)
    except Exception as e:
        print(f"[tribe] ImageEncoder not loaded: {e}", flush=True)
        IMAGE_ENCODER = None

    yield


app = FastAPI(title="TRIBE v2 inference", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def serve_html():
    p = os.path.join(SCRIPT_DIR, "tribe-v2-playground.html")
    if not os.path.exists(p):
        raise HTTPException(404, "tribe-v2-playground.html not found")
    return FileResponse(p, media_type="text/html")

@app.get("/brain.obj")
def serve_brain_obj():
    p = os.path.join(SCRIPT_DIR, "brain.obj")
    if not os.path.exists(p):
        raise HTTPException(404, "brain.obj not found")
    return FileResponse(p, media_type="model/obj")


# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text:       str = ""
    audio_b64:  str = ""
    image_b64:  str = ""   # base64 image bytes (JPEG / PNG / WebP)
    seq_len:    int = 16
    subject_id: int = 0


class PredictResponse(BaseModel):
    region_stats:  dict
    global_stats:  dict
    vertex_acts:   list   # 20484 floats (mean over T) for per-vertex BOLD coloring
    temporal_acts: list   # [T][6] regional means per timepoint for TR animation
    seq_len:       int
    modality:      str
    elapsed_ms:    float
    demo_mode:     bool


# ── Core prediction logic ─────────────────────────────────────────────────────

def _predict_core(req: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(503, "Model not ready")
    if not req.text.strip() and not req.audio_b64 and not req.image_b64:
        raise HTTPException(400, "Provide text, audio, or image input")

    t0        = time.perf_counter()
    seq       = max(1, min(req.seq_len, 100))
    demo_mode = False
    modalities: list[str] = []

    text_feat  = None
    audio_feat = None

    # ── Text ──────────────────────────────────────────────────────────────────
    if req.text.strip():
        modalities.append("text")
        if TEXT_ENCODER is not None:
            text_feat = TEXT_ENCODER.encode(req.text, seq)
        else:
            text_feat = text_to_features_demo(req.text, seq)
            demo_mode = True

    # ── Audio ─────────────────────────────────────────────────────────────────
    if req.audio_b64:
        modalities.append("audio")
        if AUDIO_ENCODER is not None:
            try:
                audio_bytes = base64.b64decode(req.audio_b64)
                audio_feat  = AUDIO_ENCODER.encode(audio_bytes, seq)
            except Exception as e:
                print(f"[tribe] Audio encoding error: {e}", flush=True)
                demo_mode = True
        else:
            demo_mode = True
            # still need something for the text projector if no text provided
            if text_feat is None:
                text_feat = text_to_features_demo("", seq)

    # ── Image ─────────────────────────────────────────────────────────────────
    image_feat = None
    if req.image_b64 and not req.audio_b64 and not req.text.strip():
        modalities.append("image")
        if IMAGE_ENCODER is not None:
            try:
                image_bytes = base64.b64decode(req.image_b64)
                image_feat  = IMAGE_ENCODER.encode(image_bytes, seq)
            except Exception as e:
                print(f"[tribe] Image encoding error: {e}", flush=True)
                demo_mode = True
                if text_feat is None:
                    text_feat = text_to_features_demo("", seq)
        else:
            demo_mode = True
            text_feat = text_to_features_demo("", seq)

    # Must have at least one feature tensor
    if text_feat is None and audio_feat is None and image_feat is None:
        raise HTTPException(400, "Could not produce any features from the input")

    # ── Run FmriEncoder ───────────────────────────────────────────────────────
    with torch.no_grad():
        out = MODEL(text=text_feat, audio=audio_feat, video=image_feat)  # [1, T, 20484]

    act      = out[0].numpy()          # [T, 20484]
    rstats, gstats = region_stats(act)

    mean_act    = act.mean(axis=0)     # [20484]
    vertex_acts = [round(float(v), 4) for v in mean_act]

    region_order = ["visual", "auditory", "language", "prefrontal", "motor", "parietal"]
    temporal_acts: list = []
    for t_idx in range(act.shape[0]):
        row = []
        for r in region_order:
            lo, hi = REGION_RANGES[r]
            row.append(round(float(act[t_idx, lo:hi].mean()), 4))
        temporal_acts.append(row)

    elapsed = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        region_stats  = rstats,
        global_stats  = gstats,
        vertex_acts   = vertex_acts,
        temporal_acts = temporal_acts,
        seq_len       = seq,
        modality      = "+".join(modalities) if modalities else "unknown",
        elapsed_ms    = round(elapsed, 1),
        demo_mode     = demo_mode,
    )


# ── HTTP predict endpoint ─────────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _predict_core(req)


# ── Health / Info endpoints ───────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"ready": MODEL is not None}


@app.get("/api/info")
def model_info():
    n_params = sum(p.numel() for p in MODEL.parameters()) if MODEL else 0
    return {
        "model_class":   "FmriEncoder",
        "n_params":      n_params,
        "checkpoint":    "facebook/tribev2 · best.ckpt",
        "architecture": {
            "hidden_dim":  1152,
            "n_heads":     8,
            "n_layers":    16,
            "ff_mult":     4,
            "n_vertices":  20484,
            "max_seq_len": 1024,
            "surface":     "fsaverage5",
        },
        "feature_dims": {
            "text":  {"n_groups": 2, "dim_per_group": 3072, "total": 6144,
                      "encoder": "LLaMA-3.2-3B"},
            "audio": {"n_groups": 2, "dim_per_group": 1024, "total": 2048,
                      "encoder": "Wav2Vec-BERT 2.0"},
            "video": {"n_groups": 2, "dim_per_group": 1408, "total": 2816,
                      "encoder": "V-JEPA2 ViT-G"},
        },
        "encoders": {
            "text":  ("LLaMA-3.2-3B · loaded"      if TEXT_ENCODER  else "demo · hash-based fallback"),
            "audio": ("Wav2Vec-BERT 2.0 · loaded"   if AUDIO_ENCODER else "not loaded"),
            "image": ("CLIP ViT-L/14 · loaded"      if IMAGE_ENCODER else "not loaded"),
            "video": "not loaded (V-JEPA2 not integrated)",
        },
        "training": {
            "n_subjects": 25,
            "n_epochs":   15,
            "datasets":   ["Algonauts2025Bold", "Lahner2024Bold", "Lebel2023Bold", "Wen2017"],
        },
        "demo_mode": TEXT_ENCODER is None,
    }


@app.websocket("/ws")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    async def send(msg: dict):
        await websocket.send_text(_json.dumps(msg))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req_data = _json.loads(raw)
            except _json.JSONDecodeError as exc:
                await send({"type": "error", "message": f"Invalid JSON: {exc}"})
                continue

            await send({"type": "progress", "pct": 5,  "msg": "Received request"})

            try:
                req = PredictRequest(
                    text       = req_data.get("text", ""),
                    audio_b64  = req_data.get("audio_b64", ""),
                    image_b64  = req_data.get("image_b64", ""),
                    seq_len    = int(req_data.get("seq_len", 16)),
                    subject_id = int(req_data.get("subject_id", 0)),
                )
                await send({"type": "progress", "pct": 20, "msg": "Encoding stimulus"})
                result = await loop.run_in_executor(None, _predict_core, req)
                await send({"type": "progress", "pct": 90, "msg": "Coloring brain"})
                payload = result.model_dump()
                payload["type"] = "result"
                await send(payload)
            except HTTPException as e:
                await send({"type": "error", "message": e.detail})
            except Exception as e:
                await send({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
