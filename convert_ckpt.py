#!/usr/bin/env python3
"""
One-time setup: convert PyTorch Lightning checkpoint and save audio preprocessing
artifacts for use by the Rust inference engine.

Run once:  python3 convert_ckpt.py
"""
import os, sys

os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "tribe-v2-weights")

# ── 1. FmriEncoder checkpoint → safetensors ────────────────────────────────
CKPT = os.path.join(WEIGHTS_DIR, "best.ckpt")
ST   = os.path.join(WEIGHTS_DIR, "best.safetensors")

if os.path.exists(ST):
    print(f"[convert] {ST} already exists, skipping checkpoint conversion.")
else:
    print(f"[convert] Loading {CKPT} …")
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {
        k[len("model."):]: v.contiguous()
        for k, v in ck["state_dict"].items()
        if k.startswith("model.")
    }
    print(f"[convert] Saving {len(sd)} tensors to {ST} …")
    from safetensors.torch import save_file
    save_file(sd, ST)
    print(f"[convert] best.safetensors ✓")

# ── 2. Mel filterbank matrix ────────────────────────────────────────────────
MEL_BIN = os.path.join(WEIGHTS_DIR, "mel_filters_f32.bin")

if os.path.exists(MEL_BIN):
    print(f"[convert] {MEL_BIN} already exists, skipping.")
else:
    print("[convert] Computing mel filterbank …")
    from transformers import AutoFeatureExtractor
    import numpy as np
    fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    # mel_filters shape: (257, 80) float32
    mel = fe.mel_filters.astype(np.float32)
    with open(MEL_BIN, "wb") as f:
        f.write(mel.tobytes())   # row-major C-order: [257][80]
    print(f"[convert] mel_filters_f32.bin  shape={mel.shape} ✓")

# ── 3. Povey window ─────────────────────────────────────────────────────────
WIN_BIN = os.path.join(WEIGHTS_DIR, "povey_window_f32.bin")

if os.path.exists(WIN_BIN):
    print(f"[convert] {WIN_BIN} already exists, skipping.")
else:
    print("[convert] Computing Povey window …")
    from transformers import AutoFeatureExtractor
    import numpy as np
    fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    win = fe.window.astype(np.float32)
    with open(WIN_BIN, "wb") as f:
        f.write(win.tobytes())
    print(f"[convert] povey_window_f32.bin  len={len(win)} ✓")

print("[convert] All artifacts ready.")
