#!/usr/bin/env python3
"""
Download OpenAI CLIP ViT-L/14 weights for TRIBE v2 image encoder.

Ungated — no login required.

Run:
    python3 download_clip.py
"""

import sys
from pathlib import Path

REPO_ID  = "openai/clip-vit-large-patch14"
OUT_DIR  = Path(__file__).parent / "tribe-v2-weights" / "clip"
OUT_FILE = OUT_DIR / "model.safetensors"

def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download] huggingface_hub not installed — run: pip install huggingface_hub")
        sys.exit(1)

    if OUT_FILE.exists():
        print(f"[download] Already exists: {OUT_FILE}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading {REPO_ID}/model.safetensors (~1.7 GB)…")

    try:
        tmp = hf_hub_download(
            repo_id=REPO_ID,
            filename="model.safetensors",
        )
    except Exception as e:
        # Fallback: try pytorch_model.bin and convert
        print(f"[download] safetensors not found ({e}), trying pytorch_model.bin…")
        _download_and_convert()
        return

    import shutil
    shutil.copy(tmp, OUT_FILE)
    print(f"[download] Saved → {OUT_FILE}")
    print("[download] Restart the TRIBE server — image encoder will load automatically.")

def _download_and_convert():
    try:
        from huggingface_hub import hf_hub_download
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"[download] Missing dependency: {e}")
        print("           Run: pip install torch safetensors")
        sys.exit(1)

    print("[download] Downloading pytorch_model.bin…")
    tmp = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
    print("[download] Converting to safetensors…")
    state = torch.load(tmp, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(OUT_FILE))
    print(f"[download] Saved → {OUT_FILE}")
    print("[download] Restart the TRIBE server — image encoder will load automatically.")

if __name__ == "__main__":
    main()
