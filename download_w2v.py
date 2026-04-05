#!/usr/bin/env python3
"""
Download Wav2Vec2-BERT 2.0 weights for TRIBE v2 audio encoder.

Ungated — no login required.

Run:
    python3 download_w2v.py
"""

import sys
from pathlib import Path

REPO_ID  = "facebook/w2v-bert-2.0"
OUT_FILE = Path(__file__).parent / "tribe-v2-weights" / "w2v-bert-2.0.safetensors"

def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download] huggingface_hub not installed — run: pip install huggingface_hub")
        sys.exit(1)

    if OUT_FILE.exists():
        print(f"[download] Already exists: {OUT_FILE}")
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading {REPO_ID}/model.safetensors (~1.2 GB)…")

    try:
        tmp = hf_hub_download(
            repo_id=REPO_ID,
            filename="model.safetensors",
        )
    except Exception as e:
        print(f"[download] Error: {e}")
        sys.exit(1)

    import shutil
    shutil.copy(tmp, OUT_FILE)
    print(f"[download] Saved → {OUT_FILE}")
    print("[download] Restart the TRIBE server — audio encoder will load automatically.")

if __name__ == "__main__":
    main()
