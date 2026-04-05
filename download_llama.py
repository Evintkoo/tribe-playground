#!/usr/bin/env python3
"""
Download LLaMA-3.2-3B weights for TRIBE v2 text encoder.

Requirements:
    pip install huggingface_hub

You need a HuggingFace account + accepted the Meta LLaMA 3.2 license at:
    https://huggingface.co/meta-llama/Llama-3.2-3B

Run:
    python3 download_llama.py
    # or supply token directly:
    HF_TOKEN=hf_... python3 download_llama.py
"""

import os, sys
from pathlib import Path

REPO_ID   = "NousResearch/Hermes-3-Llama-3.2-3B"
OUT_DIR   = Path(__file__).parent / "tribe-v2-weights" / "llama"

# Files needed by LlamaTextEncoder (tokenizer + model shards)
PATTERNS = [
    "tokenizer.json",
    "*.safetensors",
]

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download] huggingface_hub not installed.")
        print("           Run: pip install huggingface_hub")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading {REPO_ID} → {OUT_DIR}")
    print(f"[download] No login required — this is an open, ungated model.")
    print(f"[download] ~6 GB download — may take a while…")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(OUT_DIR),
            allow_patterns=PATTERNS,
            ignore_patterns=["*.bin"],  # safetensors only
        )
    except Exception as e:
        print(f"\n[download] Error: {e}")
        sys.exit(1)

    # Verify tokenizer.json landed
    tok = OUT_DIR / "tokenizer.json"
    shards = list(OUT_DIR.glob("*.safetensors"))
    if not tok.exists() or not shards:
        print("[download] WARNING: expected files not found — check the output directory.")
        sys.exit(1)

    print(f"\n[download] Done!  {len(shards)} shard(s) + tokenizer.json")
    print(f"[download] Restart the TRIBE server — LLaMA text encoder will load automatically.")

if __name__ == "__main__":
    main()
