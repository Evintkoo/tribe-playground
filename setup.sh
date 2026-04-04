#!/usr/bin/env bash
# TRIBE v2 — setup script
# Installs Python dependencies and optionally sets up HuggingFace auth for LLaMA-3.2-3B.

set -e
BOLD='\033[1m'; RESET='\033[0m'; GREEN='\033[32m'; YELLOW='\033[33m'; CYAN='\033[36m'

echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  TRIBE v2 · Feature encoder setup           ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo

# ── Python dependencies ──────────────────────────────────────────────────────
echo -e "${CYAN}[1/3] Installing Python dependencies…${RESET}"
pip3 install --quiet --upgrade \
  transformers \
  accelerate \
  huggingface_hub \
  librosa \
  soundfile \
  fastapi \
  uvicorn \
  pydantic \
  numpy \
  torch

echo -e "${GREEN}      ✓ Dependencies installed${RESET}"
echo

# ── Wav2Vec-BERT 2.0 (audio encoder, public) ─────────────────────────────────
echo -e "${CYAN}[2/3] Pre-downloading Wav2Vec-BERT 2.0 (audio encoder, public)…${RESET}"
python3 - <<'PYEOF'
import os
os.environ.setdefault("USE_TF","0")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
print("  Downloading facebook/w2v-bert-2.0 …")
AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
print("  ✓ Wav2Vec-BERT 2.0 cached")
PYEOF
echo

# ── LLaMA-3.2-3B (text encoder, gated) ──────────────────────────────────────
echo -e "${CYAN}[3/3] LLaMA-3.2-3B setup (text encoder, requires Meta licence)${RESET}"
echo
echo -e "  ${YELLOW}LLaMA-3.2-3B is a gated model. To enable full semantic text encoding:${RESET}"
echo
echo -e "  1. Accept the Meta licence at:"
echo -e "     ${BOLD}https://huggingface.co/meta-llama/Llama-3.2-3B${RESET}"
echo
echo -e "  2. Generate a HuggingFace token at:"
echo -e "     ${BOLD}https://huggingface.co/settings/tokens${RESET}"
echo
echo -e "  3. Log in (run this command and paste your token when prompted):"
echo -e "     ${BOLD}huggingface-cli login${RESET}"
echo
echo -e "  4. Then re-run this script or start the server with:"
echo -e "     ${BOLD}cargo run${RESET}"
echo

read -r -p "  Do you have a HuggingFace token and want to log in now? [y/N] " REPLY
echo
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
  huggingface-cli login
  echo
  echo -e "${CYAN}  Downloading meta-llama/Llama-3.2-3B (this will take a while)…${RESET}"
  python3 - <<'PYEOF2'
import os
os.environ.setdefault("USE_TF","0")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import torch
from transformers import AutoTokenizer, AutoModel
print("  Downloading meta-llama/Llama-3.2-3B …")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
AutoModel.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print("  ✓ LLaMA-3.2-3B cached")
PYEOF2
  echo
  echo -e "${GREEN}✓ LLaMA-3.2-3B ready — full semantic text encoding enabled${RESET}"
else
  echo -e "  ${YELLOW}Skipped — text predictions will use hash-based demo features${RESET}"
fi

echo
echo -e "${GREEN}${BOLD}Setup complete. Run the server with:  cargo run${RESET}"
