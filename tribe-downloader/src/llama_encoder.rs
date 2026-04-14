/// llama_encoder.rs — LLaMA-3.2-3B text encoder for TRIBE v2.
///
/// Extracts hidden states at layers 14 and 28 (1-indexed, 0-indexed: 13, 27),
/// concatenates along the feature dimension → [T, 6144].
/// temporal_pool to [seq_len, 6144] → [1, seq_len, 6144].
///
/// Weights are loaded from tribe-v2-weights/llama/ which should contain:
///   tokenizer.json
///   model.safetensors  (or shards: model-00001-of-NNNNN.safetensors …)

use anyhow::{Context, Result as AResult};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, VarBuilder};
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, info};

// ── Architecture constants (meta-llama/Llama-3.2-3B) ─────────────────────────
const HIDDEN:      usize = 3072;
const N_HEADS:     usize = 24;
const N_KV_HEADS:  usize = 8;
const HEAD_DIM:    usize = HIDDEN / N_HEADS;   // 128
const N_LAYERS:    usize = 28;
const FF_INNER:    usize = 8192;
const VOCAB:       usize = 128256;
const RMS_EPS:     f64   = 1e-5;
const ROPE_THETA:  f64   = 500000.0;

/// TRIBE: extract two layers → concat → 2×HIDDEN = 6144.
const EXTRACT_AT: [usize; 2] = [13, 27];  // 0-indexed (layers 14 and 28)
#[allow(dead_code)]
pub const TEXT_DIM: usize = HIDDEN * 2;    // 6144

// ── RMSNorm ───────────────────────────────────────────────────────────────────

struct RmsNorm {
    w: Tensor,
}

impl RmsNorm {
    fn load(size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { w: vb.get((size,), "weight")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rms = x.sqr()?.mean_keepdim(D::Minus1)?
            .clamp(RMS_EPS, f64::MAX)?.sqrt()?;
        x.broadcast_div(&rms)?.broadcast_mul(&self.w)
    }
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

/// Returns (cos, sin) each [T, HEAD_DIM].
fn build_rope(t: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (ROPE_THETA as f32).powf(-(2.0 * i as f32) / HEAD_DIM as f32))
        .collect();
    let inv = Tensor::from_vec(inv, (half,), device)?;
    let pos = Tensor::arange(0u32, t as u32, device)?.to_dtype(DType::F32)?;
    let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)?;  // [T, half]
    // repeat to full HEAD_DIM
    let cos = Tensor::cat(&[freqs.cos()?, freqs.cos()?], 1)?;  // [T, HEAD_DIM]
    let sin = Tensor::cat(&[freqs.sin()?, freqs.sin()?], 1)?;
    Ok((cos, sin))
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let h = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, h)?;
    let x2 = x.narrow(D::Minus1, h, h)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [B, H, T, HD], cos/sin: [T, HD] → broadcast over B and H
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;  // [1, 1, T, HD]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

// ── GQA Self-Attention ────────────────────────────────────────────────────────

struct GQAttention {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    o: candle_nn::Linear,
}

impl GQAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        Ok(Self {
            q: linear_no_bias(HIDDEN, HIDDEN,  vb.pp("q_proj"))?,
            k: linear_no_bias(HIDDEN, kv_dim,  vb.pp("k_proj"))?,
            v: linear_no_bias(HIDDEN, kv_dim,  vb.pp("v_proj"))?,
            o: linear_no_bias(HIDDEN, HIDDEN,  vb.pp("o_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self.q.forward(x)?
            .reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?;     // [B, Hq, T, HD]
        let k = self.k.forward(x)?
            .reshape((b, t, N_KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;  // [B, Hkv, T, HD]
        let v = self.v.forward(x)?
            .reshape((b, t, N_KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // Expand KV heads: Hkv → Hq via cat
        let rep = N_HEADS / N_KV_HEADS;
        let k = Tensor::cat(&(0..rep).map(|_| k.clone()).collect::<Vec<_>>(), 1)?;  // [B, Hq, T, HD]
        let v = Tensor::cat(&(0..rep).map(|_| v.clone()).collect::<Vec<_>>(), 1)?;

        let scale = (HEAD_DIM as f64).powf(-0.5);

        // Flatten B×H for Metal 3-D matmul compatibility
        let bh = b * N_HEADS;
        let q3 = q.reshape((bh, t, HEAD_DIM))?;
        let k3 = k.reshape((bh, t, HEAD_DIM))?;
        let v3 = v.reshape((bh, t, HEAD_DIM))?;
        let attn = candle_nn::ops::softmax(
            &(q3.matmul(&k3.transpose(1, 2)?)? * scale)?,
            D::Minus1,
        )?;
        let out = attn.matmul(&v3)?
            .reshape((b, N_HEADS, t, HEAD_DIM))?
            .transpose(1, 2)?
            .reshape((b, t, HIDDEN))?;
        self.o.forward(&out)
    }
}

// ── SwiGLU MLP ───────────────────────────────────────────────────────────────

struct MLP {
    gate: candle_nn::Linear,
    up:   candle_nn::Linear,
    down: candle_nn::Linear,
}

impl MLP {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(HIDDEN, FF_INNER, vb.pp("gate_proj"))?,
            up:   linear_no_bias(HIDDEN, FF_INNER, vb.pp("up_proj"))?,
            down: linear_no_bias(FF_INNER, HIDDEN, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(gate * self.up.forward(x)?)?)
    }
}

// ── Decoder layer ─────────────────────────────────────────────────────────────

struct LlamaLayer {
    ln1:  RmsNorm,
    attn: GQAttention,
    ln2:  RmsNorm,
    mlp:  MLP,
}

impl LlamaLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1:  RmsNorm::load(HIDDEN, vb.pp("input_layernorm"))?,
            attn: GQAttention::load(vb.pp("self_attn"))?,
            ln2:  RmsNorm::load(HIDDEN, vb.pp("post_attention_layernorm"))?,
            mlp:  MLP::load(vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = (x + self.attn.forward(&self.ln1.forward(x)?, cos, sin)?)?;
        &h + self.mlp.forward(&self.ln2.forward(&h)?)?
    }
}

// ── Public encoder ─────────────────────────────────────────────────────────────

pub struct LlamaTextEncoder {
    embed:     candle_nn::Embedding,
    layers:    Vec<LlamaLayer>,
    tokenizer: Tokenizer,
    device:    Device,
}

impl LlamaTextEncoder {
    /// Load from a directory containing tokenizer.json and *.safetensors shards.
    pub fn load(dir: &Path, device: &Device) -> AResult<Self> {
        // Collect all model shards sorted by name
        let mut shards: Vec<String> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| {
                let p = e.ok()?.path();
                let n = p.file_name()?.to_str()?.to_owned();
                if n.ends_with(".safetensors") && (n.contains("model") || n == "model.safetensors") {
                    p.to_str().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect();
        shards.sort();
        anyhow::ensure!(!shards.is_empty(),
            "No model safetensors in {}", dir.display());

        info!("LLaMA: found {} shard(s)", shards.len());
        for s in &shards {
            debug!("  shard: {s}");
        }

        let refs: Vec<&str> = shards.iter().map(String::as_str).collect();
        info!("LLaMA: mmap-loading weights (dtype=f32)");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&refs, DType::F32, device)
                .context("loading LLaMA weights")?
        };
        let m = vb.pp("model");

        info!("LLaMA: loading embed_tokens ({VOCAB}×{HIDDEN})");
        let embed_t = embedding(VOCAB, HIDDEN, m.pp("embed_tokens"))
            .context("embed_tokens")?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            debug!("LLaMA: loading layer {i}/{N_LAYERS}");
            layers.push(
                LlamaLayer::load(m.pp("layers").pp(&i.to_string()))
                    .with_context(|| format!("layers.{i}"))?,
            );
        }
        info!("LLaMA: all {N_LAYERS} layers loaded");

        let tok_path = dir.join("tokenizer.json");
        info!("LLaMA: loading tokenizer from {}", tok_path.display());
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        Ok(Self { embed: embed_t, layers, tokenizer, device: device.clone() })
    }

    pub fn n_params(&self) -> usize {
        let embed      = VOCAB * HIDDEN;
        let per_layer  = HIDDEN * HIDDEN                    // q_proj
            + HIDDEN * (N_KV_HEADS * HEAD_DIM)             // k_proj
            + HIDDEN * (N_KV_HEADS * HEAD_DIM)             // v_proj
            + HIDDEN * HIDDEN                              // o_proj
            + HIDDEN * FF_INNER                            // gate_proj
            + HIDDEN * FF_INNER                            // up_proj
            + FF_INNER * HIDDEN                            // down_proj
            + HIDDEN                                       // input_layernorm
            + HIDDEN;                                      // post_attention_layernorm
        embed + N_LAYERS * per_layer
    }

    /// Encode text → [1, seq_len, 6144].
    pub fn encode(&self, text: &str, seq_len: usize) -> AResult<Tensor> {
        let enc = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let t = ids.len().max(1);

        let input = Tensor::from_vec(ids, (1, t), &self.device)?;
        let mut h = self.embed.forward(&input)
            .map_err(|e| anyhow::anyhow!("embed: {e}"))?;  // [1, T, HIDDEN] bf16

        let (cos, sin) = build_rope(t, &self.device)
            .map_err(|e| anyhow::anyhow!("rope: {e}"))?;

        let mut extracted: Vec<Tensor> = Vec::with_capacity(EXTRACT_AT.len());
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin)
                .map_err(|e| anyhow::anyhow!("layer {i}: {e}"))?;
            if EXTRACT_AT.contains(&i) {
                extracted.push(h.clone());
            }
        }

        // [1, T, HIDDEN] × 2 → [1, T, 6144]
        let feats = Tensor::cat(&extracted, D::Minus1)
            .map_err(|e| anyhow::anyhow!("cat: {e}"))?;

        // Pool tokens → [T, 6144] → temporal_pool → [seq_len, 6144]
        let feats_2d = feats.squeeze(0)
            .map_err(|e| anyhow::anyhow!("squeeze: {e}"))?;
        let pooled = crate::features::temporal_pool(&feats_2d, seq_len)
            .map_err(|e| anyhow::anyhow!("pool: {e}"))?;
        Ok(pooled.unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("unsqueeze: {e}"))?)
    }
}
