/// wav2vec2bert.rs — Wav2Vec2Bert conformer encoder (facebook/w2v-bert-2.0).
///
/// Architecture:
///   feature_projection  (160 → 1024, with layer_norm)
///   24 conformer layers, each:
///     FFN1  (half-step residual)      ← ffn1_layer_norm, ffn1.*
///     Self-attention (relative key)   ← self_attn_layer_norm, self_attn.*
///     Convolutional module            ← conv_module.*
///     FFN2  (half-step residual)      ← ffn2_layer_norm, ffn2.*
///     Final layer norm                ← final_layer_norm
use anyhow::{Context, Result as AResult};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Module, VarBuilder};

// ── Model constants (facebook/w2v-bert-2.0 config.json) ──────────────────────

const HIDDEN:        usize = 1024;
const HEADS:         usize = 16;
const HEAD_DIM:      usize = HIDDEN / HEADS;  // 64
const FF_INNER:      usize = 4096;
const N_LAYERS:      usize = 24;
const FEAT_IN:       usize = 160;             // 80 mel × stride 2
const LEFT_MAX:      usize = 64;
const RIGHT_MAX:     usize = 8;
const N_DIST:        usize = LEFT_MAX + RIGHT_MAX + 1;  // 73
const DW_K:          usize = 31;
const DW_PAD:        usize = 15;  // (31-1)/2
const LN_EPS:        f64   = 1e-5;

// ── Feature projection ─────────────────────────────────��──────────────────────

struct FeatProj {
    proj: candle_nn::Linear,
    norm: LayerNorm,
}

impl FeatProj {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(FEAT_IN, LN_EPS, vb.pp("layer_norm"))?,
            proj: linear(FEAT_IN, HIDDEN, vb.pp("projection"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(&self.norm.forward(x)?)
    }
}

// ── FFN sub-block (intermediate_dense → swish → output_dense) ────────────���───

struct FFN {
    w_in:  candle_nn::Linear,
    w_out: candle_nn::Linear,
}

impl FFN {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w_in:  linear(HIDDEN, FF_INNER, vb.pp("intermediate_dense"))?,
            w_out: linear(FF_INNER, HIDDEN, vb.pp("output_dense"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w_in.forward(x)?;
        let h = h.mul(&candle_nn::ops::sigmoid(&h)?)?;  // silu / swish
        self.w_out.forward(&h)
    }
}

// ── Self-attention with relative key-position embeddings ──────────────────────

struct SelfAttn {
    lq:  candle_nn::Linear,
    lk:  candle_nn::Linear,
    lv:  candle_nn::Linear,
    lo:  candle_nn::Linear,
    dist_emb: Tensor,   // [N_DIST, HEAD_DIM]
}

impl SelfAttn {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lq:       linear(HIDDEN, HIDDEN, vb.pp("linear_q"))?,
            lk:       linear(HIDDEN, HIDDEN, vb.pp("linear_k"))?,
            lv:       linear(HIDDEN, HIDDEN, vb.pp("linear_v"))?,
            lo:       linear(HIDDEN, HIDDEN, vb.pp("linear_out"))?,
            dist_emb: vb.get((N_DIST, HEAD_DIM), "distance_embedding.weight")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let dev = x.device();

        let rshp = |l: &candle_nn::Linear| -> Result<Tensor> {
            l.forward(x)?
                .reshape((b, t, HEADS, HEAD_DIM))?
                .transpose(1, 2)   // [B, H, T, HD]
        };
        let q = rshp(&self.lq)?;
        let k = rshp(&self.lk)?;
        let v = rshp(&self.lv)?;

        let scale = (HEAD_DIM as f64).powf(-0.5);

        // Flatten B and H → 3-D matmul to avoid Metal 4-D batch matmul bug
        // with non-power-of-2 HEAD_DIM=64 on some Metal kernel paths.
        let bh = b * HEADS;
        let q3 = q.reshape((bh, t, HEAD_DIM))?;  // [B*H, T, HD]
        let k3 = k.reshape((bh, t, HEAD_DIM))?;
        let v3 = v.reshape((bh, t, HEAD_DIM))?;

        // Standard QK: [B*H, T, T]
        let qk = (q3.matmul(&k3.transpose(1, 2)?)? * scale)?;

        // Relative position: [B, H, T, N_DIST] flattened to [B*H, T, N_DIST]
        let s = (q3.matmul(
            &self.dist_emb
                .transpose(0, 1)?               // [HD, N_DIST]
                .unsqueeze(0)?
                .broadcast_as((bh, HEAD_DIM, N_DIST))?
        )? * scale)?;  // [B*H, T, N_DIST]

        // Build relative index matrix [T, T] once per call
        let rel_idx: Vec<u32> = (0..t).flat_map(|i| {
            (0..t).map(move |j| {
                let r = (j as i64 - i as i64)
                    .clamp(-(LEFT_MAX as i64), RIGHT_MAX as i64)
                    + LEFT_MAX as i64;
                r as u32
            })
        }).collect();
        let rel_idx = Tensor::from_vec(rel_idx, (t, t), dev)?; // [T, T]

        // Gather: rel_score[bh,i,j] = s[bh,i, rel_idx[i,j]]
        let mut rel_rows = Vec::with_capacity(t);
        for i in 0..t {
            let si    = s.narrow(1, i, 1)?.squeeze(1)?;         // [B*H, N_DIST]
            let idx_i = rel_idx.narrow(0, i, 1)?.squeeze(0)?;   // [T]
            rel_rows.push(si.index_select(&idx_i, 1)?);          // [B*H, T]
        }
        let rel_scores = Tensor::stack(&rel_rows, 1)?; // [B*H, T, T]

        let attn = candle_nn::ops::softmax(&(qk + rel_scores)?, D::Minus1)?;
        let out  = attn.matmul(&v3)?   // [B*H, T, HD]
            .reshape((b, HEADS, t, HEAD_DIM))?
            .transpose(1, 2)?
            .reshape((b, t, HIDDEN))?;
        self.lo.forward(&out)
    }
}

// ── Convolutional module ────────────────────────────��─────────────────────────
//
// Forward (matching HF Wav2Vec2BertConvolutionModule):
//   x [B,T,H] → layer_norm → .T → pw1 [B,2H,T] → GLU [B,H,T]
//             → depthwise [B,H,T] → .T → dw_norm → swish
//             → .T → pw2 [B,H,T] → .T  [B,T,H]

struct ConvModule {
    ln:      LayerNorm,
    pw1_w:   Tensor,   // [2H, H, 1]
    dw_w:    Tensor,   // [H,  1, K]
    dw_ln:   LayerNorm,
    pw2_w:   Tensor,   // [H,  H, 1]
}

impl ConvModule {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln:    layer_norm(HIDDEN, LN_EPS, vb.pp("layer_norm"))?,
            pw1_w: vb.get((2 * HIDDEN, HIDDEN, 1),   "pointwise_conv1.weight")?,
            dw_w:  vb.get((HIDDEN,     1, DW_K),      "depthwise_conv.weight")?,
            dw_ln: layer_norm(HIDDEN, LN_EPS, vb.pp("depthwise_layer_norm"))?,
            pw2_w: vb.get((HIDDEN,     HIDDEN, 1),    "pointwise_conv2.weight")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, T, H] → norm
        let h = self.ln.forward(x)?;
        // → [B, H, T]
        let h = h.transpose(1, 2)?;
        // Pointwise expand: [B, 2H, T]
        let h = h.conv1d(&self.pw1_w, 0, 1, 1, 1)?;
        // GLU: split and gate
        let c = HIDDEN;
        let h1 = h.narrow(1, 0, c)?;
        let h2 = h.narrow(1, c, c)?;
        let h = h1.mul(&candle_nn::ops::sigmoid(&h2)?)?;  // [B, H, T]
        // Depthwise conv: groups = HIDDEN
        let h = h.conv1d(&self.dw_w, DW_PAD, 1, 1, HIDDEN)?;  // [B, H, T]
        // Back to [B, T, H] for layer norm
        let h = h.transpose(1, 2)?;
        let h = self.dw_ln.forward(&h)?;
        let h = h.mul(&candle_nn::ops::sigmoid(&h)?)?; // swish
        // → [B, H, T] for pointwise project
        let h = h.transpose(1, 2)?;
        let h = h.conv1d(&self.pw2_w, 0, 1, 1, 1)?;  // [B, H, T]
        // → [B, T, H]
        h.transpose(1, 2)
    }
}

// ── Conformer encoder layer ───────────────────────────────────────────────────

struct ConformerLayer {
    ffn1_ln: LayerNorm,
    ffn1:    FFN,
    attn_ln: LayerNorm,
    attn:    SelfAttn,
    conv:    ConvModule,
    ffn2_ln: LayerNorm,
    ffn2:    FFN,
    fin_ln:  LayerNorm,
}

impl ConformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ffn1_ln: layer_norm(HIDDEN, LN_EPS, vb.pp("ffn1_layer_norm"))?,
            ffn1:    FFN::load(vb.pp("ffn1"))?,
            attn_ln: layer_norm(HIDDEN, LN_EPS, vb.pp("self_attn_layer_norm"))?,
            attn:    SelfAttn::load(vb.pp("self_attn"))?,
            conv:    ConvModule::load(vb.pp("conv_module"))?,
            ffn2_ln: layer_norm(HIDDEN, LN_EPS, vb.pp("ffn2_layer_norm"))?,
            ffn2:    FFN::load(vb.pp("ffn2"))?,
            fin_ln:  layer_norm(HIDDEN, LN_EPS, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // FFN1 (half-step)
        let x = (x + (self.ffn1.forward(&self.ffn1_ln.forward(x)?)? * 0.5)?)?;
        // Self-attention
        let x = (&x + self.attn.forward(&self.attn_ln.forward(&x)?)?)?;
        // Conv module
        let x = (&x + self.conv.forward(&x)?)?;
        // FFN2 (half-step)
        let x = (&x + (self.ffn2.forward(&self.ffn2_ln.forward(&x)?)? * 0.5)?)?;
        // Final norm
        self.fin_ln.forward(&x)
    }
}

// ── Public encoder ──────────────────────────────────��─────────────────────────

pub struct Wav2Vec2Bert {
    feat_proj:  FeatProj,
    layers:     Vec<ConformerLayer>,
    /// Conformer-layer indices to extract (1-based after each forward): 12, 18, 24
    extract_at: [usize; 3],
}

impl Wav2Vec2Bert {
    /// Load from a local safetensors file (HF cache).
    pub fn load(st_path: &str, device: &Device) -> AResult<Self> {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[st_path], DType::F32, device)?
        };

        let feat_proj = FeatProj::load(vb.pp("feature_projection"))
            .context("feature_projection")?;

        let enc = vb.pp("encoder");
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            layers.push(
                ConformerLayer::load(enc.pp("layers").pp(&i.to_string()))
                    .with_context(|| format!("encoder.layers.{i}"))?,
            );
        }

        Ok(Self { feat_proj, layers, extract_at: [12, 18, 24] })
    }

    pub fn n_params(&self) -> usize {
        let feat_proj  = FEAT_IN * HIDDEN + HIDDEN         // projection weight+bias
            + FEAT_IN * 2;                                 // layer_norm weight+bias
        let per_layer  = HIDDEN * HIDDEN * 4              // lq,lk,lv,lo
            + HIDDEN * 4                                   // their biases
            + N_DIST * HEAD_DIM                            // distance_embedding
            + HIDDEN * 2                                   // self_attn layer_norm
            + HIDDEN * FF_INNER + FF_INNER + FF_INNER * HIDDEN + HIDDEN  // ffn1 w+b
            + HIDDEN * 2                                   // ffn1 layer_norm
            + HIDDEN * FF_INNER + FF_INNER + FF_INNER * HIDDEN + HIDDEN  // ffn2 w+b
            + HIDDEN * 2                                   // ffn2 layer_norm
            + HIDDEN * 2 * HIDDEN + HIDDEN * 2             // conv pw1 (gated) w+b
            + HIDDEN * DW_K                                // depthwise conv weight
            + HIDDEN                                       // depthwise conv bias
            + HIDDEN * 2                                   // depthwise layer_norm
            + HIDDEN * HIDDEN + HIDDEN                     // pointwise_conv2 w+b
            + HIDDEN * 2;                                  // final layer_norm
        feat_proj + N_LAYERS * per_layer
    }

    /// Encode stacked-mel features `x` of shape [1, T, 160].
    /// Returns three hidden-state tensors at conformer layers 12, 18, 24
    /// (each [1, T, 1024]).
    pub fn encode(&self, x: &Tensor) -> Result<[Tensor; 3]> {
        let mut h = self.feat_proj.forward(x)?;
        let mut out: [Option<Tensor>; 3] = [None, None, None];

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            for (ci, &tgt) in self.extract_at.iter().enumerate() {
                if i + 1 == tgt {
                    out[ci] = Some(h.clone());
                }
            }
        }

        Ok([
            out[0].take().unwrap(),
            out[1].take().unwrap(),
            out[2].take().unwrap(),
        ])
    }
}
