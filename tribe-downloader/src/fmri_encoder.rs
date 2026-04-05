/// fmri_encoder.rs — TRIBEv2 FmriEncoder implemented with candle.
///
/// Matches the exact state_dict structure of best.safetensors
/// (produced by convert_ckpt.py from the PyTorch Lightning checkpoint).
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Module, VarBuilder};

// ── Architecture constants ────────────────────────────────────────────────────
const DIM:       usize = 1152;
const HEADS:     usize = 8;
const HEAD_DIM:  usize = DIM / HEADS; // 144
const DEPTH:     usize = 8;           // attention layers; equal number of FF layers
const FF_INNER:  usize = DIM * 4;     // 4608
const N_LAYERS:  usize = DEPTH * 2;   // 16 total (alternating attn / ff)
const MAX_SEQ:   usize = 1024;
pub const N_VERT: usize = 20484;
const ROT_DIM:   usize = 72;          // first 72 of 144 head dims are rotated

// ── ScaleNorm ─────────────────────────────────────────────────────────────────

struct ScaleNorm {
    g: Tensor,
}

impl ScaleNorm {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self { g: vb.get((1,), "g")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.clamp(1e-8f64, f64::MAX)?;
        x.broadcast_div(&norm)?.broadcast_mul(&self.g)
    }
}

// ── Rotary positional embedding ───────────────────────────────────────────────

struct RotaryEmb {
    inv_freq: Tensor,
}

impl RotaryEmb {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self { inv_freq: vb.get((36,), "inv_freq")? })
    }

    /// Returns (cos, sin) each of shape [1, 1, T, 72].
    fn forward(&self, t: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let pos = Tensor::arange(0u32, t as u32, device)?.to_dtype(DType::F32)?;
        let freqs = pos.unsqueeze(1)?.broadcast_mul(&self.inv_freq.unsqueeze(0)?)?; // [T,36]
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // [T,72]
        let cos = emb.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = emb.sin()?.unsqueeze(0)?.unsqueeze(0)?;
        Ok((cos, sin))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_rot  = q.narrow(D::Minus1, 0, ROT_DIM)?;
    let q_pass = q.narrow(D::Minus1, ROT_DIM, HEAD_DIM - ROT_DIM)?;
    let k_rot  = k.narrow(D::Minus1, 0, ROT_DIM)?;
    let k_pass = k.narrow(D::Minus1, ROT_DIM, HEAD_DIM - ROT_DIM)?;

    let q_new = (q_rot.broadcast_mul(cos)? + rotate_half(&q_rot)?.broadcast_mul(sin)?)?;
    let k_new = (k_rot.broadcast_mul(cos)? + rotate_half(&k_rot)?.broadcast_mul(sin)?)?;

    Ok((
        Tensor::cat(&[&q_new, &q_pass], D::Minus1)?,
        Tensor::cat(&[&k_new, &k_pass], D::Minus1)?,
    ))
}

// ── Attention ─────────────────────────────────────────────────────────────────

struct Attention {
    to_q:   candle_nn::Linear,
    to_k:   candle_nn::Linear,
    to_v:   candle_nn::Linear,
    to_out: candle_nn::Linear,
}

impl Attention {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            to_q:   linear_no_bias(DIM, DIM, vb.pp("to_q"))?,
            to_k:   linear_no_bias(DIM, DIM, vb.pp("to_k"))?,
            to_v:   linear_no_bias(DIM, DIM, vb.pp("to_v"))?,
            to_out: linear_no_bias(DIM, DIM, vb.pp("to_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let reshape = |proj: &candle_nn::Linear| -> Result<Tensor> {
            proj.forward(x)?
                .reshape((b, t, HEADS, HEAD_DIM))?
                .transpose(1, 2)   // [B, H, T, HD]
        };

        let (q, k, v) = (reshape(&self.to_q)?, reshape(&self.to_k)?, reshape(&self.to_v)?);
        let (q, k) = apply_rotary(&q, &k, cos, sin)?;

        let scale = (HEAD_DIM as f64).powf(-0.5);

        // Metal's 4-D batch matmul kernel rejects non-power-of-2 HEAD_DIM=144.
        // Squeeze the batch dim → 3-D matmul [H, T, HD] which uses a different
        // (working) Metal kernel path, then unsqueeze back.
        let q3 = q.reshape((b * HEADS, t, HEAD_DIM))?;
        let k3 = k.reshape((b * HEADS, t, HEAD_DIM))?;
        let v3 = v.reshape((b * HEADS, t, HEAD_DIM))?;
        let attn = candle_nn::ops::softmax(
            &(q3.matmul(&k3.transpose(1, 2)?)? * scale)?,
            D::Minus1,
        )?;  // [B*H, T, T]
        let out = attn.matmul(&v3)?  // [B*H, T, HD]
            .reshape((b, HEADS, t, HEAD_DIM))?
            .transpose(1, 2)?
            .reshape((b, t, DIM))?;
        self.to_out.forward(&out)
    }
}

// ── Feed-forward ──────────────────────────────────────────────────────────────

struct FeedForward {
    w1: candle_nn::Linear,
    w2: candle_nn::Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder) -> Result<Self> {
        let ff = vb.pp("ff");
        Ok(Self {
            // State-dict paths: ff.0.0.weight  and  ff.2.weight
            w1: linear_no_bias(DIM, FF_INNER, ff.pp("0").pp("0"))?,
            w2: linear_no_bias(FF_INNER, DIM, ff.pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2.forward(&self.w1.forward(x)?.gelu()?)
    }
}

// ── Transformer encoder layer ─────────────────────────────────────────────────

enum Block {
    Attn(Attention),
    FF(FeedForward),
}

struct EncoderLayer {
    norm:  ScaleNorm,
    block: Block,
    rscale: Tensor,  // learned residual scale [DIM]
}

struct TransformerEncoder {
    rotary:     RotaryEmb,
    layers:     Vec<EncoderLayer>,
    final_norm: ScaleNorm,
}

impl TransformerEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let rotary = RotaryEmb::load(vb.pp("rotary_pos_emb"))?;
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let lv = vb.pp("layers").pp(&i.to_string());
            let norm   = ScaleNorm::load(lv.pp("0").pp("0"))?;
            let block  = if i % 2 == 0 {
                Block::Attn(Attention::load(lv.pp("1"))?)
            } else {
                Block::FF(FeedForward::load(lv.pp("1"))?)
            };
            let rscale = lv.pp("2").get((DIM,), "residual_scale")?;
            layers.push(EncoderLayer { norm, block, rscale });
        }
        let final_norm = ScaleNorm::load(vb.pp("final_norm"))?;
        Ok(Self { rotary, layers, final_norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let (cos, sin) = self.rotary.forward(t, x.device())?;
        let mut h = x.clone();
        for layer in &self.layers {
            let normed = layer.norm.forward(&h)?;
            let out = match &layer.block {
                Block::Attn(a) => a.forward(&normed, &cos, &sin)?,
                Block::FF(ff)  => ff.forward(&normed)?,
            };
            h = (h + out.broadcast_mul(&layer.rscale)?)?;
        }
        self.final_norm.forward(&h)
    }
}

// ── Predictor ─────────────────────────────────────────────────────────────────

struct Predictor {
    weights: Tensor,  // [1, 2048, N_VERT]
    bias:    Tensor,  // [1, N_VERT]
}

impl Predictor {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weights: vb.get((1, 2048, N_VERT), "weights")?,
            bias:    vb.get((1, N_VERT), "bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, 2048]  weights[0]: [2048, N_VERT]  bias[0]: [N_VERT]
        let (b, t, _) = x.dims3()?;
        let w = self.weights.narrow(0, 0, 1)?.squeeze(0)?; // [2048, N_VERT]
        let bias = self.bias.narrow(0, 0, 1)?.squeeze(0)?; // [N_VERT]
        // Reshape to 2D for matmul, then restore
        let out = x.reshape((b * t, 2048))?.matmul(&w)?; // [B*T, N_VERT]
        out.reshape((b, t, N_VERT))?.broadcast_add(&bias)
    }
}

// ── FmriEncoder (public) ──────────────────────────────────────────────────────

pub struct FmriEncoder {
    time_pos:  Tensor,                // [1, MAX_SEQ, DIM]
    proj_text: candle_nn::Linear,     // 6144 → 384
    proj_audio: candle_nn::Linear,    // 2048 → 384
    proj_video: candle_nn::Linear,    // 2816 → 384
    encoder:   TransformerEncoder,
    low_rank:  candle_nn::Linear,     // DIM → 2048, no bias
    predictor: Predictor,
    device:    Device,
}

impl FmriEncoder {
    pub fn load(safetensors_path: &str, device: &Device) -> anyhow::Result<Self> {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[safetensors_path],
                DType::F32,
                device,
            )?
        };

        let proj = vb.pp("projectors");
        Ok(Self {
            time_pos:   vb.get((1, MAX_SEQ, DIM), "time_pos_embed")?,
            proj_text:  linear(6144, 384, proj.pp("text"))?,
            proj_audio: linear(2048, 384, proj.pp("audio"))?,
            proj_video: linear(2816, 384, proj.pp("video"))?,
            encoder:    TransformerEncoder::load(vb.pp("encoder"))?,
            low_rank:   linear_no_bias(DIM, 2048, vb.pp("low_rank_head"))?,
            predictor:  Predictor::load(vb.pp("predictor"))?,
            device:     device.clone(),
        })
    }

    /// Returns [B, T, N_VERT].
    pub fn forward(
        &self,
        text:  Option<&Tensor>,
        audio: Option<&Tensor>,
        video: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ref_feat = text.or(audio).or(video)
            .ok_or_else(|| candle_core::Error::Msg("No input modality provided".into()))?;
        let b = ref_feat.dim(0)?;
        let t = ref_feat.dim(1)?;

        let zeros = |d: usize| Tensor::zeros((b, t, d), DType::F32, &self.device);

        let tf = self.proj_text.forward( text .unwrap_or(&zeros(6144)?))?;
        let af = self.proj_audio.forward(audio.unwrap_or(&zeros(2048)?))?;
        let vf = self.proj_video.forward(video.unwrap_or(&zeros(2816)?))?;

        let x = Tensor::cat(&[&tf, &af, &vf], 2)?;            // [B, T, DIM]
        let x = (x + self.time_pos.narrow(1, 0, t)?)?;
        let x = self.encoder.forward(&x)?;                     // [B, T, DIM]
        let x = self.low_rank.forward(&x)?;                    // [B, T, 2048]
        self.predictor.forward(&x)                             // [B, T, N_VERT]
    }

    pub fn n_params(&self) -> usize {
        // Fixed count from state dict (177 M)
        177_205_397
    }
}
