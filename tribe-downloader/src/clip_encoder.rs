/// clip_encoder.rs — OpenAI CLIP ViT-L/14 visual encoder for TRIBE v2.
///
/// Weights: tribe-v2-weights/clip/model.safetensors (openai/clip-vit-large-patch14)
/// Download: python3 download_clip.py
///
/// Pipeline:
///   image bytes → resize 224×224 → CLIP ViT-L/14 → CLS [1024]
///                → fixed LCG projection → [2816] → tile to [1, seq_len, 2816]
///
/// The 2816-dim output feeds directly into FmriEncoder.proj_video.

use anyhow::{Context, Result as AResult};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{conv2d_no_bias, layer_norm, linear, Conv2dConfig, LayerNorm, Module, VarBuilder};

// ── ViT-L/14 architecture constants ──────────────────────────────────────────
const HIDDEN:    usize = 1024;
const HEADS:     usize = 16;
const HEAD_DIM:  usize = HIDDEN / HEADS;  // 64
const N_LAYERS:  usize = 24;
const FF_INNER:  usize = 4096;
const PATCH:     usize = 14;
const IMG:       usize = 224;
const N_PATCHES: usize = (IMG / PATCH) * (IMG / PATCH); // 256
const N_TOKENS:  usize = N_PATCHES + 1;                 // 257 (CLS + patches)
const LN_EPS:    f64   = 1e-5;

// Output dimension for FmriEncoder.proj_video
const PROJ_OUT:  usize = 2816;

// CLIP normalisation stats (ImageNet-derived)
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275,  0.40821073];
const CLIP_STD:  [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

// ── quick_gelu: x * σ(1.702·x) ───────────────────────────────────────────────
fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(&(x * 1.702_f64)?)?)
}

// ── Multi-head self-attention ─────────────────────────────────────────────────
struct ClipAttn {
    q:   candle_nn::Linear,
    k:   candle_nn::Linear,
    v:   candle_nn::Linear,
    out: candle_nn::Linear,
}

impl ClipAttn {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q:   linear(HIDDEN, HIDDEN, vb.pp("q_proj"))?,
            k:   linear(HIDDEN, HIDDEN, vb.pp("k_proj"))?,
            v:   linear(HIDDEN, HIDDEN, vb.pp("v_proj"))?,
            out: linear(HIDDEN, HIDDEN, vb.pp("out_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let scale = (HEAD_DIM as f64).powf(-0.5);

        let rshp = |l: &candle_nn::Linear| -> Result<Tensor> {
            l.forward(x)?
                .reshape((b, t, HEADS, HEAD_DIM))?
                .transpose(1, 2)  // [B, H, T, HD]
        };
        let q = rshp(&self.q)?;
        let k = rshp(&self.k)?;
        let v = rshp(&self.v)?;

        // Flatten B×H → 3-D for Metal compatibility
        let bh = b * HEADS;
        let q3 = q.reshape((bh, t, HEAD_DIM))?;
        let k3 = k.reshape((bh, t, HEAD_DIM))?;
        let v3 = v.reshape((bh, t, HEAD_DIM))?;

        let attn = candle_nn::ops::softmax(
            &(q3.matmul(&k3.transpose(1, 2)?)? * scale)?,
            D::Minus1,
        )?;
        let out = attn.matmul(&v3)?
            .reshape((b, HEADS, t, HEAD_DIM))?
            .transpose(1, 2)?
            .reshape((b, t, HIDDEN))?;
        self.out.forward(&out)
    }
}

// ── MLP with quick_gelu ───────────────────────────────────────────────────────
struct ClipMlp {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl ClipMlp {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(HIDDEN, FF_INNER, vb.pp("fc1"))?,
            fc2: linear(FF_INNER, HIDDEN, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&quick_gelu(&self.fc1.forward(x)?)?)
    }
}

// ── Transformer layer ─────────────────────────────────────────────────────────
struct ClipLayer {
    ln1:  LayerNorm,
    attn: ClipAttn,
    ln2:  LayerNorm,
    mlp:  ClipMlp,
}

impl ClipLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1:  layer_norm(HIDDEN, LN_EPS, vb.pp("layer_norm1"))?,
            attn: ClipAttn::load(vb.pp("self_attn"))?,
            ln2:  layer_norm(HIDDEN, LN_EPS, vb.pp("layer_norm2"))?,
            mlp:  ClipMlp::load(vb.pp("mlp"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?)?)?;
        &x + self.mlp.forward(&self.ln2.forward(&x)?)?
    }
}

// ── Public encoder ────────────────────────────────────────────────────────────
pub struct ClipVisualEncoder {
    cls_token: Tensor,             // [1, 1, HIDDEN]
    patch_emb: candle_nn::Conv2d,  // stride-14 conv, no bias
    pos_emb:   Tensor,             // [1, N_TOKENS, HIDDEN]
    pre_ln:    LayerNorm,
    layers:    Vec<ClipLayer>,
    post_ln:   LayerNorm,
    proj:      Vec<f32>,           // fixed [PROJ_OUT × HIDDEN] random projection
    device:    Device,
}

impl ClipVisualEncoder {
    /// Load from `tribe-v2-weights/clip/model.safetensors`.
    pub fn load(path: &str, device: &Device) -> AResult<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)
                .context("loading CLIP weights")?
        };
        let vm  = vb.pp("vision_model");
        let emb = vm.pp("embeddings");

        // class_embedding is stored as [HIDDEN]; reshape to [1,1,HIDDEN]
        let cls_token = emb.get((HIDDEN,), "class_embedding")
            .context("class_embedding")?
            .reshape((1, 1, HIDDEN))?;

        // Patch embedding: conv2d(in=3, out=1024, k=14, stride=14, no bias)
        let patch_emb = conv2d_no_bias(
            3, HIDDEN, PATCH,
            Conv2dConfig { stride: PATCH, padding: 0, dilation: 1, groups: 1 },
            emb.pp("patch_embedding"),
        ).context("patch_embedding")?;

        // Position embedding: nn.Embedding → weight [N_TOKENS, HIDDEN]
        let pos_emb = emb.get((N_TOKENS, HIDDEN), "position_embedding.weight")
            .context("position_embedding")?
            .unsqueeze(0)?;  // [1, N_TOKENS, HIDDEN]

        // Note: HuggingFace has a typo in the pre-LN key: "pre_layrnorm" (not "pre_layernorm")
        let pre_ln = layer_norm(HIDDEN, LN_EPS, vm.pp("pre_layrnorm"))
            .context("pre_layrnorm")?;

        let enc = vm.pp("encoder");
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            layers.push(
                ClipLayer::load(enc.pp("layers").pp(&i.to_string()))
                    .with_context(|| format!("encoder.layers.{i}"))?,
            );
        }

        let post_ln = layer_norm(HIDDEN, LN_EPS, vm.pp("post_layernorm"))
            .context("post_layernorm")?;

        Ok(Self {
            cls_token, patch_emb, pos_emb, pre_ln, layers, post_ln,
            proj: build_proj_matrix(),
            device: device.clone(),
        })
    }

    /// Encode raw image bytes → [1, 1, PROJ_OUT].
    /// Call this once per image; tile across seq_len in `visual_features_clip`.
    pub fn encode_image(&self, bytes: &[u8]) -> AResult<Tensor> {
        // Decode + resize
        let img = image::load_from_memory(bytes)
            .context("image decode")?
            .resize_exact(IMG as u32, IMG as u32, image::imageops::FilterType::CatmullRom)
            .to_rgb8();

        // CHW float tensor, CLIP-normalised
        let mut chw = vec![0f32; 3 * IMG * IMG];
        for y in 0..IMG {
            for x in 0..IMG {
                let p = img.get_pixel(x as u32, y as u32);
                for c in 0..3usize {
                    let v = p[c] as f32 / 255.0;
                    chw[c * IMG * IMG + y * IMG + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
                }
            }
        }
        let pixels = Tensor::from_vec(chw, (1, 3, IMG, IMG), &self.device)?
            .to_dtype(DType::F32)?;

        // Patch embed: [1,HIDDEN,16,16] → [1,256,HIDDEN]
        let patches = self.patch_emb.forward(&pixels)?
            .flatten_from(2)?   // [1, HIDDEN, 256]
            .transpose(1, 2)?;  // [1, 256, HIDDEN]

        // Prepend CLS: [1, 257, HIDDEN]
        let cls = self.cls_token.broadcast_as((1, 1, HIDDEN))?;
        let x = Tensor::cat(&[&cls, &patches], 1)?;

        // Add positional embedding
        let x = (x + &self.pos_emb)?;

        // Pre layer norm
        let x = self.pre_ln.forward(&x)?;

        // 24 transformer layers
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // Post layer norm
        let h = self.post_ln.forward(&h)?;

        // Extract CLS token: [1, HIDDEN]
        let cls_out = h.narrow(1, 0, 1)?.squeeze(1)?;

        // Project HIDDEN → PROJ_OUT using fixed matrix
        let proj_t = Tensor::from_slice(&self.proj, (PROJ_OUT, HIDDEN), &self.device)?;
        let out = cls_out.matmul(&proj_t.t()?)?;  // [1, PROJ_OUT]

        Ok(out.unsqueeze(1)?)  // [1, 1, PROJ_OUT]
    }
}

/// Fixed [PROJ_OUT × HIDDEN] random projection matrix, seeded at 43.
/// Preserves relative distances between CLIP embeddings (JL embedding).
fn build_proj_matrix() -> Vec<f32> {
    let scale = 1.0 / (HIDDEN as f32).sqrt();
    let mut state: u64 = 43;
    let n = PROJ_OUT * HIDDEN;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 33) as f32 / (u32::MAX as f32);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        let r = (-2.0 * u1.max(1e-7).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push(r * theta.cos() * scale);
        if out.len() < n { out.push(r * theta.sin() * scale); }
    }
    out.truncate(n);
    out
}
