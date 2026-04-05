/// features.rs — tensor helper utilities and hash-based demo text encoder.
use candle_core::{DType, Device, Result, Tensor};
use sha2::{Digest, Sha256};

use crate::clip_encoder::ClipVisualEncoder;

// ── Temporal pooling ──────────────────────────────────────────────────────────

/// Nearest-neighbour resample [T_src, D] → [target_len, D].
pub fn temporal_pool(feats: &Tensor, target_len: usize) -> Result<Tensor> {
    let t_src = feats.dim(0)?;
    if t_src == target_len {
        return Ok(feats.clone());
    }
    let mut rows = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let src = if target_len <= 1 {
            0
        } else {
            (i * (t_src - 1) / (target_len - 1)).min(t_src - 1)
        };
        rows.push(feats.narrow(0, src, 1)?.squeeze(0)?);
    }
    Tensor::stack(&rows, 0)
}

// ── TRIBE group_mean ──────────────────────────────────────────────────────────

/// Group-mean over layers [0.5, 0.75, 1.0]:
///   group0 = h0
///   group1 = (h1 + h2) / 2
///   output = cat([group0, group1], dim=-1)   →  2× feature dim
pub fn tribe_group_mean(h0: &Tensor, h1: &Tensor, h2: &Tensor) -> Result<Tensor> {
    let group1 = ((h1 + h2)? * 0.5)?;
    Tensor::cat(&[h0, &group1], candle_core::D::Minus1)
}

// ── Region statistics ─────────────────────────────────────────────────────────

pub const REGION_RANGES: &[(&str, usize, usize)] = &[
    ("visual",     0,     3600),
    ("auditory",   3600,  6800),
    ("language",   6800,  10500),
    ("prefrontal", 10500, 14000),
    ("motor",      14000, 17200),
    ("parietal",   17200, 20484),
];

#[derive(serde::Serialize, Clone)]
pub struct RegionStat {
    pub mean:           f32,
    pub std:            f32,
    pub rel_activation: f32,
    pub peak:           f32,
    pub n_vertices:     usize,
}

#[derive(serde::Serialize, Clone)]
pub struct GlobalStat {
    pub global_mean: f32,
    pub global_std:  f32,
    pub global_min:  f32,
    pub global_max:  f32,
}

/// Compute per-region stats from [T, N_VERTICES] activations (averaged over T).
pub fn region_stats(act: &[f32], n_vertices: usize) -> (std::collections::HashMap<String, RegionStat>, GlobalStat) {
    let t = act.len() / n_vertices;

    // Mean over time → [N_VERTICES]
    let mut mean_t = vec![0f32; n_vertices];
    for ti in 0..t {
        for vi in 0..n_vertices {
            mean_t[vi] += act[ti * n_vertices + vi];
        }
    }
    let t_f = t as f32;
    for v in &mut mean_t {
        *v /= t_f;
    }

    let g_mean: f32 = mean_t.iter().sum::<f32>() / n_vertices as f32;
    let g_var: f32 = mean_t.iter().map(|x| (x - g_mean).powi(2)).sum::<f32>() / n_vertices as f32;
    let g_std = g_var.sqrt() + 1e-8;
    let g_min = mean_t.iter().cloned().fold(f32::INFINITY, f32::min);
    let g_max = mean_t.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut stats = std::collections::HashMap::new();
    for &(name, lo, hi) in REGION_RANGES {
        let hi = hi.min(n_vertices);
        let v = &mean_t[lo..hi];
        let rm: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let rs = (v.iter().map(|x| (x - rm).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let mut abs_v: Vec<f32> = v.iter().map(|x| x.abs()).collect();
        abs_v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let peak = abs_v[((abs_v.len() as f32 * 0.95) as usize).min(abs_v.len().saturating_sub(1))];
        stats.insert(name.to_string(), RegionStat {
            mean: rm,
            std: rs,
            rel_activation: (rm - g_mean) / g_std,
            peak,
            n_vertices: hi - lo,
        });
    }

    (stats, GlobalStat {
        global_mean: g_mean,
        global_std: g_std - 1e-8,
        global_min: g_min,
        global_max: g_max,
    })
}

// ── CLIP visual features (real encoder) ──────────────────────────────────────

/// Encode a single image with CLIP ViT-L/14 → [1, seq_len, 2816].
/// The same CLS embedding is tiled across all seq_len timesteps.
pub fn visual_features_clip(
    bytes: &[u8],
    seq_len: usize,
    enc: &ClipVisualEncoder,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let feat = enc.encode_image(bytes)?;            // [1, 1, 2816]
    let feat_1d = feat.squeeze(0)?.squeeze(0)?;     // [2816]
    let rows: Vec<Tensor> = (0..seq_len).map(|_| feat_1d.clone()).collect();
    let stacked = Tensor::stack(&rows, 0)?;         // [seq_len, 2816]
    let out = stacked.unsqueeze(0)?;                // [1, seq_len, 2816]
    Ok(out.to_device(device)?)
}

// ── Hash-based demo visual features (no V-JEPA2 required) ───────────────────

const VISUAL_FEAT_DIM: usize = 2816;

/// Deterministic pseudo-V-JEPA2 visual features from raw bytes.
/// Returns a [1, seq_len, 2816] tensor.
pub fn visual_features_demo(bytes: &[u8], seq_len: usize, device: &Device) -> Result<Tensor> {
    let chunk_size = (bytes.len() / seq_len.max(1)).max(1);
    let mut feats: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
    for fi in 0..seq_len {
        let start = (fi * chunk_size).min(bytes.len().saturating_sub(1));
        let end   = ((fi + 1) * chunk_size).min(bytes.len());
        let slice = if start < end { &bytes[start..end] } else { &bytes[..bytes.len().min(16)] };
        let mut hasher = Sha256::new();
        hasher.update(&(fi as u32).to_le_bytes());
        hasher.update(slice);
        let hash = hasher.finalize();
        let seed = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        let raw = lcg_normal_n(seed, VISUAL_FEAT_DIM);
        // z-score normalise
        let mean = raw.iter().sum::<f32>() / raw.len() as f32;
        let std  = (raw.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / raw.len() as f32).sqrt() + 1e-8;
        feats.push(raw.into_iter().map(|x| ((x - mean) / std).clamp(-5.0, 5.0)).collect());
    }
    let flat: Vec<f32> = feats.into_iter().flatten().collect();
    Tensor::from_vec(flat, (1, seq_len, VISUAL_FEAT_DIM), device)?.to_dtype(DType::F32)
}

// ── Hash-based demo text features (no LLaMA required) ────────────────────────

const TEXT_FEAT_DIM: usize = 6144;
const PROJ_IN_DIM:   usize = 768;

/// Deterministic pseudo-LLaMA text features.  Returns a [1, seq_len, 6144] tensor.
pub fn text_to_features_demo(text: &str, seq_len: usize, device: &Device) -> Result<Tensor> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let words: Vec<&str> = if words.is_empty() { vec![text] } else { words };
    let n = words.len().min(seq_len).max(1);

    // Fixed projection matrix seeded with 42
    let proj = build_projection_matrix();

    let mut feats: Vec<Vec<f32>> = Vec::with_capacity(n);
    for w in &words[..n] {
        let mut hasher = Sha256::new();
        hasher.update(w.as_bytes());
        let hash = hasher.finalize();
        let seed = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        let v768 = lcg_normal_768(seed);

        // Project 768 → 6144
        let row: Vec<f32> = (0..TEXT_FEAT_DIM)
            .map(|j| {
                v768.iter()
                    .zip(proj[j * PROJ_IN_DIM..(j + 1) * PROJ_IN_DIM].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .collect();
        feats.push(row);
    }

    // Z-score normalise each feature
    for row in &mut feats {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let std = (row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32)
            .sqrt()
            + 1e-8;
        for x in row.iter_mut() {
            *x = ((*x - mean) / std).clamp(-5.0, 5.0);
        }
    }

    // Interpolate to seq_len rows
    let pooled = pool_feat_rows(&feats, seq_len);  // [seq_len, 6144]

    let flat: Vec<f32> = pooled.into_iter().flatten().collect();
    Tensor::from_vec(flat, (1, seq_len, TEXT_FEAT_DIM), device)?
        .to_dtype(DType::F32)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn build_projection_matrix() -> Vec<f32> {
    // Seeded 42 LCG → [TEXT_FEAT_DIM * PROJ_IN_DIM] normal-ish floats
    // Scale by 1/sqrt(PROJ_IN_DIM)
    let scale = 1.0 / (PROJ_IN_DIM as f32).sqrt();
    let mut state: u64 = 42;
    let n = TEXT_FEAT_DIM * PROJ_IN_DIM;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Box-Muller using two LCG draws
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 33) as f32 / (u32::MAX as f32);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        let z = (-2.0 * (u1.max(1e-7).ln())).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        out.push(z * scale);
    }
    out
}

fn lcg_normal_n(seed: u32, n: usize) -> Vec<f32> {
    let mut state: u64 = seed as u64 | 1;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 33) as f32 / (u32::MAX as f32);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        let r = (-2.0 * u1.max(1e-7).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push((r * theta.cos()).clamp(-3.0, 3.0));
        if out.len() < n { out.push((r * theta.sin()).clamp(-3.0, 3.0)); }
    }
    out
}

fn lcg_normal_768(seed: u32) -> Vec<f32> {
    let mut state: u64 = seed as u64 | 1;
    let mut out = Vec::with_capacity(PROJ_IN_DIM);
    for _ in 0..(PROJ_IN_DIM / 2) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 33) as f32 / (u32::MAX as f32);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        let r = (-2.0 * u1.max(1e-7).ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push((r * theta.cos()).clamp(-3.0, 3.0));
        out.push((r * theta.sin()).clamp(-3.0, 3.0));
    }
    out
}

fn pool_feat_rows(rows: &[Vec<f32>], target: usize) -> Vec<Vec<f32>> {
    let n = rows.len();
    if n == target {
        return rows.to_vec();
    }
    (0..target)
        .map(|i| {
            let src = if target <= 1 { 0 } else { (i * (n - 1) / (target - 1)).min(n - 1) };
            rows[src].clone()
        })
        .collect()
}
