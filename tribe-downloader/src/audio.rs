/// audio.rs — audio file decoding and mel-spectrogram computation.
///
/// Produces 160-dim stacked log mel features matching SeamlessM4TFeatureExtractor:
///   * Povey window (400 samples, loaded from povey_window_f32.bin)
///   * Mel filterbank (257 × 80, loaded from mel_filters_f32.bin)
///   * Pre-emphasis coefficient 0.97
///   * Frame length 400, hop 160, FFT 512
///   * stride=2 frame stacking → [T//2, 160]

use anyhow::{Context, Result};
use std::path::Path;

// ── Audio file decoding (symphonia) ──────────────────────────────────────────

/// Decode any audio file bytes to mono f32 PCM at 16 000 Hz.
pub fn decode_audio(bytes: &[u8]) -> Result<Vec<f32>> {
    decode_audio_owned(bytes.to_vec())
}

fn decode_audio_owned(bytes: Vec<u8>) -> Result<Vec<f32>> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = std::io::Cursor::new(bytes);
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = Hint::new();
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("audio format probe failed")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("no audio track found")?
        .clone();
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("codec not supported")?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        match format.next_packet() {
            Ok(packet) => {
                let decoded = match decoder.decode(&packet) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        let n_frames = buf.frames();
                        for f in 0..n_frames {
                            // Average channels → mono
                            let s: f32 = (0..channels).map(|c| buf.chan(c)[f]).sum::<f32>()
                                / channels as f32;
                            samples.push(s);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        let n_frames = buf.frames();
                        for f in 0..n_frames {
                            let s: f32 = (0..channels)
                                .map(|c| buf.chan(c)[f] as f32 / 32768.0)
                                .sum::<f32>()
                                / channels as f32;
                            samples.push(s);
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        let n_frames = buf.frames();
                        for f in 0..n_frames {
                            let s: f32 = (0..channels)
                                .map(|c| buf.chan(c)[f] as f32 / 2_147_483_648.0)
                                .sum::<f32>()
                                / channels as f32;
                            samples.push(s);
                        }
                    }
                    _ => {
                        // Convert via f64
                        let mut buf_f32 = decoded.make_equivalent::<f32>();
                        decoded.convert(&mut buf_f32);
                        let n_frames = buf_f32.frames();
                        for f in 0..n_frames {
                            let s: f32 = (0..channels).map(|c| buf_f32.chan(c)[f]).sum::<f32>()
                                / channels as f32;
                            samples.push(s);
                        }
                    }
                }
            }
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(_) => break,
        }
    }

    // Resample to 16 000 Hz using linear interpolation if needed
    if sample_rate != 16000 {
        samples = resample_linear(&samples, sample_rate as usize, 16000);
    }

    Ok(samples)
}

fn resample_linear(samples: &[f32], src_rate: usize, dst_rate: usize) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let lo = src_pos as usize;
        let hi = (lo + 1).min(samples.len() - 1);
        let frac = src_pos - lo as f64;
        out.push(samples[lo] * (1.0 - frac) as f32 + samples[hi] * frac as f32);
    }
    out
}

// ── Mel-spectrogram computation ───────────────────────────────────────────────

pub struct MelSpec {
    window:      Vec<f32>,           // 400 Povey values
    mel_filters: Vec<f32>,           // [257 × 80] row-major (mel_filters[f*80 + m])
    fft_size:    usize,              // 512
    hop:         usize,              // 160
    frame_len:   usize,              // 400
    n_mels:      usize,              // 80
    n_fft_bins:  usize,              // 257  (fft_size/2 + 1)
    preemphasis: f32,                // 0.97
    mel_floor:   f32,
    stride:      usize,              // 2 → stacked dim = 160
}

impl MelSpec {
    /// Load pre-computed Povey window and mel filterbank saved by convert_ckpt.py.
    pub fn load(weights_dir: &Path) -> Result<Self> {
        let win_path = weights_dir.join("povey_window_f32.bin");
        let mel_path = weights_dir.join("mel_filters_f32.bin");

        let window = read_f32_bin(&win_path)
            .with_context(|| format!("povey_window_f32.bin not found at {:?}; run convert_ckpt.py first", win_path))?;
        let mel_filters = read_f32_bin(&mel_path)
            .with_context(|| format!("mel_filters_f32.bin not found at {:?}; run convert_ckpt.py first", mel_path))?;

        assert_eq!(window.len(), 400, "Povey window must have 400 samples");
        assert_eq!(mel_filters.len(), 257 * 80, "mel_filters must be 257×80");

        Ok(Self {
            window,
            mel_filters,
            fft_size:   512,
            hop:        160,
            frame_len:  400,
            n_mels:     80,
            n_fft_bins: 257,
            preemphasis: 0.97,
            mel_floor:  1.192_092_9e-7,
            stride:     2,
        })
    }

    /// Convert mono 16 kHz f32 waveform to [T//stride, n_mels*stride] mel features.
    pub fn compute(&self, waveform: &[f32]) -> Vec<Vec<f32>> {
        use rustfft::{num_complex::Complex, FftPlanner};

        if waveform.is_empty() {
            return Vec::new();
        }

        // 1. Scale (Kaldi 16-bit compliance)
        let mut sig: Vec<f32> = waveform.iter().map(|x| x * 32768.0).collect();

        // 2. Global pre-emphasis
        for i in (1..sig.len()).rev() {
            sig[i] -= self.preemphasis * sig[i - 1];
        }

        // 3. Framing (no center padding)
        if sig.len() < self.frame_len {
            return Vec::new();
        }
        let n_frames = (sig.len() - self.frame_len) / self.hop + 1;

        // 4. FFT planner (reused across frames)
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.fft_size);

        let mut mel_frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);

        for fi in 0..n_frames {
            let start = fi * self.hop;
            let frame = &sig[start..start + self.frame_len];

            // 4a. Remove DC offset
            let dc: f32 = frame.iter().sum::<f32>() / self.frame_len as f32;

            // 4b. Apply Povey window
            let mut buf: Vec<Complex<f32>> = (0..self.fft_size).map(|i| {
                if i < self.frame_len {
                    Complex::new((frame[i] - dc) * self.window[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0) // zero-pad
                }
            }).collect();

            // 4c. FFT
            fft.process(&mut buf);

            // 4d. Power spectrum (one-sided)
            let power: Vec<f32> = buf[..self.n_fft_bins]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // 4e. Mel filterbank: [n_fft_bins × n_mels] — stored as mel_filters[f][m]
            let mut mel: Vec<f32> = vec![0.0; self.n_mels];
            for f in 0..self.n_fft_bins {
                let p = power[f];
                let row_off = f * self.n_mels;
                for m in 0..self.n_mels {
                    mel[m] += p * self.mel_filters[row_off + m];
                }
            }

            // 4f. Log
            for v in &mut mel {
                *v = v.max(self.mel_floor).ln();
            }

            mel_frames.push(mel);
        }

        // 5. Stride-2 stacking: pair consecutive frames → 160-dim
        let n_pairs = mel_frames.len() / self.stride;
        let out_dim = self.n_mels * self.stride;
        (0..n_pairs)
            .map(|i| {
                let mut row = Vec::with_capacity(out_dim);
                row.extend_from_slice(&mel_frames[i * self.stride]);
                row.extend_from_slice(&mel_frames[i * self.stride + 1]);
                row
            })
            .collect()
    }
}

fn read_f32_bin(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        anyhow::bail!("binary file {:?} length not multiple of 4", path);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
