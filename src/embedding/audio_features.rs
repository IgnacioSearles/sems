//! Port of the Hugging Face `Gemma4AudioFeatureExtractor`: log-mel spectrogram of 16 kHz audio.
//!
//! 20 ms frames every 10 ms (after 10 ms of leading zeros), periodic Hann window, 512-point real
//! FFT magnitude, 128 HTK mel filters over 0-8 kHz, `ln(mel + 0.001)`. Precision follows the
//! reference exactly: the windowed frame is computed in f32, then (as NumPy's FFT promotes) the FFT,
//! magnitude, mel projection and log run in f64 before the result is cast back to f32.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: usize = 16_000;
/// The feature extractor truncates longer audio; the model sees at most 30 seconds per input.
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE;
const MEL_BINS: usize = 128;
const FRAME_LENGTH: usize = 320;
const HOP_LENGTH: usize = 160;
const FFT_LENGTH: usize = 512;
const FREQUENCY_BINS: usize = FFT_LENGTH / 2 + 1;
const MIN_FREQUENCY: f64 = 0.0;
const MAX_FREQUENCY: f64 = 8_000.0;
const MEL_FLOOR: f64 = 1e-3;
/// Batches pad audio to a multiple of this many samples (upstream's `pad_to_multiple_of`).
const PAD_TO_MULTIPLE_OF: usize = 128;

/// Model input for one clip: `[frames, 128]` features and a per-frame validity mask.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    pub features: Vec<f32>,
    pub mask: Vec<i64>,
    pub frames: usize,
}

impl AudioFeatures {
    pub const MEL_BINS: usize = MEL_BINS;
    /// The audio graph's fixed input length (AUDIO_FRAMES in tools/export/export_onnx.py). A full
    /// 30 s clip yields 2999 frames, so every clip fits after zero-padding.
    pub const WINDOW_FRAMES: usize = 3000;
}

pub struct AudioFeatureExtractor {
    window: Vec<f32>,
    /// `[FREQUENCY_BINS, MEL_BINS]`, row-major.
    mel_filters: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl Default for AudioFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFeatureExtractor {
    pub fn new() -> Self {
        Self {
            window: periodic_hann_window(FRAME_LENGTH),
            mel_filters: mel_filter_bank(),
            fft: RealFftPlanner::<f64>::new().plan_fft_forward(FFT_LENGTH),
        }
    }

    /// Features for mono 16 kHz samples in [-1, 1]; audio beyond [`MAX_SAMPLES`] is ignored.
    pub fn extract(&self, samples: &[f32]) -> AudioFeatures {
        let samples = &samples[..samples.len().min(MAX_SAMPLES)];
        let padded_length = samples.len().div_ceil(PAD_TO_MULTIPLE_OF) * PAD_TO_MULTIPLE_OF;
        // Semicausal padding: FRAME_LENGTH / 2 leading zeros centre the first frame on t = 0.
        let leading = FRAME_LENGTH / 2;
        let mut waveform = vec![0.0_f32; leading + padded_length];
        waveform[leading..leading + samples.len()].copy_from_slice(samples);
        let is_real_sample = |index: usize| index >= leading && index < leading + samples.len();

        let window_span = FRAME_LENGTH + 1;
        let frames = if waveform.len() < window_span { 0 } else { (waveform.len() - window_span) / HOP_LENGTH + 1 };
        let mut features = vec![0.0_f32; frames * MEL_BINS];
        let mut mask = vec![0_i64; frames];

        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut magnitudes = vec![0.0_f64; FREQUENCY_BINS];
        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            // A frame counts only if its whole analysis window is real audio (last sample checked).
            if !is_real_sample(start + window_span - 1) {
                continue;
            }
            mask[frame] = 1;
            input.fill(0.0);
            for (offset, (value, weight)) in waveform[start..start + FRAME_LENGTH].iter().zip(&self.window).enumerate()
            {
                input[offset] = f64::from(value * weight);
            }
            self.fft.process(&mut input, &mut spectrum).expect("buffers come from the planner");
            for (magnitude, bin) in magnitudes.iter_mut().zip(&spectrum) {
                *magnitude = bin.norm();
            }
            let row = &mut features[frame * MEL_BINS..(frame + 1) * MEL_BINS];
            for (mel, feature) in row.iter_mut().enumerate() {
                let energy: f64 = magnitudes
                    .iter()
                    .enumerate()
                    .map(|(bin, magnitude)| magnitude * self.mel_filters[bin * MEL_BINS + mel])
                    .sum();
                *feature = (energy + MEL_FLOOR).ln() as f32;
            }
        }
        AudioFeatures { features, mask, frames }
    }
}

/// `transformers.audio_utils.window_function(length, "hann", periodic=True)`, cast to f32.
fn periodic_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / length as f64).cos()) as f32)
        .collect()
}

/// `transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk")`: triangular filters
/// with peaks evenly spaced on the HTK mel scale, evaluated at the linear FFT bin frequencies.
fn mel_filter_bank() -> Vec<f64> {
    let hertz_to_mel = |hertz: f64| 2595.0 * (1.0 + hertz / 700.0).log10();
    let mel_to_hertz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let (mel_min, mel_max) = (hertz_to_mel(MIN_FREQUENCY), hertz_to_mel(MAX_FREQUENCY));
    let filter_frequencies: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| mel_to_hertz(mel_min + (mel_max - mel_min) * index as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let nyquist = (SAMPLE_RATE / 2) as f64;
    let mut filters = vec![0.0; FREQUENCY_BINS * MEL_BINS];
    for bin in 0..FREQUENCY_BINS {
        let frequency = nyquist * bin as f64 / (FREQUENCY_BINS - 1) as f64;
        for mel in 0..MEL_BINS {
            let (low, center, high) =
                (filter_frequencies[mel], filter_frequencies[mel + 1], filter_frequencies[mel + 2]);
            let rising = (frequency - low) / (center - low);
            let falling = (high - frequency) / (high - center);
            filters[bin * MEL_BINS + mel] = rising.min(falling).max(0.0);
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_and_mask_match_upstream_arithmetic() {
        // 8.635 s, as in the reference memo: 863 frames, all valid (verified against Python).
        let features = AudioFeatureExtractor::new().extract(&vec![0.1; 138_160]);
        assert_eq!(features.frames, 863);
        assert!(features.mask.iter().all(|&valid| valid == 1));
        assert_eq!(features.features.len(), 863 * MEL_BINS);
    }

    #[test]
    fn frames_reaching_into_padding_are_masked_and_zeroed() {
        // 900 samples pad to 1024: the last frame's window ends at sample 960, past the real audio.
        let features = AudioFeatureExtractor::new().extract(&vec![0.1; 900]);
        let last = features.frames - 1;
        assert_eq!(features.mask[last], 0);
        assert!(features.features[last * MEL_BINS..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn a_full_thirty_seconds_fits_the_graph_window() {
        let full_window = AudioFeatureExtractor::new().extract(&vec![0.0; MAX_SAMPLES]).frames;
        assert_eq!(full_window, 2999);
        assert!(full_window <= AudioFeatures::WINDOW_FRAMES);
    }

    #[test]
    fn audio_beyond_thirty_seconds_is_ignored() {
        let extractor = AudioFeatureExtractor::new();
        assert_eq!(
            extractor.extract(&vec![0.0; MAX_SAMPLES + 50_000]).frames,
            extractor.extract(&vec![0.0; MAX_SAMPLES]).frames
        );
    }

    #[test]
    fn silence_sits_at_the_mel_floor() {
        let features = AudioFeatureExtractor::new().extract(&vec![0.0; 16_000]);
        let floor = (MEL_FLOOR).ln() as f32;
        assert!(features.features.iter().take(MEL_BINS).all(|&value| (value - floor).abs() < 1e-6));
    }

    #[test]
    fn a_pure_tone_peaks_in_the_matching_mel_band() {
        let tone: Vec<f32> =
            (0..16_000).map(|index| (2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / 16_000.0).sin()).collect();
        let features = AudioFeatureExtractor::new().extract(&tone);
        let row = &features.features[50 * MEL_BINS..51 * MEL_BINS];
        let peak = (0..MEL_BINS).max_by(|&left, &right| row[left].total_cmp(&row[right])).unwrap();
        // 1 kHz is 1000 mel on the HTK scale; bands are ~22 mel apart from 0 to ~2840 mel.
        let mel_of_peak = peak as f64 * 2840.0 / 129.0;
        assert!((mel_of_peak - 1000.0).abs() < 60.0, "1 kHz tone peaked in band {peak}");
    }
}
