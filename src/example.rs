//! The vector `sems similar` searches with: what a given file is about.
//!
//! Text is embedded as a search query: the model is trained for query-to-document retrieval, and
//! comparing documents with documents matched their style (short notes with short notes) more than
//! their subject. Photos and recordings have no such asymmetry, so one already in the index costs
//! no model work (its embeddings are averaged). Anything else is embedded from its beginning (the
//! first 30 s of sound, the first video segment), which keeps a CPU search to seconds.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::DynamicImage;

use crate::av::{DecodeError, Ffmpeg, SAMPLE_RATE};
use crate::discovery::{FileKind, decode_text};
use crate::documents::extract_pdf_text;
use crate::encoder::Encoder;
use crate::media::{LoadedImage, load_image};
use crate::segments::{MediaSegmentation, fit_within, is_silent};
use crate::store::IndexStore;

/// The beginning of a text example, about 1,000 tokens: enough to say what it is about, and
/// about a second to embed on a CPU.
const EXAMPLE_TEXT_CHARACTERS: usize = 4_000;

/// Unit-length vector for `path`.
pub fn example_embedding(
    store: &IndexStore,
    encoder: &mut dyn Encoder,
    path: &Path,
    ffmpeg: Option<&Ffmpeg>,
    segmentation: &MediaSegmentation,
) -> Result<Vec<f32>> {
    let kind = FileKind::of(path);
    match kind {
        FileKind::Text => return embed_text(encoder, path, &read_text(path)?),
        FileKind::Pdf => return embed_text(encoder, path, &read_pdf_text(path)?),
        FileKind::Image | FileKind::Audio | FileKind::Video => {}
    }
    let stored = store.embeddings_of_file(path)?;
    if !stored.is_empty() {
        return Ok(mean_direction(&stored));
    }
    if kind == FileKind::Image {
        return embed_image(encoder, path);
    }
    let ffmpeg = ffmpeg.context("finding files like a recording needs ffmpeg on PATH or SEMS_FFMPEG")?;
    embed_recording(encoder, ffmpeg, path, kind, segmentation)
}

/// The normalized mean: the direction the vectors share.
fn mean_direction(vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut sum = vec![0.0_f32; vectors[0].len()];
    for vector in vectors {
        for (total, value) in sum.iter_mut().zip(vector) {
            *total += value;
        }
    }
    let norm = sum.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 { sum } else { sum.iter().map(|value| value / norm).collect() }
}

fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    decode_text(&bytes).with_context(|| format!("{} is not a text file", path.display()))
}

fn read_pdf_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let text = extract_pdf_text(&bytes).with_context(|| format!("cannot read {}", path.display()))?;
    if !text.has_text() {
        bail!("{} has no text layer (scanned PDFs need OCR)", path.display());
    }
    Ok(text.pages.join("\n"))
}

fn embed_text(encoder: &mut dyn Encoder, path: &Path, text: &str) -> Result<Vec<f32>> {
    let text: String = text.chars().take(EXAMPLE_TEXT_CHARACTERS).collect();
    if text.trim().is_empty() {
        bail!("{} is empty", path.display());
    }
    encoder.encode_query(&text)
}

fn embed_image(encoder: &mut dyn Encoder, path: &Path) -> Result<Vec<f32>> {
    match load_image(path)? {
        LoadedImage::Image(image) => encoder.encode_image(&image),
        LoadedImage::TooSmall => bail!("{} is too small to compare", path.display()),
    }
}

/// The first video segment's frames, or, for audio and videos without picture, the first
/// 30 seconds that are not silent.
fn embed_recording(
    encoder: &mut dyn Encoder,
    ffmpeg: &Ffmpeg,
    path: &Path,
    kind: FileKind,
    segmentation: &MediaSegmentation,
) -> Result<Vec<f32>> {
    let info = ffmpeg.probe(path)?;
    if let (FileKind::Video, Some((width, height))) = (kind, info.video_size) {
        let frames = first_video_frames(ffmpeg, path, width, height, segmentation)?;
        return encoder.encode_video(&frames);
    }
    let samples = first_sound(ffmpeg, path, segmentation)?;
    encoder.encode_audio(&samples)
}

fn first_video_frames(
    ffmpeg: &Ffmpeg,
    path: &Path,
    width: u32,
    height: u32,
    segmentation: &MediaSegmentation,
) -> Result<Vec<DynamicImage>> {
    let (width, height) = fit_within(width, height, segmentation.video_max_frame_edge);
    let wanted = segmentation.video_frames_per_segment;
    let mut frames = Vec::with_capacity(wanted);
    let decoded = ffmpeg.for_each_video_frame(path, segmentation.video_seconds_per_frame, width, height, |_, frame| {
        frames.push(DynamicImage::ImageRgb8(frame));
        // Stops decoding the rest of the video; recognized below.
        if frames.len() == wanted { bail!(EnoughDecoded) } else { Ok(()) }
    });
    finish_early_stop(decoded)?;
    if frames.is_empty() {
        bail!("{} has no frames", path.display());
    }
    Ok(frames)
}

fn first_sound(ffmpeg: &Ffmpeg, path: &Path, segmentation: &MediaSegmentation) -> Result<Vec<f32>> {
    let window = segmentation.audio_window_seconds as usize * SAMPLE_RATE;
    let mut sound = None;
    let decoded = ffmpeg.for_each_audio_window(path, window, window, |_, samples| {
        if is_silent(samples) {
            return Ok(());
        }
        sound = Some(samples.to_vec());
        bail!(EnoughDecoded)
    });
    finish_early_stop(decoded)?;
    sound.with_context(|| format!("{} is silent", path.display()))
}

/// Returned by a decoding callback once it has what it needs, to stop ffmpeg early.
#[derive(Debug, thiserror::Error)]
#[error("enough decoded")]
struct EnoughDecoded;

fn finish_early_stop(decoded: Result<(), DecodeError>) -> Result<()> {
    match decoded {
        Err(DecodeError::Consumer(error)) if error.is::<EnoughDecoded>() => Ok(()),
        other => other.map_err(Into::into),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mean_direction_is_unit_length_and_between_its_inputs() {
        let mean = mean_direction(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((mean[0] - expected).abs() < 1e-6 && (mean[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn an_early_stop_is_not_an_error_but_a_decoding_failure_is() {
        assert!(finish_early_stop(Err(DecodeError::Consumer(EnoughDecoded.into()))).is_ok());
        let unreadable = DecodeError::Unreadable { path: "x.mp4".into(), message: "bad".into() };
        assert!(finish_early_stop(Err(unreadable)).is_err());
        assert!(finish_early_stop(Err(DecodeError::Consumer(anyhow::anyhow!("embedding failed")))).is_err());
    }
}
