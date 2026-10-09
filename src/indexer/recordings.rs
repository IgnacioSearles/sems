//! Audio and video: decoded by ffmpeg, cut into time segments, and embedded segment by segment.
//!
//! A video contributes two kinds of segments: its frames (what is seen) and its soundtrack (what
//! is said or heard). Every recording's frames are embedded before any audio, so the vision and
//! audio encoders each load once per run. A video's frame segments (one small vector each) wait in
//! memory until its soundtrack is done, and the file is then written in one transaction.
//!
//! A file ffmpeg cannot decode is skipped, keeping whatever segments decoded before the failure.
//! Failing to embed stops the run, as it does for images.

use std::path::PathBuf;

use anyhow::{Context, Result};
use image::DynamicImage;

use super::{IndexProgress, IndexSummary};
use crate::av::{DecodeError, Ffmpeg, MediaInfo, SAMPLE_RATE};
use crate::discovery::FileKind;
use crate::encoder::Encoder;
use crate::segments::{MediaSegmentation, TimeSpan, fit_within, is_silent};
use crate::store::{FileRecord, IndexStore};

/// An audio or video file whose content changed, waiting to be decoded.
pub(super) struct PendingRecording {
    pub(super) path: PathBuf,
    pub(super) record: FileRecord,
    pub(super) kind: FileKind,
}

/// A recording being embedded, and the segments it has produced so far.
struct Recording {
    file: PendingRecording,
    info: MediaInfo,
    segments: Vec<(TimeSpan, Vec<f32>)>,
    decoding_failed: bool,
}

/// Counts segments as they are embedded and reports progress; recordings are long, so progress
/// moves per segment rather than per file.
struct Tally<'a> {
    progress: &'a mut dyn IndexProgress,
    summary: &'a mut IndexSummary,
}

impl Tally<'_> {
    fn segment_embedded(&mut self) {
        self.summary.chunks_embedded += 1;
        self.progress.files_embedded(self.summary.files_embedded, self.summary.chunks_embedded);
    }
}

pub(super) fn embed_recordings(
    store: &mut IndexStore,
    encoder: &mut dyn Encoder,
    ffmpeg: &Ffmpeg,
    pending: Vec<PendingRecording>,
    segmentation: &MediaSegmentation,
    progress: &mut dyn IndexProgress,
    summary: &mut IndexSummary,
) -> Result<()> {
    let mut recordings = Vec::with_capacity(pending.len());
    for file in pending {
        match unless_unreadable(ffmpeg.probe(&file.path))? {
            Some(info) => recordings.push(Recording { file, info, segments: Vec::new(), decoding_failed: false }),
            None => {
                store.record_skipped_file(&file.path, &file.record)?;
                summary.recordings_unreadable += 1;
            }
        }
    }

    let mut tally = Tally { progress, summary };
    for recording in &mut recordings {
        let decoded = embed_frames(encoder, ffmpeg, recording, segmentation, &mut tally);
        recording.decoding_failed |= unless_unreadable(decoded)?.is_none();
    }
    for mut recording in recordings {
        let decoded = embed_soundtrack(encoder, ffmpeg, &mut recording, segmentation, &mut tally);
        recording.decoding_failed |= unless_unreadable(decoded)?.is_none();
        store_recording(store, recording, tally.summary)?;
        tally.progress.files_embedded(tally.summary.files_embedded, tally.summary.chunks_embedded);
    }
    Ok(())
}

/// `None` when ffmpeg could not decode the file (it is skipped); an error for anything that
/// should stop the run (ffmpeg would not start, or embedding failed).
fn unless_unreadable<T>(result: Result<T, DecodeError>) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(DecodeError::Unreadable { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Embeds frames sampled every `video_seconds_per_frame`, `video_frames_per_segment` at a time.
fn embed_frames(
    encoder: &mut dyn Encoder,
    ffmpeg: &Ffmpeg,
    recording: &mut Recording,
    segmentation: &MediaSegmentation,
    tally: &mut Tally<'_>,
) -> Result<(), DecodeError> {
    let Some((width, height)) = recording.info.video_size else { return Ok(()) };
    let (width, height) = fit_within(width, height, segmentation.video_max_frame_edge);
    let clock = FrameClock {
        seconds_per_frame: segmentation.video_seconds_per_frame,
        duration_seconds: recording.info.duration_seconds,
    };
    let path = recording.file.path.clone();
    let segments = &mut recording.segments;
    let mut frames: Vec<DynamicImage> = Vec::with_capacity(segmentation.video_frames_per_segment);
    let mut first_frame = 0;
    let mut embed_segment = |frames: &mut Vec<DynamicImage>, first_frame: usize| -> Result<()> {
        let span = clock.span(first_frame, frames.len());
        let embedding = encoder
            .encode_video(frames)
            .with_context(|| format!("failed to embed {} at {:.0} s", path.display(), span.start_seconds()))?;
        segments.push((span, embedding));
        frames.clear();
        tally.segment_embedded();
        Ok(())
    };

    ffmpeg.for_each_video_frame(&path, clock.seconds_per_frame, width, height, |index, frame| {
        if frames.is_empty() {
            first_frame = index;
        }
        frames.push(DynamicImage::ImageRgb8(frame));
        if frames.len() == segmentation.video_frames_per_segment {
            embed_segment(&mut frames, first_frame)?;
        }
        Ok(())
    })?;
    if !frames.is_empty() {
        embed_segment(&mut frames, first_frame).map_err(DecodeError::Consumer)?;
    }
    Ok(())
}

/// Maps sampled frame indices to the time they cover.
#[derive(Clone, Copy)]
struct FrameClock {
    seconds_per_frame: u32,
    /// 0 when the container does not say.
    duration_seconds: f64,
}

impl FrameClock {
    /// Frame `i` stands for the `seconds_per_frame` after it; the last segment ends with the video.
    fn span(&self, first_frame: usize, frame_count: usize) -> TimeSpan {
        let seconds_per_frame = f64::from(self.seconds_per_frame);
        let start = first_frame as f64 * seconds_per_frame;
        let mut end = (first_frame + frame_count) as f64 * seconds_per_frame;
        if self.duration_seconds > start {
            end = end.min(self.duration_seconds);
        }
        TimeSpan::from_seconds(start, end)
    }
}

/// Embeds overlapping audio windows, skipping silence and fragments too short to mean anything.
fn embed_soundtrack(
    encoder: &mut dyn Encoder,
    ffmpeg: &Ffmpeg,
    recording: &mut Recording,
    segmentation: &MediaSegmentation,
    tally: &mut Tally<'_>,
) -> Result<(), DecodeError> {
    if !recording.info.has_audio {
        return Ok(());
    }
    let window = segmentation.audio_window_seconds as usize * SAMPLE_RATE;
    let hop = segmentation.audio_hop_seconds as usize * SAMPLE_RATE;
    let path = recording.file.path.clone();
    ffmpeg.for_each_audio_window(&path, window, hop, |start_sample, samples| {
        let span = TimeSpan::from_samples(start_sample, samples.len(), SAMPLE_RATE);
        if span.duration_seconds() < MediaSegmentation::MIN_AUDIO_SECONDS || is_silent(samples) {
            return Ok(());
        }
        let embedding = encoder
            .encode_audio(samples)
            .with_context(|| format!("failed to embed {} at {:.0} s", path.display(), span.start_seconds()))?;
        recording.segments.push((span, embedding));
        tally.segment_embedded();
        Ok(())
    })
}

/// Writes whatever segments the recording produced; a recording without any is tracked as
/// skipped (unreadable, or silent and too short), so unchanged runs do not decode it again.
fn store_recording(store: &mut IndexStore, recording: Recording, summary: &mut IndexSummary) -> Result<()> {
    let Recording { file, segments, decoding_failed, .. } = recording;
    if segments.is_empty() {
        store.record_skipped_file(&file.path, &file.record)?;
        if decoding_failed {
            summary.recordings_unreadable += 1;
        } else {
            summary.recordings_without_content += 1;
        }
        return Ok(());
    }
    store.replace_recording_file(&file.path, &file.record, file.kind, &segments)?;
    summary.files_embedded += 1;
    summary.recordings_embedded += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_segments_cover_their_frames_and_end_with_the_video() {
        let clock = FrameClock { seconds_per_frame: 3, duration_seconds: 24.0 };
        assert_eq!(clock.span(0, 3), TimeSpan::from_seconds(0.0, 9.0));
        assert_eq!(clock.span(6, 3), TimeSpan::from_seconds(18.0, 24.0), "clipped to the duration");
        let unknown_duration = FrameClock { seconds_per_frame: 3, duration_seconds: 0.0 };
        assert_eq!(unknown_duration.span(6, 2), TimeSpan::from_seconds(18.0, 24.0));
    }
}
