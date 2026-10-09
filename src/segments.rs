//! How recordings are cut into time segments, each embedded on its own, so a search can point at
//! the moment that matched instead of a whole hour-long file.

/// A stretch of a recording, in milliseconds from its start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSpan {
    pub start_milliseconds: u64,
    pub end_milliseconds: u64,
}

impl TimeSpan {
    pub fn from_seconds(start_seconds: f64, end_seconds: f64) -> Self {
        Self { start_milliseconds: milliseconds(start_seconds), end_milliseconds: milliseconds(end_seconds) }
    }

    /// The span covered by `sample_count` samples starting at `start_sample`.
    pub fn from_samples(start_sample: usize, sample_count: usize, sample_rate: usize) -> Self {
        let seconds = |sample: usize| sample as f64 / sample_rate as f64;
        Self::from_seconds(seconds(start_sample), seconds(start_sample + sample_count))
    }

    pub fn start_seconds(&self) -> f64 {
        self.start_milliseconds as f64 / 1000.0
    }

    pub fn end_seconds(&self) -> f64 {
        self.end_milliseconds as f64 / 1000.0
    }

    pub fn duration_seconds(&self) -> f64 {
        self.end_seconds() - self.start_seconds()
    }

    pub fn contains_second(&self, second: f64) -> bool {
        (self.start_seconds()..=self.end_seconds()).contains(&second)
    }

    /// Spans sharing more than an instant; adjacent segments (`0-9`, `9-18`) do not overlap.
    pub fn overlaps(&self, other: &TimeSpan) -> bool {
        self.start_milliseconds < other.end_milliseconds && other.start_milliseconds < self.end_milliseconds
    }
}

fn milliseconds(seconds: f64) -> u64 {
    (seconds.max(0.0) * 1000.0).round() as u64
}

/// Segment sizes for audio and video. Part of the index identity (through [`Self::VERSION`]):
/// changing them changes what a chunk is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSegmentation {
    /// Audio is embedded in windows this long. 30 s is the model's limit, and shorter windows cost
    /// almost as much: the exported audio graph always processes 30 s (measured on CPU: 2.1 s for
    /// a 10 s window, 2.8 s for 30 s).
    pub audio_window_seconds: u32,
    /// Start of one audio window to the start of the next; less than the window, so speech cut at
    /// one window's edge is whole in the next.
    pub audio_hop_seconds: u32,
    /// One video frame is sampled every this many seconds.
    pub video_seconds_per_frame: u32,
    /// Frames embedded together as one video segment.
    pub video_frames_per_segment: usize,
    /// Frames are decoded no larger than this (longest edge); the model shrinks them to its token
    /// budget anyway (~570 px square at 140 tokens), so larger frames only cost memory.
    pub video_max_frame_edge: u32,
}

impl MediaSegmentation {
    /// Bumped whenever the defaults or the segmenting logic change, so stale indexes are rebuilt.
    pub const VERSION: u32 = 1;
    /// Shorter audio (a trailing fragment, a click) is not worth a segment.
    pub const MIN_AUDIO_SECONDS: f64 = 1.0;

    pub fn video_segment_seconds(&self) -> u32 {
        self.video_seconds_per_frame * self.video_frames_per_segment as u32
    }
}

impl Default for MediaSegmentation {
    fn default() -> Self {
        Self {
            audio_window_seconds: 30,
            audio_hop_seconds: 25,
            video_seconds_per_frame: 3,
            video_frames_per_segment: 3,
            video_max_frame_edge: 768,
        }
    }
}

/// Scales `width` x `height` so the longer edge is at most `max_edge`, keeping the aspect ratio.
/// Never upscales, and keeps both edges at least 1 px.
pub fn fit_within(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max_edge {
        return (width, height);
    }
    let scale =
        |edge: u32| ((u64::from(edge) * u64::from(max_edge) + u64::from(longest) / 2) / u64::from(longest)).max(1);
    (scale(width) as u32, scale(height) as u32)
}

/// Root-mean-square level below which audio is treated as silence (-60 dBFS): digital silence
/// and the hiss of an idle microphone, which would only add noise to search results.
const SILENCE_RMS: f32 = 0.001;

pub fn is_silent(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let mean_square = samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    mean_square.sqrt() < SILENCE_RMS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_convert_samples_to_milliseconds() {
        let span = TimeSpan::from_samples(16_000, 8_000, 16_000);
        assert_eq!(span, TimeSpan { start_milliseconds: 1_000, end_milliseconds: 1_500 });
        assert!(span.contains_second(1.2));
        assert!(!span.contains_second(1.6));
    }

    #[test]
    fn adjacent_spans_do_not_overlap() {
        let first = TimeSpan::from_seconds(0.0, 9.0);
        assert!(!first.overlaps(&TimeSpan::from_seconds(9.0, 18.0)));
        assert!(first.overlaps(&TimeSpan::from_seconds(8.0, 18.0)));
    }

    #[test]
    fn frames_shrink_to_the_longest_edge_keeping_aspect() {
        assert_eq!(fit_within(1920, 1080, 768), (768, 432));
        assert_eq!(fit_within(1080, 1920, 768), (432, 768));
        assert_eq!(fit_within(480, 320, 768), (480, 320), "never upscales");
        assert_eq!(fit_within(10_000, 1, 768), (768, 1), "keeps at least one pixel");
    }

    #[test]
    fn silence_is_detected_by_level() {
        assert!(is_silent(&[0.0; 1_000]));
        assert!(is_silent(&[]));
        assert!(!is_silent(&[0.1, -0.1, 0.1, -0.1]));
    }
}
