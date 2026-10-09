//! Audio and video decoding through an external ffmpeg.
//!
//! Decoding every container and codec in Rust would mean bundling FFmpeg anyway, so sems runs the
//! user's ffmpeg and streams raw samples and frames from its stdout. The argument lists must stay
//! identical to tools/export/media_decoding.py, so references and the runtime see the same data.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

use anyhow::Context;
use image::RgbImage;

pub const SAMPLE_RATE: usize = 16_000;

/// Recording formats ffmpeg decodes. `.ogg` can hold video but almost always holds audio; `.ts` is
/// left out because it is far more often TypeScript than an MPEG transport stream.
const AUDIO_EXTENSIONS: [&str; 8] = ["mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma"];
const VIDEO_EXTENSIONS: [&str; 7] = ["mp4", "mov", "mkv", "webm", "avi", "m4v", "wmv"];

pub fn is_audio_path(path: &Path) -> bool {
    has_extension_in(path, &AUDIO_EXTENSIONS)
}

pub fn is_video_path(path: &Path) -> bool {
    has_extension_in(path, &VIDEO_EXTENSIONS)
}

fn has_extension_in(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extensions.contains(&extension.to_ascii_lowercase().as_str()))
}

/// Located ffmpeg and ffprobe executables.
#[derive(Debug, Clone)]
pub struct Ffmpeg {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
}

/// What `ffprobe` reports about a media file.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaInfo {
    pub duration_seconds: f64,
    pub has_audio: bool,
    /// Width and height of the first video stream as displayed (after rotation metadata, which
    /// ffmpeg applies while decoding), if there is one. Cover art in audio files does not count.
    pub video_size: Option<(u32, u32)>,
}

/// Why reading a recording stopped. A file ffmpeg cannot decode is the file's problem (callers
/// skip it); failing to start ffmpeg, or the caller's own processing failing, is not.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("failed to run {}: {source}", program.display())]
    Launch { program: PathBuf, source: std::io::Error },
    #[error("cannot decode {}: {message}", path.display())]
    Unreadable { path: PathBuf, message: String },
    /// Returned by the caller's callback; decoding was stopped.
    #[error(transparent)]
    Consumer(anyhow::Error),
}

impl Ffmpeg {
    /// `SEMS_FFMPEG` (the ffmpeg executable; ffprobe must sit beside it), otherwise `PATH`.
    pub fn locate() -> Option<Self> {
        let ffmpeg = match std::env::var_os("SEMS_FFMPEG") {
            Some(path) => PathBuf::from(path),
            None => find_on_path(&executable_name("ffmpeg"))?,
        };
        let ffprobe = ffmpeg.with_file_name(executable_name("ffprobe"));
        (ffmpeg.is_file() && ffprobe.is_file()).then_some(Self { ffmpeg, ffprobe })
    }

    /// The ffmpeg executable, for callers that need it directly (tests generate media with it).
    pub fn executable(&self) -> &Path {
        &self.ffmpeg
    }

    pub fn probe(&self, path: &Path) -> Result<MediaInfo, DecodeError> {
        let output = Command::new(&self.ffprobe)
            // Whole stream sections rather than selected entries: section names for side data
            // differ between ffprobe versions, while -show_streams output is stable.
            .args(["-v", "error", "-show_entries", "format=duration", "-show_streams", "-of", "json"])
            .arg(path)
            .stdin(Stdio::null())
            .output()
            .map_err(|source| DecodeError::Launch { program: self.ffprobe.clone(), source })?;
        if !output.status.success() {
            return Err(unreadable(path, &String::from_utf8_lossy(&output.stderr), output.status));
        }
        parse_probe_output(&output.stdout)
            .map_err(|error| DecodeError::Unreadable { path: path.to_path_buf(), message: format!("{error:#}") })
    }

    /// Streams the file's audio as 16 kHz mono and calls `on_window(start_sample, samples)` for
    /// windows of `window` samples starting every `hop` samples. A shorter final window is emitted
    /// only if it covers audio no earlier window did, so the end of every recording is included.
    pub fn for_each_audio_window(
        &self,
        path: &Path,
        window: usize,
        hop: usize,
        mut on_window: impl FnMut(usize, &[f32]) -> anyhow::Result<()>,
    ) -> Result<(), DecodeError> {
        assert!(hop > 0 && hop <= window, "hop must be in 1..=window");
        let sample_rate = SAMPLE_RATE.to_string();
        let mut process = self.decode(path, &["-vn", "-ac", "1", "-ar", &sample_rate, "-f", "f32le", "-"])?;
        let mut stdout = process.take_stdout();

        let mut buffer: Vec<f32> = Vec::with_capacity(window);
        let mut start = 0;
        let mut emitted_any = false;
        let mut bytes = vec![0_u8; 64 * 1024];
        let mut carry: Vec<u8> = Vec::new();
        loop {
            let read = stdout.read(&mut bytes).map_err(|error| process.read_failed(&error))?;
            if read == 0 {
                break;
            }
            carry.extend_from_slice(&bytes[..read]);
            let whole = carry.len() / 4 * 4;
            buffer.extend(carry[..whole].as_chunks::<4>().0.iter().map(|&sample| f32::from_le_bytes(sample)));
            carry.drain(..whole);
            while buffer.len() >= window {
                on_window(start, &buffer[..window]).map_err(DecodeError::Consumer)?;
                emitted_any = true;
                buffer.drain(..hop);
                start += hop;
            }
        }
        process.finish()?;
        let uncovered_tail = if emitted_any { buffer.len() > window - hop } else { !buffer.is_empty() };
        if uncovered_tail {
            on_window(start, &buffer).map_err(DecodeError::Consumer)?;
        }
        Ok(())
    }

    /// Streams RGB frames sampled every `seconds_per_frame` seconds (the first at 0 s), scaled to
    /// `width` x `height`, calling `on_frame(frame_index, frame)`.
    pub fn for_each_video_frame(
        &self,
        path: &Path,
        seconds_per_frame: u32,
        width: u32,
        height: u32,
        mut on_frame: impl FnMut(usize, RgbImage) -> anyhow::Result<()>,
    ) -> Result<(), DecodeError> {
        let filter = format!("fps=1/{seconds_per_frame},scale={width}:{height}");
        let mut process = self.decode(path, &["-vf", &filter, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])?;
        let mut stdout = process.take_stdout();
        let frame_bytes = width as usize * height as usize * 3;
        let mut index = 0;
        loop {
            let mut frame = vec![0_u8; frame_bytes];
            match stdout.read_exact(&mut frame) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(process.read_failed(&error)),
            }
            let image = RgbImage::from_raw(width, height, frame).expect("buffer has width * height * 3 bytes");
            on_frame(index, image).map_err(DecodeError::Consumer)?;
            index += 1;
        }
        process.finish()
    }

    fn decode<'a>(&self, path: &'a Path, output_arguments: &[&str]) -> Result<DecoderProcess<'a>, DecodeError> {
        let mut child = Command::new(&self.ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(output_arguments)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|source| DecodeError::Launch { program: self.ffmpeg.clone(), source })?;
        // Read stderr on a thread: a corrupt file can make ffmpeg log per frame, and an unread full
        // pipe would block it while we wait on stdout.
        let mut stderr = child.stderr.take().expect("stderr is piped");
        let stderr = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        Ok(DecoderProcess { child, stderr: Some(stderr), path })
    }
}

/// A running ffmpeg. Dropping it before [`Self::finish`] (the consumer stopped early) kills the
/// process, so no decoder outlives the caller.
struct DecoderProcess<'a> {
    child: Child,
    stderr: Option<JoinHandle<String>>,
    path: &'a Path,
}

impl DecoderProcess<'_> {
    fn take_stdout(&mut self) -> ChildStdout {
        self.child.stdout.take().expect("stdout is piped and taken once")
    }

    fn read_failed(&self, error: &std::io::Error) -> DecodeError {
        DecodeError::Unreadable {
            path: self.path.to_path_buf(),
            message: format!("failed to read ffmpeg output: {error}"),
        }
    }

    fn finish(mut self) -> Result<(), DecodeError> {
        let status = self.child.wait().map_err(|error| self.read_failed(&error))?;
        let messages = self.stderr.take().and_then(|handle| handle.join().ok()).unwrap_or_default();
        if !status.success() {
            return Err(unreadable(self.path, &messages, status));
        }
        Ok(())
    }
}

impl Drop for DecoderProcess<'_> {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn unreadable(path: &Path, messages: &str, status: std::process::ExitStatus) -> DecodeError {
    let message = match messages.trim() {
        "" => format!("ffmpeg exited with {status}"),
        messages => messages.to_string(),
    };
    DecodeError::Unreadable { path: path.to_path_buf(), message }
}

fn parse_probe_output(json: &[u8]) -> anyhow::Result<MediaInfo> {
    let value: serde_json::Value = serde_json::from_slice(json).context("unexpected ffprobe output")?;
    let duration_seconds = value["format"]["duration"].as_str().and_then(|text| text.parse().ok()).unwrap_or(0.0);
    let streams = value["streams"].as_array().cloned().unwrap_or_default();
    let has_audio = streams.iter().any(|stream| stream["codec_type"] == "audio");
    let video_size = streams
        .iter()
        .find(|stream| stream["codec_type"] == "video" && stream["disposition"]["attached_pic"] != 1)
        .and_then(displayed_size);
    Ok(MediaInfo { duration_seconds, has_audio, video_size })
}

/// Phones record portrait video as landscape frames plus a rotation, either as side data
/// (current ffmpeg) or as a `rotate` tag (older muxers).
fn displayed_size(stream: &serde_json::Value) -> Option<(u32, u32)> {
    let width = u32::try_from(stream["width"].as_u64()?).ok()?;
    let height = u32::try_from(stream["height"].as_u64()?).ok()?;
    let side_data_rotation = stream["side_data_list"]
        .as_array()
        .and_then(|side_data| side_data.iter().find_map(|entry| entry["rotation"].as_i64()));
    let tag_rotation = stream["tags"]["rotate"].as_str().and_then(|text| text.parse::<i64>().ok());
    let rotation = side_data_rotation.or(tag_rotation).unwrap_or(0);
    let quarter_turn = rotation.rem_euclid(180) == 90;
    Some(if quarter_turn { (height, width) } else { (width, height) })
}

fn executable_name(stem: &str) -> String {
    format!("{stem}{}", std::env::consts::EXE_SUFFIX)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|directory| directory.join(name)).find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval_corpus").join(relative)
    }

    #[test]
    fn parses_probe_output() {
        let json = br#"{"streams":[{"codec_type":"video","width":480,"height":320},{"codec_type":"audio"}],
                        "format":{"duration":"24.000000"}}"#;
        let info = parse_probe_output(json).unwrap();
        assert_eq!(info, MediaInfo { duration_seconds: 24.0, has_audio: true, video_size: Some((480, 320)) });
    }

    #[test]
    fn audio_only_files_have_no_video_size() {
        let info = parse_probe_output(br#"{"streams":[{"codec_type":"audio"}],"format":{"duration":"7.38"}}"#).unwrap();
        assert_eq!((info.has_audio, info.video_size), (true, None));
    }

    #[test]
    fn cover_art_is_not_video() {
        let json = br#"{"streams":[{"codec_type":"audio"},
                        {"codec_type":"video","width":600,"height":600,"disposition":{"attached_pic":1}}],
                        "format":{"duration":"180.0"}}"#;
        assert_eq!(parse_probe_output(json).unwrap().video_size, None);
    }

    #[test]
    fn rotated_phone_video_reports_its_displayed_size() {
        let side_data = br#"{"streams":[{"codec_type":"video","width":1920,"height":1080,
                             "side_data_list":[{"rotation":-90}]}],"format":{"duration":"5"}}"#;
        let tag = br#"{"streams":[{"codec_type":"video","width":1920,"height":1080,
                       "tags":{"rotate":"270"}}],"format":{"duration":"5"}}"#;
        let upside_down = br#"{"streams":[{"codec_type":"video","width":1920,"height":1080,
                               "side_data_list":[{"rotation":180}]}],"format":{"duration":"5"}}"#;
        assert_eq!(parse_probe_output(side_data).unwrap().video_size, Some((1080, 1920)));
        assert_eq!(parse_probe_output(tag).unwrap().video_size, Some((1080, 1920)));
        assert_eq!(parse_probe_output(upside_down).unwrap().video_size, Some((1920, 1080)));
    }

    #[test]
    fn recognizes_recordings_by_extension() {
        assert!(is_audio_path(Path::new("Voice Memo.M4A")));
        assert!(is_video_path(Path::new("IMG_0042.MOV")));
        assert!(!is_video_path(Path::new("index.ts")));
        assert!(!is_audio_path(Path::new("notes.md")));
    }

    /// The tests below need ffmpeg; they are skipped (not failed) when it is not installed.
    fn ffmpeg_or_skip() -> Option<Ffmpeg> {
        let ffmpeg = Ffmpeg::locate();
        if ffmpeg.is_none() {
            eprintln!("ffmpeg not found; skipping");
        }
        ffmpeg
    }

    #[test]
    fn audio_windows_cover_the_whole_recording() {
        let Some(ffmpeg) = ffmpeg_or_skip() else { return };
        let path = fixture("audio/memo_0003.m4a"); // 32.9 s
        let mut windows = Vec::new();
        ffmpeg
            .for_each_audio_window(&path, 10 * SAMPLE_RATE, 8 * SAMPLE_RATE, |start, samples| {
                windows.push((start, samples.len()));
                Ok(())
            })
            .unwrap();
        let (last_start, last_length) = *windows.last().unwrap();
        let total = ffmpeg.probe(&path).unwrap().duration_seconds * SAMPLE_RATE as f64;
        assert!(windows.iter().take(windows.len() - 1).all(|&(_, length)| length == 10 * SAMPLE_RATE));
        assert!(((last_start + last_length) as f64 - total).abs() < SAMPLE_RATE as f64 * 0.1);
    }

    #[test]
    fn video_frames_are_sampled_at_the_requested_interval() {
        let Some(ffmpeg) = ffmpeg_or_skip() else { return };
        let mut count = 0;
        ffmpeg
            .for_each_video_frame(&fixture("videos/clip_0001.mp4"), 8, 48, 32, |_, frame| {
                assert_eq!(frame.dimensions(), (48, 32));
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 3, "24 s at one frame every 8 s");
    }

    #[test]
    fn undecodable_files_are_errors() {
        let Some(ffmpeg) = ffmpeg_or_skip() else { return };
        let directory = tempfile::tempdir().unwrap();
        let broken = directory.path().join("broken.mp4");
        std::fs::write(&broken, b"not a video").unwrap();
        assert!(matches!(ffmpeg.probe(&broken), Err(DecodeError::Unreadable { .. })));
        let decoded = ffmpeg.for_each_audio_window(&broken, 100, 100, |_, _| Ok(()));
        assert!(matches!(decoded, Err(DecodeError::Unreadable { .. })));
    }

    #[test]
    fn a_failing_consumer_stops_decoding_with_its_own_error() {
        let Some(ffmpeg) = ffmpeg_or_skip() else { return };
        let mut windows = 0;
        let decoded = ffmpeg.for_each_audio_window(&fixture("audio/memo_0003.m4a"), 1_000, 1_000, |_, _| {
            windows += 1;
            anyhow::bail!("embedding failed")
        });
        assert!(matches!(decoded, Err(DecodeError::Consumer(_))));
        assert_eq!(windows, 1, "no window is delivered after the consumer fails");
    }
}
