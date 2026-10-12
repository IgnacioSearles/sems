//! Progress and sizes as shown on the terminal.

use std::io::IsTerminal;
use std::time::Instant;

use sems::download::DownloadProgress;
use sems::indexer::IndexProgress;

/// Download progress (model and ONNX Runtime) on stderr: a line redrawn in place on a terminal,
/// otherwise one line per file so logs stay readable.
pub struct TerminalDownloadProgress {
    redraw: bool,
}

impl TerminalDownloadProgress {
    pub fn new() -> Self {
        Self { redraw: std::io::stderr().is_terminal() }
    }
}

impl DownloadProgress for TerminalDownloadProgress {
    fn started(&self, file_name: &str, size: u64) {
        if self.redraw {
            eprint!("\rdownloading {file_name} ({})   ", readable_size(size));
        } else {
            eprintln!("downloading {file_name} ({})", readable_size(size));
        }
    }

    fn advanced(&self, file_name: &str, downloaded: u64, size: u64) {
        if self.redraw {
            let percent = downloaded * 100 / size.max(1);
            eprint!("\rdownloading {file_name} ({}) {percent}%   ", readable_size(size));
        }
    }

    fn finished(&self, _file_name: &str) {
        if self.redraw {
            eprintln!();
        }
    }
}

/// Indexing progress on stderr, a single line redrawn in place.
pub struct TerminalProgress {
    started: Instant,
    files_total: usize,
    enabled: bool,
}

impl TerminalProgress {
    pub fn new() -> Self {
        Self { started: Instant::now(), files_total: 0, enabled: std::io::stderr().is_terminal() }
    }

    pub fn finish(&self) {
        if self.enabled && self.files_total > 0 {
            eprintln!();
        }
    }
}

impl IndexProgress for TerminalProgress {
    fn embedding_started(&mut self, files_to_embed: usize) {
        self.started = Instant::now();
        self.files_total = files_to_embed;
    }

    fn files_embedded(&mut self, files_done: usize, chunks_done: usize) {
        if !self.enabled {
            return;
        }
        let rate = chunks_done as f64 / self.started.elapsed().as_secs_f64().max(f64::EPSILON);
        eprint!("\rembedding {files_done}/{} files ({chunks_done} chunks, {rate:.1} chunks/s)   ", self.files_total);
    }
}

/// `4 KB`, `274 MB`: whole units, decimal as download sizes usually are.
pub fn readable_size(bytes: u64) -> String {
    if bytes < 1_000_000 { format!("{} KB", bytes.div_ceil(1_000)) } else { format!("{} MB", bytes / 1_000_000) }
}

/// `1 file`, `3 files`.
pub fn files(count: usize) -> String {
    format!("{count} file{}", if count == 1 { "" } else { "s" })
}

/// `just now`, `5 minutes ago`, `3 days ago`.
pub fn readable_age(seconds: i64) -> String {
    let plural = |count: i64, unit: &str| format!("{count} {unit}{} ago", if count == 1 { "" } else { "s" });
    match seconds.max(0) {
        0..60 => "just now".to_string(),
        seconds @ 60..3_600 => plural(seconds / 60, "minute"),
        seconds @ 3_600..86_400 => plural(seconds / 3_600, "hour"),
        seconds => plural(seconds / 86_400, "day"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_read_naturally() {
        assert_eq!(readable_age(5), "just now");
        assert_eq!(readable_age(60), "1 minute ago");
        assert_eq!(readable_age(7_200), "2 hours ago");
        assert_eq!(readable_age(3 * 86_400 + 5), "3 days ago");
        assert_eq!(readable_age(-10), "just now", "a clock set back is not the future");
    }

    #[test]
    fn sizes_use_whole_decimal_units() {
        assert_eq!(readable_size(4_455), "5 KB");
        assert_eq!(readable_size(273_705_984), "273 MB");
    }
}
