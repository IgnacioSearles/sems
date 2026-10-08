//! Formats search results for terminals, pipes, and machine consumers.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::search::{ResultContent, SearchResult};

/// Longer lines (minified code, long prose lines) are cut so one result cannot flood the screen.
const MAX_LINE_CHARACTERS: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// `path:start-end  similarity` followed by the matched lines, numbered.
    Text,
    /// One path per line, best first, without duplicates (like `grep -l`).
    FilesWithMatches,
    /// A JSON array of results, including full chunk text, for scripts and coding agents.
    Json,
}

/// Terminal decorations, enabled only when writing to a terminal so piped output stays clean.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub color: bool,
    /// Wrap result paths in OSC 8 hyperlinks, which supporting terminals open on (Ctrl+)click.
    pub hyperlinks: bool,
}

impl Style {
    pub const PLAIN: Self = Self { color: false, hyperlinks: false };

    fn paint(self, code: &str, text: &str) -> String {
        if self.color { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_string() }
    }

    /// `text` as a link to `path`. OSC 8: `ESC ] 8 ; ; URI ESC \ text ESC ] 8 ; ; ESC \`.
    fn link(self, path: &Path, text: &str) -> String {
        if self.hyperlinks { format!("\x1b]8;;{}\x1b\\{text}\x1b]8;;\x1b\\", file_uri(path)) } else { text.to_string() }
    }
}

/// A `file://` URI for an absolute path, percent-encoding everything but unreserved characters and
/// separators so spaces and non-ASCII names survive (`C:\My Docs\é.pdf` ->
/// `file:///C:/My%20Docs/%C3%A9.pdf`). Windows UNC paths (`\\server\share`) keep their host.
pub fn file_uri(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let (prefix, rest) = match text.strip_prefix("//") {
        Some(unc) => ("file://", unc.to_string()),
        None => ("file:///", text.trim_start_matches('/').to_string()),
    };
    let mut uri = String::from(prefix);
    for byte in rest.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~/:".contains(&byte) {
            uri.push(byte as char);
        } else {
            write!(uri, "%{byte:02X}").unwrap();
        }
    }
    uri
}

pub fn render(results: &[SearchResult], format: OutputFormat, style: Style, working_directory: &Path) -> String {
    match format {
        OutputFormat::Text => render_text(results, style, working_directory),
        OutputFormat::FilesWithMatches => render_files(results, style, working_directory),
        OutputFormat::Json => serde_json::to_string_pretty(results).expect("search results always serialize") + "\n",
    }
}

/// Each text result is its whole chunk (one function or section, at most a few dozen lines), so
/// the lines that matched are shown in full; image results show the file and its dimensions.
fn render_text(results: &[SearchResult], style: Style, working_directory: &Path) -> String {
    let mut output = String::new();
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        // Colors follow ripgrep: magenta paths, green line numbers.
        let path = style.link(&result.path, &style.paint("35", &display_path(&result.path, working_directory)));
        let similarity = style.paint("2", &format!("{:.2}", result.similarity));
        match &result.content {
            ResultContent::Text { start_line, end_line, text } => {
                let lines = style.paint("32", &format!("{start_line}-{end_line}"));
                writeln!(output, "{path}:{lines}  {similarity}").unwrap();
                write_numbered_lines(&mut output, style, *start_line, *end_line, text);
            }
            ResultContent::Image => {
                writeln!(output, "{path}  {similarity}  {}", style.paint("36", &image_label(&result.path))).unwrap();
            }
            ResultContent::Pdf { page, text, .. } => {
                writeln!(output, "{path}  {}  {similarity}", style.paint("32", &format!("page {page}"))).unwrap();
                // Line numbers within extracted PDF text mean nothing to a reader; the page does.
                for (_, line) in numbered_lines(1, text) {
                    if line.is_empty() {
                        output.push('\n');
                    } else {
                        writeln!(output, "    {line}").unwrap();
                    }
                }
            }
        }
    }
    output
}

fn write_numbered_lines(output: &mut String, style: Style, start_line: usize, end_line: usize, text: &str) {
    let width = end_line.to_string().len();
    for (line_number, line) in numbered_lines(start_line, text) {
        let gutter = style.paint("32", &format!("{line_number:>width$}"));
        if line.is_empty() {
            writeln!(output, "{gutter}:").unwrap();
        } else {
            writeln!(output, "{gutter}: {line}").unwrap();
        }
    }
}

/// `[image 4032x3024]`, read from the file header only; just `[image]` if the file has changed
/// or moved since it was indexed.
fn image_label(path: &Path) -> String {
    match image::image_dimensions(path) {
        Ok((width, height)) => format!("[image {width}x{height}]"),
        Err(_) => "[image]".to_string(),
    }
}

fn render_files(results: &[SearchResult], style: Style, working_directory: &Path) -> String {
    let mut seen: Vec<&PathBuf> = Vec::new();
    let mut output = String::new();
    for result in results {
        if !seen.contains(&&result.path) {
            seen.push(&result.path);
            writeln!(output, "{}", style.link(&result.path, &display_path(&result.path, working_directory))).unwrap();
        }
    }
    output
}

/// Paths under the working directory are shown relative, as grep does; others stay absolute.
pub fn display_path(path: &Path, working_directory: &Path) -> String {
    match path.strip_prefix(working_directory) {
        Ok(relative) if relative.as_os_str().is_empty() => ".".to_string(),
        Ok(relative) => relative.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

/// The chunk's lines with their line numbers, minus the indentation they all share so nested code
/// does not drift right. Overlong lines are cut on a character boundary.
fn numbered_lines(start_line: usize, text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().map(str::trim_end).collect();
    let shared_indent = lines
        .iter()
        .filter(|line| !line.is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .enumerate()
        .map(|(offset, line)| {
            // Indentation is ASCII whitespace, so byte slicing stays on a char boundary.
            let line = line.get(shared_indent..).unwrap_or("");
            let line = match line.char_indices().nth(MAX_LINE_CHARACTERS) {
                Some((cut, _)) => format!("{}…", &line[..cut]),
                None => line.to_string(),
            };
            (start_line + offset, line)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(path: &str, start_line: usize, text: &str) -> SearchResult {
        SearchResult {
            path: PathBuf::from(path),
            similarity: 0.5,
            content: ResultContent::Text {
                start_line,
                end_line: start_line + text.lines().count().max(1) - 1,
                text: text.into(),
            },
        }
    }

    #[test]
    fn image_results_show_dimensions_from_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let photo = directory.path().join("beach.png");
        image::RgbImage::new(120, 80).save(&photo).unwrap();
        let results = [
            SearchResult { path: photo, similarity: 0.73, content: ResultContent::Image },
            SearchResult { path: directory.path().join("gone.jpg"), similarity: 0.5, content: ResultContent::Image },
        ];
        let output = render(&results, OutputFormat::Text, Style::PLAIN, directory.path());
        assert_eq!(output, "beach.png  0.73  [image 120x80]\n\ngone.jpg  0.50  [image]\n");
    }

    #[test]
    fn pdf_results_show_the_page_and_text_without_line_numbers() {
        let results = [SearchResult {
            path: PathBuf::from("lease.pdf"),
            similarity: 0.81,
            content: ResultContent::Pdf {
                page: 3,
                start_line: 1,
                end_line: 2,
                text: "4. Termination\nTwo months' notice.".into(),
            },
        }];
        let output = render(&results, OutputFormat::Text, Style::PLAIN, &working_directory());
        assert_eq!(output, "lease.pdf  page 3  0.81\n    4. Termination\n    Two months' notice.\n");
        let json: serde_json::Value =
            serde_json::from_str(&render(&results, OutputFormat::Json, Style::PLAIN, &working_directory())).unwrap();
        assert_eq!((json[0]["kind"].as_str(), json[0]["page"].as_u64()), (Some("pdf"), Some(3)));
        assert!(json[0].get("start_line").is_none(), "lines within a PDF page are internal");
    }

    #[test]
    fn hyperlinks_wrap_the_path_and_leave_the_text_unchanged() {
        let path = working_directory().join("notes.txt");
        let results = [result(path.to_str().unwrap(), 1, "hello")];
        let style = Style { color: false, hyperlinks: true };
        let output = render(&results, OutputFormat::FilesWithMatches, style, &working_directory());
        let uri = file_uri(&path);
        assert_eq!(output, format!("\x1b]8;;{uri}\x1b\\notes.txt\x1b]8;;\x1b\\\n"));
    }

    #[test]
    fn file_uris_percent_encode_spaces_and_non_ascii() {
        assert_eq!(
            file_uri(Path::new(r"C:\Users\Ana María\My Docs\plan #2.pdf")),
            "file:///C:/Users/Ana%20Mar%C3%ADa/My%20Docs/plan%20%232.pdf"
        );
        assert_eq!(file_uri(Path::new("/home/ana/notes.md")), "file:///home/ana/notes.md");
        assert_eq!(file_uri(Path::new(r"\\server\share\a.txt")), "file://server/share/a.txt");
    }

    #[test]
    fn json_tags_each_result_with_its_kind() {
        let results = [
            result("a.txt", 1, "x"),
            SearchResult { path: PathBuf::from("b.jpg"), similarity: 0.5, content: ResultContent::Image },
        ];
        let json: serde_json::Value =
            serde_json::from_str(&render(&results, OutputFormat::Json, Style::PLAIN, &working_directory())).unwrap();
        assert_eq!(json[0]["kind"], "text");
        assert_eq!(json[0]["start_line"], 1);
        assert_eq!(json[1]["kind"], "image");
        assert!(json[1].get("start_line").is_none());
    }

    fn working_directory() -> PathBuf {
        PathBuf::from("project")
    }

    #[test]
    fn text_output_shows_every_matched_line_numbered() {
        let path = working_directory().join("src").join("main.rs");
        let results = [
            result(path.to_str().unwrap(), 9, "    fn main() {\n\n        run();\n    }"),
            result("other.txt", 1, "note"),
        ];
        let output = render(&results, OutputFormat::Text, Style::PLAIN, &working_directory());
        let expected_path = Path::new("src").join("main.rs");
        let expected = format!(
            "{}:9-12  0.50\n 9: fn main() {{\n10:\n11:     run();\n12: }}\n\nother.txt:1-1  0.50\n1: note\n",
            expected_path.display()
        );
        assert_eq!(output, expected);
    }

    #[test]
    fn files_with_matches_lists_each_file_once_in_rank_order() {
        let results = [result("b.txt", 1, "x"), result("a.txt", 1, "y"), result("b.txt", 9, "z")];
        let output = render(&results, OutputFormat::FilesWithMatches, Style::PLAIN, &working_directory());
        assert_eq!(output, "b.txt\na.txt\n");
    }

    #[test]
    fn long_lines_are_cut_on_character_boundaries() {
        let lines = numbered_lines(1, &"é".repeat(500));
        assert_eq!(lines[0].1.chars().count(), MAX_LINE_CHARACTERS + 1);
    }

    #[test]
    fn paths_outside_the_working_directory_stay_absolute() {
        assert_eq!(display_path(Path::new("elsewhere/file.txt"), &working_directory()), "elsewhere/file.txt");
    }

    #[test]
    fn the_working_directory_itself_displays_as_dot() {
        assert_eq!(display_path(&working_directory(), &working_directory()), ".");
    }
}
