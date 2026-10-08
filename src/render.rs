//! Formats search results for terminals, pipes, and machine consumers.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::search::SearchResult;

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

/// ANSI styling, enabled only when writing to a terminal so piped output stays clean.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub color: bool,
}

impl Style {
    fn paint(self, code: &str, text: &str) -> String {
        if self.color { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_string() }
    }
}

pub fn render(results: &[SearchResult], format: OutputFormat, style: Style, working_directory: &Path) -> String {
    match format {
        OutputFormat::Text => render_text(results, style, working_directory),
        OutputFormat::FilesWithMatches => render_files(results, working_directory),
        OutputFormat::Json => serde_json::to_string_pretty(results).expect("search results always serialize") + "\n",
    }
}

/// Each result is its whole chunk (one function or section, at most a few dozen lines), so the
/// lines that matched are shown in full rather than a preview of the chunk's start.
fn render_text(results: &[SearchResult], style: Style, working_directory: &Path) -> String {
    let mut output = String::new();
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        // Colors follow ripgrep: magenta paths, green line numbers.
        let path = style.paint("35", &display_path(&result.path, working_directory));
        let lines = style.paint("32", &format!("{}-{}", result.start_line, result.end_line));
        writeln!(output, "{path}:{lines}  {}", style.paint("2", &format!("{:.2}", result.similarity))).unwrap();
        let width = result.end_line.to_string().len();
        for (line_number, line) in numbered_lines(result) {
            let gutter = style.paint("32", &format!("{line_number:>width$}"));
            if line.is_empty() {
                writeln!(output, "{gutter}:").unwrap();
            } else {
                writeln!(output, "{gutter}: {line}").unwrap();
            }
        }
    }
    output
}

fn render_files(results: &[SearchResult], working_directory: &Path) -> String {
    let mut seen: Vec<&PathBuf> = Vec::new();
    let mut output = String::new();
    for result in results {
        if !seen.contains(&&result.path) {
            seen.push(&result.path);
            writeln!(output, "{}", display_path(&result.path, working_directory)).unwrap();
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
fn numbered_lines(result: &SearchResult) -> Vec<(usize, String)> {
    let lines: Vec<&str> = result.text.lines().map(str::trim_end).collect();
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
            (result.start_line + offset, line)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(path: &str, start_line: usize, text: &str) -> SearchResult {
        SearchResult {
            path: PathBuf::from(path),
            start_line,
            end_line: start_line + text.lines().count().max(1) - 1,
            similarity: 0.5,
            text: text.into(),
        }
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
        let output = render(&results, OutputFormat::Text, Style { color: false }, &working_directory());
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
        let output = render(&results, OutputFormat::FilesWithMatches, Style { color: false }, &working_directory());
        assert_eq!(output, "b.txt\na.txt\n");
    }

    #[test]
    fn long_lines_are_cut_on_character_boundaries() {
        let lines = numbered_lines(&result("a.txt", 1, &"é".repeat(500)));
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
