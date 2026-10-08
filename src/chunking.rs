//! Splits file text into chunks that follow the text's own structure.
//!
//! A chunk should be one coherent unit (a function, a markdown section, a paragraph) so that its
//! embedding is not an average of unrelated topics and results can point at the lines that match.
//! Chunks therefore end at natural boundaries: a blank line followed by a less indented line, with
//! headings and top-level code preferred. Only units longer than `max_lines` are cut mid-way, and
//! those cuts overlap so text straddling the cut still appears whole in one chunk.
//!
//! Measured on tests/fixtures (search_quality): fixed 60-line windows put the right lines first for
//! 47% of queries (mean span 43 lines); structural chunks of 4-30 lines for 100% (mean span 7).

/// A slice of a file. Line numbers are 1-based and inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkingConfig {
    /// Boundaries closer than this to a chunk's start are ignored, so tiny units (an import block,
    /// a two-line section) join their neighbour instead of becoming context-free fragments.
    pub min_lines: usize,
    pub max_lines: usize,
    /// Overlap applied only when a unit longer than `max_lines` has to be cut without a boundary.
    pub overlap_lines: usize,
    /// Upper bound on chunk size, roughly 4 characters per token. Protects against minified or
    /// generated files whose few lines are enormous.
    pub max_characters: usize,
}

impl ChunkingConfig {
    /// Bumped whenever the chunking output changes, so stale indexes are rebuilt.
    pub const VERSION: u32 = 2;
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self { min_lines: 4, max_lines: 30, overlap_lines: 5, max_characters: 3_000 }
    }
}

pub fn chunk_text(text: &str, config: &ChunkingConfig) -> Vec<Chunk> {
    assert!(config.overlap_lines < config.max_lines, "overlap must be smaller than the window");
    assert!(config.min_lines >= 1 && config.min_lines <= config.max_lines, "min_lines must be in 1..=max_lines");
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    loop {
        // Chunks never begin with blank lines; a file's trailing blank lines produce nothing.
        while start < lines.len() && is_blank(lines[start]) {
            start += 1;
        }
        if start == lines.len() {
            break;
        }
        let window_end = window_end(&lines, start, config);
        if window_end == start + 1 && lines[start].len() > config.max_characters {
            chunks.extend(split_long_line(lines[start], start + 1, config.max_characters));
            start = window_end;
            continue;
        }
        match next_chunk_start(&lines, start, window_end, config.min_lines) {
            Some(boundary) => {
                chunks.push(make_chunk(&lines, start, boundary));
                start = boundary;
            }
            None if window_end == lines.len() => {
                chunks.push(make_chunk(&lines, start, window_end));
                break;
            }
            None => {
                // No natural break in reach: cut, and overlap so the cut does not split meaning.
                chunks.push(make_chunk(&lines, start, window_end));
                start = window_end.saturating_sub(config.overlap_lines).max(start + 1);
            }
        }
    }
    chunks
}

fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

/// Exclusive end of the longest window from `start` within the line and character budgets.
fn window_end(lines: &[&str], start: usize, config: &ChunkingConfig) -> usize {
    let mut end = start;
    let mut size = 0;
    while end < lines.len() && end - start < config.max_lines {
        let line_size = lines[end].len() + 1;
        if end > start && size + line_size > config.max_characters {
            break;
        }
        size += line_size;
        end += 1;
    }
    end
}

/// Where the chunk starting at `start` should end, if the text offers a natural place.
///
/// Candidates are non-blank lines right after a blank line, within the window. The first one at
/// the same or a shallower indentation than the chunk's first line wins (the next function, the
/// next paragraph or section). Failing that, a window that does not reach the end of the text
/// takes the least indented candidate, so long units split between their own sub-blocks.
fn next_chunk_start(lines: &[&str], start: usize, window_end: usize, min_lines: usize) -> Option<usize> {
    let candidates: Vec<usize> = (start + 1..=window_end.min(lines.len() - 1))
        .filter(|&index| is_blank(lines[index - 1]) && !is_blank(lines[index]) && !is_closing_line(lines[index]))
        .filter(|&index| index >= start + min_lines || is_heading(lines[index]))
        .filter(|&index| !ends_with_heading(lines, start, index))
        .collect();
    let level = indentation(lines[start]);
    if let Some(&sibling) = candidates.iter().find(|&&index| indentation(lines[index]) <= level) {
        return Some(sibling);
    }
    if window_end == lines.len() {
        return None;
    }
    candidates.into_iter().min_by_key(|&index| (indentation(lines[index]), index))
}

/// A line of only closing brackets (`}`, `});`) ends the block above it; it never starts a unit.
fn is_closing_line(line: &str) -> bool {
    line.trim().chars().all(|character| matches!(character, '}' | ')' | ']' | ';' | ','))
}

fn indentation(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Markdown ATX heading (`# Title`). Top-level `# comments` in scripts match too, which is
/// harmless: they also introduce what follows.
fn is_heading(line: &str) -> bool {
    line.starts_with('#') && line.trim_start_matches('#').starts_with(' ')
}

/// Whether the last non-blank line of `start..end` is a heading, which belongs with what follows.
fn ends_with_heading(lines: &[&str], start: usize, end: usize) -> bool {
    lines[start..end].iter().rev().find(|line| !is_blank(line)).is_some_and(|line| is_heading(line))
}

/// Lines `start..end` without trailing blank lines.
fn make_chunk(lines: &[&str], start: usize, end: usize) -> Chunk {
    let mut end = end;
    while end > start + 1 && is_blank(lines[end - 1]) {
        end -= 1;
    }
    Chunk { start_line: start + 1, end_line: end, text: lines[start..end].join("\n") }
}

fn split_long_line(line: &str, line_number: usize, max_characters: usize) -> Vec<Chunk> {
    let mut pieces = Vec::new();
    let mut remaining = line;
    while !remaining.is_empty() {
        let mut split_at = remaining.len().min(max_characters);
        while !remaining.is_char_boundary(split_at) {
            split_at -= 1;
        }
        let (piece, rest) = remaining.split_at(split_at);
        pieces.push(Chunk { start_line: line_number, end_line: line_number, text: piece.to_string() });
        remaining = rest;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(min_lines: usize, max_lines: usize, overlap_lines: usize) -> ChunkingConfig {
        ChunkingConfig { min_lines, max_lines, overlap_lines, max_characters: 10_000 }
    }

    fn ranges(chunks: &[Chunk]) -> Vec<(usize, usize)> {
        chunks.iter().map(|chunk| (chunk.start_line, chunk.end_line)).collect()
    }

    #[test]
    fn short_text_is_a_single_chunk() {
        let chunks = chunk_text("fn main() {}\n", &ChunkingConfig::default());
        assert_eq!(chunks, [Chunk { start_line: 1, end_line: 1, text: "fn main() {}".into() }]);
    }

    #[test]
    fn splits_between_sibling_definitions_not_inside_them() {
        let code = "def first():
    return 1


def second():
    x = 2

    return x

def third():
    pass
";
        let chunks = chunk_text(code, &config(2, 10, 0));
        // The blank line inside `second` is not a split point: what follows it is indented deeper.
        assert_eq!(ranges(&chunks), [(1, 2), (5, 8), (10, 11)]);
    }

    #[test]
    fn markdown_sections_keep_their_heading() {
        let markdown = "# Title

intro

## Section

body
";
        let chunks = chunk_text(markdown, &config(4, 30, 0));
        assert_eq!(ranges(&chunks), [(1, 3), (5, 7)]);
    }

    #[test]
    fn headings_start_a_chunk_even_before_min_lines() {
        let markdown = "short paragraph

## Next

text
";
        let chunks = chunk_text(markdown, &config(10, 30, 0));
        assert_eq!(ranges(&chunks), [(1, 1), (3, 5)]);
    }

    #[test]
    fn units_shorter_than_min_lines_join_the_next_unit() {
        let code = "import os

import re

def run():
    pass
";
        let chunks = chunk_text(code, &config(4, 10, 0));
        assert_eq!(ranges(&chunks), [(1, 3), (5, 6)]);
    }

    #[test]
    fn long_blocks_split_between_their_own_sub_blocks() {
        let methods: String = (1..=4)
            .map(|number| {
                format!(
                    "    fn method_{number}() {{
        body();
    }}

"
                )
            })
            .collect();
        let chunks = chunk_text(
            &format!(
                "impl Thing {{
{methods}}}
"
            ),
            &config(2, 10, 0),
        );
        assert_eq!(ranges(&chunks), [(1, 4), (6, 8), (10, 12), (14, 18)]);
    }

    #[test]
    fn units_longer_than_the_window_are_cut_with_overlap() {
        let body: String = (1..=25).map(|number| format!("    line {number}\n")).collect();
        let chunks = chunk_text(&format!("def long():\n{body}"), &config(2, 10, 3));
        assert_eq!(ranges(&chunks), [(1, 10), (8, 17), (15, 24), (22, 26)]);
    }

    #[test]
    fn empty_and_blank_text_produce_no_chunks() {
        assert!(chunk_text("", &ChunkingConfig::default()).is_empty());
        assert!(chunk_text("\n   \n\t\n", &ChunkingConfig::default()).is_empty());
    }

    #[test]
    fn chunks_never_start_or_end_with_blank_lines() {
        let chunks = chunk_text("\n\nfirst\n\n\n\nsecond paragraph\nstill second\n\n", &config(1, 10, 0));
        assert_eq!(ranges(&chunks), [(3, 3), (7, 8)]);
    }

    #[test]
    fn crlf_line_endings_are_not_kept() {
        let chunks = chunk_text("first\r\nsecond\r\n", &ChunkingConfig::default());
        assert_eq!(chunks[0].text, "first\nsecond");
    }

    #[test]
    fn character_budget_ends_windows_early() {
        let text = "a".repeat(40) + "\n" + &"b".repeat(40) + "\n" + &"c".repeat(40);
        let chunks =
            chunk_text(&text, &ChunkingConfig { min_lines: 1, max_lines: 60, overlap_lines: 0, max_characters: 90 });
        assert_eq!(ranges(&chunks), [(1, 2), (3, 3)]);
    }

    #[test]
    fn oversized_single_line_is_split_on_char_boundaries() {
        let line = "é".repeat(10); // 2 bytes per char
        let chunks =
            chunk_text(&line, &ChunkingConfig { min_lines: 1, max_lines: 60, overlap_lines: 0, max_characters: 7 });
        assert!(chunks.iter().all(|chunk| chunk.start_line == 1 && chunk.text.len() <= 7));
        assert_eq!(chunks.iter().map(|chunk| chunk.text.as_str()).collect::<String>(), line);
    }
}
