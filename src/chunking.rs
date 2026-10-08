//! Splits file text into overlapping line windows for embedding.
//!
//! Line windows keep results addressable as `path:start-end`, and the overlap means a function or
//! paragraph that straddles a boundary still appears whole in at least one chunk.

/// A slice of a file. Line numbers are 1-based and inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkingConfig {
    pub max_lines: usize,
    pub overlap_lines: usize,
    /// Upper bound on chunk size, roughly 4 characters per token. Protects against minified or
    /// generated files whose few lines are enormous.
    pub max_characters: usize,
}

impl ChunkingConfig {
    /// Bumped whenever the chunking output changes, so stale indexes are rebuilt.
    pub const VERSION: u32 = 1;
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self { max_lines: 60, overlap_lines: 12, max_characters: 3_000 }
    }
}

pub fn chunk_text(text: &str, config: &ChunkingConfig) -> Vec<Chunk> {
    assert!(config.overlap_lines < config.max_lines, "overlap must be smaller than the window");
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        let (end, characters) = window_end(&lines, start, config);
        if characters == 0 {
            start = end; // a run of blank lines carries no meaning worth embedding
            continue;
        }
        if end == start + 1 && lines[start].len() > config.max_characters {
            chunks.extend(split_long_line(lines[start], start + 1, config.max_characters));
        } else {
            chunks.push(Chunk { start_line: start + 1, end_line: end, text: lines[start..end].join("\n") });
        }
        if end == lines.len() {
            break;
        }
        // Step back for overlap, but always make progress.
        start = end.saturating_sub(config.overlap_lines).max(start + 1);
    }
    chunks
}

/// Exclusive end index of the window starting at `start`, and its non-whitespace character count.
fn window_end(lines: &[&str], start: usize, config: &ChunkingConfig) -> (usize, usize) {
    let mut end = start;
    let mut size = 0;
    let mut meaningful_characters = 0;
    while end < lines.len() && end - start < config.max_lines {
        let line_size = lines[end].len() + 1;
        if end > start && size + line_size > config.max_characters {
            break;
        }
        size += line_size;
        meaningful_characters += lines[end].trim().len();
        end += 1;
    }
    (end, meaningful_characters)
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

    fn numbered_lines(count: usize) -> String {
        (1..=count).map(|number| format!("line {number}")).collect::<Vec<_>>().join("\n")
    }

    fn config(max_lines: usize, overlap_lines: usize) -> ChunkingConfig {
        ChunkingConfig { max_lines, overlap_lines, max_characters: 10_000 }
    }

    #[test]
    fn short_text_is_a_single_chunk() {
        let chunks = chunk_text("fn main() {}\n", &ChunkingConfig::default());
        assert_eq!(chunks, [Chunk { start_line: 1, end_line: 1, text: "fn main() {}".into() }]);
    }

    #[test]
    fn windows_overlap_and_cover_every_line() {
        let chunks = chunk_text(&numbered_lines(25), &config(10, 3));
        let ranges: Vec<_> = chunks.iter().map(|chunk| (chunk.start_line, chunk.end_line)).collect();
        assert_eq!(ranges, [(1, 10), (8, 17), (15, 24), (22, 25)]);
    }

    #[test]
    fn empty_and_blank_text_produce_no_chunks() {
        assert!(chunk_text("", &ChunkingConfig::default()).is_empty());
        assert!(chunk_text("\n   \n\t\n", &ChunkingConfig::default()).is_empty());
    }

    #[test]
    fn crlf_line_endings_are_not_kept() {
        let chunks = chunk_text("first\r\nsecond\r\n", &ChunkingConfig::default());
        assert_eq!(chunks[0].text, "first\nsecond");
    }

    #[test]
    fn character_budget_ends_windows_early() {
        let text = "a".repeat(40) + "\n" + &"b".repeat(40) + "\n" + &"c".repeat(40);
        let chunks = chunk_text(&text, &ChunkingConfig { max_lines: 60, overlap_lines: 0, max_characters: 90 });
        let ranges: Vec<_> = chunks.iter().map(|chunk| (chunk.start_line, chunk.end_line)).collect();
        assert_eq!(ranges, [(1, 2), (3, 3)]);
    }

    #[test]
    fn oversized_single_line_is_split_on_char_boundaries() {
        let line = "é".repeat(10); // 2 bytes per char
        let chunks = chunk_text(&line, &ChunkingConfig { max_lines: 60, overlap_lines: 0, max_characters: 7 });
        assert!(chunks.iter().all(|chunk| chunk.start_line == 1 && chunk.text.len() <= 7));
        assert_eq!(chunks.iter().map(|chunk| chunk.text.as_str()).collect::<String>(), line);
    }
}
