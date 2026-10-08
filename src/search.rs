//! Hybrid retrieval: semantic (vector) and keyword (BM25) results merged by reciprocal rank fusion.
//!
//! Vectors capture meaning ("retry with backoff" finds `sleep(delay * 2)`); keywords catch exact
//! identifiers that embeddings blur. RRF combines the two rankings by position, which sidesteps
//! the fact that cosine similarities and BM25 scores live on unrelated scales.
//!
//! Text and images share one ranking. EmbeddingGemma 2 places them in one space with comparable
//! similarities (on tests/fixtures, photo queries score 0.70-0.81 against the right photo and at
//! most 0.67 against any text; text queries score at most 0.66 against any photo), so no
//! per-modality calibration is needed.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use crate::discovery::FileKind;
use crate::encoder::Encoder;
use crate::store::{IndexStore, PathScope};

/// Candidates taken from each retriever before fusion.
const CANDIDATES_PER_RETRIEVER: usize = 100;
/// Standard RRF damping constant; larger values flatten the advantage of top ranks.
const RRF_K: f32 = 60.0;

/// Number of top semantic candidates examined for a cliff (see [`relevance_floor`]).
const CLIFF_WINDOW: usize = 20;
/// A drop between consecutive candidates at least this many background standard deviations wide
/// is a cliff: the end of the results that stand out. Calibrated by search_quality's
/// relevance_separation.
pub const CLIFF_IN_STANDARD_DEVIATIONS: f32 = 0.75;

#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub limit: usize,
    /// Weight of the keyword ranking relative to the semantic one.
    pub keyword_weight: f32,
    /// Keep only each file's best chunk, so `limit` counts files (for `sems -l`).
    pub one_result_per_file: bool,
    /// Restrict results to text or to images; `None` searches both.
    pub kind: Option<FileKind>,
    /// Drop results below the similarity cliff (see [`relevance_floor`]); `sems --all` turns
    /// this off.
    pub relevance_cutoff: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self { limit: 10, keyword_weight: 1.0, one_result_per_file: false, kind: None, relevance_cutoff: true }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub path: PathBuf,
    /// Cosine similarity between query and chunk, in [-1, 1]. Shown to users; ranking uses RRF.
    pub similarity: f32,
    #[serde(flatten)]
    pub content: ResultContent,
}

/// What matched. Serialized with a `kind` tag, so JSON consumers can switch on it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResultContent {
    /// Lines of a text file (1-based, inclusive).
    Text { start_line: usize, end_line: usize, text: String },
    /// A whole image file.
    Image,
    /// Text from one page of a PDF (1-based page). Lines count within the page's extracted text,
    /// which readers never see, so they are kept for de-duplication but not serialized.
    Pdf {
        page: usize,
        #[serde(skip_serializing)]
        start_line: usize,
        #[serde(skip_serializing)]
        end_line: usize,
        text: String,
    },
}

pub fn search(
    store: &IndexStore,
    encoder: &mut dyn Encoder,
    scope: &PathScope,
    query: &str,
    options: SearchOptions,
) -> Result<Vec<SearchResult>> {
    let query_vector = encoder.encode_query(query)?;
    let (semantic, background) =
        store.nearest_chunks_with_background(scope, &query_vector, CANDIDATES_PER_RETRIEVER, options.kind)?;
    // Images carry no text, so keyword retrieval only ever contributes text chunks.
    let keyword = match keyword_query(query) {
        Some(fts_query) if options.kind != Some(FileKind::Image) => {
            store.keyword_chunks(scope, &fts_query, CANDIDATES_PER_RETRIEVER)?
        }
        _ => Vec::new(),
    };

    let semantic_similarities: Vec<f32> = semantic.iter().map(|(_, similarity)| *similarity).collect();
    let floor = relevance_floor(&semantic_similarities, background.standard_deviation(), CLIFF_IN_STANDARD_DEVIATIONS)
        .filter(|_| options.relevance_cutoff);

    let fused = reciprocal_rank_fusion(&[(&semantic, 1.0), (&keyword, options.keyword_weight)]);
    let mut results: Vec<SearchResult> = Vec::with_capacity(options.limit);
    for chunk_id in fused {
        if results.len() == options.limit {
            break;
        }
        let chunk = store.chunk(chunk_id)?;
        let content = match chunk.kind {
            FileKind::Text => {
                ResultContent::Text { start_line: chunk.start_line, end_line: chunk.end_line, text: chunk.text }
            }
            FileKind::Image => ResultContent::Image,
            FileKind::Pdf => ResultContent::Pdf {
                page: chunk.page.unwrap_or(1),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                text: chunk.text,
            },
        };
        let similarity = dot_product(&query_vector, &chunk.embedding);
        // Exact keyword matches are evidence on their own, and the best result always shows.
        let below_floor = floor.is_some_and(|floor| similarity < floor);
        let matched_every_keyword = keyword.iter().any(|(id, _)| *id == chunk_id);
        if below_floor && !matched_every_keyword && !results.is_empty() {
            continue;
        }
        let candidate = SearchResult { similarity, path: chunk.path, content };
        // Chunks overlap by design; a lower-ranked neighbour would mostly repeat a better result.
        let redundant = if options.one_result_per_file {
            results.iter().any(|kept| kept.path == candidate.path)
        } else {
            results.iter().any(|kept| overlaps(kept, &candidate))
        };
        if !redundant {
            results.push(candidate);
        }
    }
    Ok(results)
}

fn overlaps(left: &SearchResult, right: &SearchResult) -> bool {
    if left.path != right.path {
        return false;
    }
    match (&left.content, &right.content) {
        (
            ResultContent::Text { start_line: left_start, end_line: left_end, .. },
            ResultContent::Text { start_line: right_start, end_line: right_end, .. },
        ) => left_start <= right_end && right_start <= left_end,
        (
            ResultContent::Pdf { page: left_page, start_line: left_start, end_line: left_end, .. },
            ResultContent::Pdf { page: right_page, start_line: right_start, end_line: right_end, .. },
        ) => left_page == right_page && left_start <= right_end && right_start <= left_end,
        _ => true,
    }
}

/// The similarity below which results are cut, if the top candidates show a cliff.
///
/// Relevant results stand out together, then similarities fall to a long, flat tail of near-misses.
/// The cliff is the widest drop between consecutive candidates within the first [`CLIFF_WINDOW`];
/// it counts only if it is at least `cliff_factor` times `spread` (the standard deviation of all
/// similarities in scope), so a smooth curve of many relevant results is not cut at all.
///
/// Absolute and standardized thresholds were measured first and failed: on tests/fixtures right
/// answers scored from 0.663 while wrong ones reached 0.735, and in a code-only index the scores
/// cluster so tightly (std 0.026) that unrelated chunks reach 2.6 standard deviations.
pub fn relevance_floor(similarities_descending: &[f32], spread: f32, cliff_factor: f32) -> Option<f32> {
    let window = &similarities_descending[..similarities_descending.len().min(CLIFF_WINDOW)];
    let (index, drop) = window
        .windows(2)
        .enumerate()
        .map(|(index, pair)| (index, pair[0] - pair[1]))
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    (spread > 0.0 && drop >= cliff_factor * spread).then_some(window[index])
}

/// Merges rankings: each list contributes `weight / (RRF_K + rank)` per item. Returns ids by
/// descending fused score; ties keep a deterministic order by id.
pub fn reciprocal_rank_fusion(rankings: &[(&Vec<(i64, f32)>, f32)]) -> Vec<i64> {
    let mut scores: HashMap<i64, f32> = HashMap::new();
    for (ranking, weight) in rankings {
        for (rank, (id, _)) in ranking.iter().enumerate() {
            *scores.entry(*id).or_default() += weight / (RRF_K + rank as f32 + 1.0);
        }
    }
    let mut ids: Vec<(i64, f32)> = scores.into_iter().collect();
    ids.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    ids.into_iter().map(|(id, _)| id).collect()
}

/// Builds an FTS5 query requiring every query word. Any-word (OR) matching let filler words like
/// "to" and "for" outrank the semantic results (recall@1 dropped from 1.00 to 0.74 on the
/// search_quality corpus); requiring all words keeps keywords as a precision signal for
/// identifiers and exact phrases. Each term is quoted, so user input can never be
/// interpreted as FTS5 syntax (`NEAR`, `*`, column filters, unbalanced quotes).
pub fn keyword_query(query: &str) -> Option<String> {
    let mut terms: Vec<String> = Vec::new();
    for term in query.split(|character: char| !(character.is_alphanumeric() || character == '_')) {
        let term = term.to_lowercase();
        if term.chars().count() >= 2 && !terms.contains(&term) {
            terms.push(term);
        }
    }
    if terms.is_empty() {
        return None;
    }
    Some(terms.iter().map(|term| format!("\"{term}\"")).collect::<Vec<_>>().join(" AND "))
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_rewards_agreement_between_rankings() {
        let semantic = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let keyword = vec![(3, 12.0), (4, 9.0)];
        let fused = reciprocal_rank_fusion(&[(&semantic, 1.0), (&keyword, 1.0)]);
        assert_eq!(fused[0], 3, "found by both retrievers, so it should win");
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn fusion_weight_zero_ignores_a_ranking() {
        let semantic = vec![(1, 0.9)];
        let keyword = vec![(2, 5.0)];
        assert_eq!(reciprocal_rank_fusion(&[(&semantic, 1.0), (&keyword, 0.0)])[0], 1);
    }

    #[test]
    fn overlap_requires_same_file_and_shared_lines() {
        let result = |path: &str, start_line, end_line| SearchResult {
            path: PathBuf::from(path),
            similarity: 0.0,
            content: ResultContent::Text { start_line, end_line, text: String::new() },
        };
        assert!(overlaps(&result("a", 1, 60), &result("a", 49, 108)));
        assert!(!overlaps(&result("a", 1, 60), &result("a", 61, 120)));
        assert!(!overlaps(&result("a", 1, 60), &result("b", 1, 60)));
    }

    #[test]
    fn a_cliff_after_the_relevant_results_sets_the_floor() {
        // Measured for "a cat lying down" on this repository: two matches, then a flat tail.
        let similarities = [0.723, 0.723, 0.694, 0.691, 0.688, 0.687, 0.687, 0.685];
        assert_eq!(relevance_floor(&similarities, 0.026, 0.75), Some(0.723));
    }

    #[test]
    fn a_smooth_curve_is_not_cut() {
        // Measured for "how are tiny images skipped": many relevant chunks, no cliff.
        let similarities = [0.767, 0.759, 0.750, 0.750, 0.749, 0.742, 0.739, 0.738, 0.737, 0.732];
        assert_eq!(relevance_floor(&similarities, 0.039, 0.75), None);
    }

    #[test]
    fn no_floor_without_candidates_or_spread() {
        assert_eq!(relevance_floor(&[], 0.03, 0.75), None);
        assert_eq!(relevance_floor(&[0.9], 0.03, 0.75), None);
        assert_eq!(relevance_floor(&[0.9, 0.1], 0.0, 0.75), None);
    }

    #[test]
    fn keyword_query_quotes_terms_and_drops_noise() {
        assert_eq!(
            keyword_query("retry_with_backoff, a HTTP!").as_deref(),
            Some("\"retry_with_backoff\" AND \"http\"")
        );
        assert_eq!(keyword_query("say \"hi\" NEAR(x*)").as_deref(), Some("\"say\" AND \"hi\" AND \"near\""));
        assert_eq!(keyword_query("? !"), None);
    }
}
