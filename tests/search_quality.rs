//! Retrieval quality on a small labelled corpus, with the real model. Guards against regressions
//! when changing chunking, prompts, fusion, or image handling, and shows which queries fail. Needs
//! the exported model and ONNX Runtime, so it is ignored by default:
//!
//!     SEMS_ONNXRUNTIME=path/to/onnxruntime.dll cargo test --release --test search_quality -- --ignored --nocapture
//!
//! Measured over one mixed index of text files and photos:
//! - retrieval: is the right file ranked first? (`eval_queries.json`)
//! - localization: does the top result contain the right line? (`eval_localization.json`)
//! - images: does a text description rank the right photo first, among text and other photos?
//!   (`eval_images.json`; photo file names are generic, so only pixels can answer)
//! - PDFs: does the top result point at the right page of the right PDF? (`eval_pdf.json`)
//! - recordings: does the top result point at the right moment of the right audio or video file,
//!   by what is said or what is seen? (`eval_media.json`; also needs ffmpeg)
//!
//! Recall saturates on a corpus this small, so image queries also report the similarity margin:
//! the right photo's score minus the best other result's. It shrinks before rankings break.
//! Set SEMS_EVAL_VISION_BUDGETS=280,140,70 to compare vision token budgets.

use std::path::{Path, PathBuf};

use sems::av::Ffmpeg;
use sems::discovery::FileKind;
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::indexer::{IndexOptions, SilentProgress, index_directory};
use sems::search::{CLIFF_IN_STANDARD_DEVIATIONS, ResultContent, SearchOptions, SearchResult, search};
use sems::store::{IndexIdentity, IndexStore, PathScope};
use serde::Deserialize;

/// Floors set just below measured quality; raise them when quality improves.
const MINIMUM_RECALL_AT_1: f32 = 0.9;
const MINIMUM_LOCATED_AT_1: f32 = 0.9;
const MINIMUM_IMAGE_RECALL_AT_1: f32 = 0.9;
const MINIMUM_PDF_PAGE_RECALL_AT_1: f32 = 0.85;
const MINIMUM_MEDIA_RECALL_AT_1: f32 = 0.85;

#[derive(Deserialize)]
struct LabelledQuery {
    query: String,
    expected: String,
    /// For localization queries: a line the top result must contain.
    expected_line: Option<usize>,
    /// For PDF queries: the page the top result must come from.
    expected_page: Option<usize>,
    /// For recording queries: a moment the top result's time span must contain.
    expected_seconds: Option<f64>,
    /// Restricts this query to one kind of file, as `sems --kind` does.
    kind: Option<String>,
    /// Other files that answer the query just as well: the eval video is made from three of the
    /// photos, so its segments show exactly what those photos show.
    #[serde(default)]
    also_correct: Vec<String>,
}

impl LabelledQuery {
    fn kind(&self) -> Option<FileKind> {
        self.kind.as_deref().map(|kind| match kind {
            "audio" => FileKind::Audio,
            "video" => FileKind::Video,
            "image" => FileKind::Image,
            other => panic!("unsupported kind {other:?} in an eval file"),
        })
    }

    fn accepts(&self, relative_path: &str) -> bool {
        relative_path == self.expected || self.also_correct.iter().any(|path| path == relative_path)
    }
}

struct Evaluation {
    store: IndexStore,
    encoder: GemmaEncoder,
    corpus: PathBuf,
}

impl Evaluation {
    fn top_result(&mut self, query: &str, kind: Option<FileKind>) -> Option<SearchResult> {
        let scope = PathScope::new(&self.corpus);
        let options = SearchOptions { limit: 1, kind, ..SearchOptions::default() };
        search(&self.store, &mut self.encoder, &scope, query, options).unwrap().into_iter().next()
    }

    fn relative_path(&self, result: &SearchResult) -> String {
        result.path.strip_prefix(&self.corpus).unwrap().to_string_lossy().replace('\\', "/")
    }

    fn describe(&self, result: &SearchResult) -> String {
        match &result.content {
            ResultContent::Text { start_line, end_line, .. } => {
                format!("{}:{start_line}-{end_line}", self.relative_path(result))
            }
            ResultContent::Image => self.relative_path(result),
            ResultContent::Pdf { page, .. } => format!("{} page {page}", self.relative_path(result)),
            ResultContent::Audio { start_seconds, end_seconds }
            | ResultContent::Video { start_seconds, end_seconds } => {
                format!("{} {start_seconds:.0}-{end_seconds:.0} s", self.relative_path(result))
            }
        }
    }

    fn is_hit(&self, result: &SearchResult, labelled: &LabelledQuery) -> bool {
        let line_matches = match (&result.content, labelled.expected_line) {
            (_, None) => true,
            (ResultContent::Text { start_line, end_line, .. }, Some(line)) => (*start_line..=*end_line).contains(&line),
            (_, Some(_)) => false,
        };
        let moment_matches = match (result.content.time_span(), labelled.expected_seconds) {
            (_, None) => true,
            (Some(span), Some(second)) => span.contains_second(second),
            (None, Some(_)) => false,
        };
        let page_matches = match (&result.content, labelled.expected_page) {
            (_, None) => true,
            (ResultContent::Pdf { page, .. }, Some(expected)) => *page == expected,
            (_, Some(_)) => false,
        };
        labelled.accepts(&self.relative_path(result)) && line_matches && page_matches && moment_matches
    }

    /// Mean and smallest gap between the expected result's similarity and the best other result's.
    fn margins(&mut self, queries: &[LabelledQuery]) -> (f32, f32) {
        let scope = PathScope::new(&self.corpus);
        // Raw separation: the relevance cutoff would remove the very results being compared against.
        let options =
            SearchOptions { limit: 20, one_result_per_file: true, relevance_cutoff: false, ..SearchOptions::default() };
        let margins: Vec<f32> = queries
            .iter()
            .map(|labelled| {
                let results = search(&self.store, &mut self.encoder, &scope, &labelled.query, options).unwrap();
                let (expected, others): (Vec<&SearchResult>, Vec<&SearchResult>) =
                    results.iter().partition(|result| labelled.accepts(&self.relative_path(result)));
                let expected = expected.first().map_or(-1.0, |result| result.similarity);
                let best_other = others.iter().map(|result| result.similarity).fold(-1.0, f32::max);
                expected - best_other
            })
            .collect();
        let mean = margins.iter().sum::<f32>() / margins.len() as f32;
        (mean, margins.iter().copied().fold(f32::INFINITY, f32::min))
    }

    /// Fraction of queries whose top result is the expected file (and line, page or moment, if
    /// labelled). A query's own kind overrides `kind`.
    fn score(&mut self, label: &str, queries: &[LabelledQuery], kind: Option<FileKind>) -> f32 {
        let mut misses = Vec::new();
        for labelled in queries {
            let top = self.top_result(&labelled.query, labelled.kind().or(kind));
            if !top.as_ref().is_some_and(|result| self.is_hit(result, labelled)) {
                let got = top.map(|result| self.describe(&result));
                let wanted = labelled.expected_line.map_or(String::new(), |line| format!(":{line}"));
                misses.push(format!("{:?}: wanted {}{wanted}, got {got:?}", labelled.query, labelled.expected));
            }
        }
        let score = 1.0 - misses.len() as f32 / queries.len() as f32;
        println!("{label:<34} {score:.2}");
        for miss in misses {
            println!("    {miss}");
        }
        score
    }
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn load_queries(file_name: &str) -> Vec<LabelledQuery> {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(file_name)).unwrap()).unwrap()
}

fn build_evaluation(vision_token_budget: usize) -> Evaluation {
    let library = std::env::var_os("SEMS_ONNXRUNTIME").expect("set SEMS_ONNXRUNTIME to the onnxruntime library path");
    let runtime = OnnxRuntime::load(Path::new(&library)).expect("failed to load ONNX Runtime");
    let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models").join("onnx");
    let model = EmbeddingModel::load(runtime, &model_directory, ExecutionDevice::Cpu).expect("failed to load model");
    let config = GemmaEncoderConfig { vision_token_budget, ..GemmaEncoderConfig::default() };
    let mut encoder = GemmaEncoder::new(model, config).unwrap();

    let identity = IndexIdentity::current(encoder.identity(), encoder.dimensions());
    let mut store = IndexStore::open_in_memory(identity).unwrap();
    let corpus = dunce::canonicalize(fixtures().join("eval_corpus")).unwrap();
    let options = IndexOptions {
        ffmpeg: Some(Ffmpeg::locate().expect("the recordings in the corpus need ffmpeg on PATH or SEMS_FFMPEG")),
        ..IndexOptions::default()
    };
    let started = std::time::Instant::now();
    let summary = index_directory(&mut store, &mut encoder, &corpus, &options, &mut SilentProgress).unwrap();
    println!("indexed the corpus in {:.1} s ({} chunks)", started.elapsed().as_secs_f64(), summary.chunks_embedded);
    assert!(summary.images_embedded > 0, "the corpus photos were not indexed");
    assert_eq!(summary.recordings_embedded, 4, "the corpus recordings were not all indexed");
    assert_eq!(summary.pdfs_without_text, 1, "the scanned PDF should be recognized as text-less");
    Evaluation { store, encoder, corpus }
}

fn budgets_to_evaluate() -> Vec<usize> {
    match std::env::var("SEMS_EVAL_VISION_BUDGETS") {
        Ok(list) => list.split(',').map(|budget| budget.trim().parse().expect("budgets are integers")).collect(),
        Err(_) => vec![GemmaEncoderConfig::default().vision_token_budget],
    }
}

#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn retrieval_localization_and_images_meet_floor() {
    let default_budget = GemmaEncoderConfig::default().vision_token_budget;
    for budget in budgets_to_evaluate() {
        println!("=== vision token budget {budget}{}", if budget == default_budget { " [default]" } else { "" });
        let mut evaluation = build_evaluation(budget);
        let recall = evaluation.score("text retrieval recall@1", &load_queries("eval_queries.json"), None);
        let located = evaluation.score("localization located@1", &load_queries("eval_localization.json"), None);
        let images = load_queries("eval_images.json");
        let image_recall = evaluation.score("image recall@1 (mixed with text)", &images, None);
        let image_only_recall = evaluation.score("image recall@1 (--kind image)", &images, Some(FileKind::Image));
        let pdf_recall = evaluation.score("pdf page recall@1", &load_queries("eval_pdf.json"), None);
        let media = load_queries("eval_media.json");
        let media_recall = evaluation.score("audio/video moment recall@1", &media, None);
        let (mean_margin, smallest_margin) = evaluation.margins(&images);
        println!("{:<34} mean {mean_margin:.3}, smallest {smallest_margin:.3}", "image similarity margin");
        let (mean_margin, smallest_margin) = evaluation.margins(&media);
        println!("{:<34} mean {mean_margin:.3}, smallest {smallest_margin:.3}", "audio/video similarity margin");

        if budget == default_budget {
            assert!(recall >= MINIMUM_RECALL_AT_1, "text retrieval regressed: {recall:.2}");
            assert!(located >= MINIMUM_LOCATED_AT_1, "localization regressed: {located:.2}");
            assert!(image_recall >= MINIMUM_IMAGE_RECALL_AT_1, "image retrieval regressed: {image_recall:.2}");
            assert!(image_only_recall >= image_recall, "filtering to images should never hurt");
            assert!(pdf_recall >= MINIMUM_PDF_PAGE_RECALL_AT_1, "PDF retrieval regressed: {pdf_recall:.2}");
            assert!(media_recall >= MINIMUM_MEDIA_RECALL_AT_1, "audio/video retrieval regressed: {media_recall:.2}");
        }
    }
}

/// Prints, per labelled query, the widest drop between consecutive top similarities (in standard
/// deviations of all similarities in scope) and how many results the cutoff keeps. Asserts the
/// cutoff never hides the expected file. Used to calibrate CLIFF_IN_STANDARD_DEVIATIONS.
#[test]
#[ignore = "requires exported model and SEMS_ONNXRUNTIME"]
fn relevance_cutoff_calibration() {
    let mut evaluation = build_evaluation(GemmaEncoderConfig::default().vision_token_budget);
    let scope = PathScope::new(&evaluation.corpus);
    let mut kept_total = 0;
    let mut queries = 0;
    let mut hidden = Vec::new();
    for file_name in
        ["eval_queries.json", "eval_localization.json", "eval_images.json", "eval_pdf.json", "eval_media.json"]
    {
        for labelled in load_queries(file_name) {
            let query = evaluation.encoder.encode_query(&labelled.query).unwrap();
            let (ranked, background) =
                evaluation.store.nearest_chunks_with_background(&scope, &query, 20, None).unwrap();
            let (cliff_rank, cliff) = ranked
                .windows(2)
                .enumerate()
                .map(|(index, pair)| (index + 1, (pair[0].1 - pair[1].1) / background.standard_deviation()))
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .unwrap();
            let results =
                search(&evaluation.store, &mut evaluation.encoder, &scope, &labelled.query, SearchOptions::default())
                    .unwrap();
            if !results.iter().any(|result| evaluation.relative_path(result) == labelled.expected) {
                hidden.push(labelled.query.clone());
            }
            println!(
                "{:<48} widest drop {cliff:.2} sd after rank {cliff_rank:>2}  -> {} results kept",
                labelled.query.chars().take(48).collect::<String>(),
                results.len()
            );
            kept_total += results.len();
            queries += 1;
        }
    }
    println!(
        "cliff factor {CLIFF_IN_STANDARD_DEVIATIONS}: {:.1} of up to 10 results kept on average",
        kept_total as f32 / queries as f32
    );
    assert!(hidden.is_empty(), "the relevance cutoff hid the expected file for {hidden:?}");
}
