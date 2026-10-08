use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use sems::chunking::ChunkingConfig;
use sems::discovery::{DiscoveryOptions, FileKind};
use sems::embedding::{EmbeddingModel, ExecutionDevice, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::indexer::{IndexOptions, IndexProgress, IndexSummary, index_directory};
use sems::render::{OutputFormat, Style, display_path, render};
use sems::search::{SearchOptions, search};
use sems::store::{IndexIdentity, IndexStore, PathScope};

/// Semantic search for local files: like grep, but matches meaning instead of exact text.
#[derive(Parser)]
#[command(name = "sems", version, args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    search: SearchArguments,
    #[command(flatten)]
    locations: Locations,
}

#[derive(Subcommand)]
enum Command {
    /// Index (or incrementally update) the files under a directory.
    Index(IndexArguments),
    /// Show what is indexed under a directory.
    Status {
        /// Directory to report on [default: current directory]
        path: Option<PathBuf>,
    },
}

#[derive(Args)]
struct SearchArguments {
    /// What to look for, in natural language or code
    #[arg(required = true)]
    query: Option<String>,
    /// Restrict results to this file or directory [default: current directory]
    path: Option<PathBuf>,
    /// Maximum number of results
    #[arg(short = 'n', long, default_value_t = 10)]
    limit: usize,
    /// Print matching file paths only, best first
    #[arg(short = 'l', long, conflicts_with = "json")]
    files_with_matches: bool,
    /// Print results as JSON, including full chunk text
    #[arg(long)]
    json: bool,
    /// Only return this kind of content
    #[arg(long, value_enum)]
    kind: Option<KindFilter>,
}

#[derive(Clone, Copy, ValueEnum)]
enum KindFilter {
    Text,
    Image,
}

impl From<KindFilter> for FileKind {
    fn from(filter: KindFilter) -> Self {
        match filter {
            KindFilter::Text => FileKind::Text,
            KindFilter::Image => FileKind::Image,
        }
    }
}

#[derive(Args)]
struct IndexArguments {
    /// Directory to index [default: current directory]
    path: Option<PathBuf>,
    /// Discard the whole index and rebuild this directory from scratch
    #[arg(long)]
    rebuild: bool,
    /// Where to run the model while indexing. Searches always use the CPU, which starts faster.
    #[arg(long, env = "SEMS_DEVICE", default_value = "cpu", value_parser = parse_device)]
    device: ExecutionDevice,
    /// Skip text files larger than this many bytes
    #[arg(long, default_value_t = 1024 * 1024)]
    max_file_size: u64,
    /// Skip images larger than this many bytes
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    max_image_size: u64,
    /// Do not index images (each takes ~5 s on a CPU, well under 1 s with --device cuda)
    #[arg(long)]
    skip_images: bool,
}

/// Where sems keeps its index and finds its model; flags override environment variables.
#[derive(Args)]
struct Locations {
    /// Index database [default: <local data dir>/sems/index.db]
    #[arg(long, global = true, env = "SEMS_INDEX")]
    index: Option<PathBuf>,
    /// Directory with the exported model [default: <local data dir>/sems/model]
    #[arg(long, global = true, env = "SEMS_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    /// ONNX Runtime library [default: the installed runtime for the device, see `--device`]
    #[arg(long, global = true, env = "SEMS_ONNXRUNTIME")]
    onnxruntime: Option<PathBuf>,
}

/// ONNX Runtime builds sems knows how to find. Only one can be loaded per process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeFlavor {
    /// Bundled default; its CPU provider also serves searches and CPU indexing.
    DirectMl,
    /// Optional download for fast indexing on NVIDIA GPUs.
    Cuda,
}

impl RuntimeFlavor {
    fn for_device(device: ExecutionDevice) -> Self {
        match device {
            ExecutionDevice::Cuda => Self::Cuda,
            ExecutionDevice::Cpu | ExecutionDevice::DirectMl => Self::DirectMl,
        }
    }

    fn directory_name(self) -> &'static str {
        match self {
            Self::DirectMl => "directml",
            Self::Cuda => "cuda",
        }
    }
}

impl Locations {
    fn index_path(&self) -> Result<PathBuf> {
        match &self.index {
            Some(path) => Ok(path.clone()),
            None => Ok(data_directory()?.join("index.db")),
        }
    }

    fn model_directory(&self) -> Result<PathBuf> {
        let directory = match &self.model_dir {
            Some(directory) => directory.clone(),
            None => data_directory()?.join("model"),
        };
        if !directory.join("text_encoder.onnx").is_file() {
            bail!(
                "no exported model in {} (export it there with tools/export/export_onnx.py, \
                 or set --model-dir / SEMS_MODEL_DIR)",
                directory.display()
            );
        }
        Ok(directory)
    }

    /// An explicit `--onnxruntime` wins; otherwise the installed runtime for the device's flavor.
    /// Release builds ship the DirectML runtime next to the executable; development setups install
    /// runtimes under the data directory with tools/runtime/install_runtime.py.
    fn onnxruntime_library(&self, device: ExecutionDevice) -> Result<PathBuf> {
        if let Some(library) = &self.onnxruntime {
            if !library.is_file() {
                bail!("ONNX Runtime not found at {} (from --onnxruntime or SEMS_ONNXRUNTIME)", library.display());
            }
            return Ok(library.clone());
        }
        let flavor = RuntimeFlavor::for_device(device);
        let mut candidates = Vec::new();
        if flavor == RuntimeFlavor::DirectMl {
            candidates.push(std::env::current_exe()?.with_file_name(ONNXRUNTIME_LIBRARY));
        }
        candidates.push(data_directory()?.join("runtime").join(flavor.directory_name()).join(ONNXRUNTIME_LIBRARY));
        if let Some(found) = candidates.iter().find(|candidate| candidate.is_file()) {
            return Ok(found.clone());
        }
        let searched: Vec<String> = candidates.iter().map(|candidate| candidate.display().to_string()).collect();
        bail!(
            "no {name} ONNX Runtime installed (looked for {}); install it with \
             `python tools/runtime/install_runtime.py {name}` or set --onnxruntime",
            searched.join(", "),
            name = flavor.directory_name()
        )
    }
}

#[cfg(windows)]
const ONNXRUNTIME_LIBRARY: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
const ONNXRUNTIME_LIBRARY: &str = "libonnxruntime.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
const ONNXRUNTIME_LIBRARY: &str = "libonnxruntime.so";

fn data_directory() -> Result<PathBuf> {
    Ok(dirs::data_local_dir().context("cannot determine the local data directory")?.join("sems"))
}

fn parse_device(value: &str) -> Result<ExecutionDevice, String> {
    value.parse()
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = match &cli.command {
        Some(Command::Index(arguments)) => run_index(arguments, &cli.locations),
        Some(Command::Status { path }) => run_status(path.as_deref(), &cli.locations),
        None => run_search(&cli.search, &cli.locations),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sems: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn index_identity() -> IndexIdentity {
    let config = GemmaEncoderConfig::default();
    IndexIdentity {
        encoder: config.identity(),
        dimensions: config.dimensions,
        chunker_version: ChunkingConfig::VERSION,
    }
}

fn load_encoder(locations: &Locations, device: ExecutionDevice) -> Result<GemmaEncoder> {
    let runtime = OnnxRuntime::load(&locations.onnxruntime_library(device)?)?;
    let model = EmbeddingModel::load(runtime, &locations.model_directory()?, device)?;
    GemmaEncoder::new(model, GemmaEncoderConfig::default())
}

/// Canonical form of a user-supplied path (defaulting to the working directory).
fn resolve_scope_root(path: Option<&Path>) -> Result<PathBuf> {
    let path = path.map_or_else(std::env::current_dir, |path| Ok(path.to_path_buf()))?;
    dunce::canonicalize(&path).with_context(|| format!("{} does not exist", path.display()))
}

fn run_search(arguments: &SearchArguments, locations: &Locations) -> Result<()> {
    let query = arguments.query.as_deref().context("a query is required")?;
    let root = resolve_scope_root(arguments.path.as_deref())?;
    let store = IndexStore::open(&locations.index_path()?, index_identity())?;
    let scope = PathScope::new(&root);
    if store.statistics(&scope)?.files == 0 {
        bail!("nothing is indexed under {}; run `sems index {}` first", root.display(), root.display());
    }

    let mut encoder = load_encoder(locations, ExecutionDevice::Cpu)?;
    let options = SearchOptions {
        limit: arguments.limit,
        one_result_per_file: arguments.files_with_matches,
        kind: arguments.kind.map(FileKind::from),
        ..SearchOptions::default()
    };
    let results = search(&store, &mut encoder, &scope, query, options)?;

    let format = if arguments.json {
        OutputFormat::Json
    } else if arguments.files_with_matches {
        OutputFormat::FilesWithMatches
    } else {
        OutputFormat::Text
    };
    let style = Style { color: std::io::stdout().is_terminal() };
    let working_directory = std::env::current_dir()?;
    write_stdout(&render(&results, format, style, &working_directory))
}

fn run_index(arguments: &IndexArguments, locations: &Locations) -> Result<()> {
    let root = resolve_scope_root(arguments.path.as_deref())?;
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    // Load the model first: a missing runtime should fail before the index is created or cleared.
    let mut encoder = load_encoder(locations, arguments.device)?;
    let index_path = locations.index_path()?;
    let mut store = if arguments.rebuild {
        IndexStore::recreate(&index_path, index_identity())?
    } else {
        IndexStore::open(&index_path, index_identity())?
    };
    let options = IndexOptions {
        discovery: DiscoveryOptions {
            max_text_size: arguments.max_file_size,
            max_image_size: arguments.max_image_size,
            include_images: !arguments.skip_images,
        },
        ..IndexOptions::default()
    };
    let mut progress = TerminalProgress::new();
    let summary = index_directory(&mut store, &mut encoder, &root, &options, &mut progress)?;
    progress.finish();
    eprintln!("{}", summary_line(&root, &summary, encoder.dimensions()));
    Ok(())
}

fn run_status(path: Option<&Path>, locations: &Locations) -> Result<()> {
    let root = resolve_scope_root(path)?;
    let index_path = locations.index_path()?;
    let store = IndexStore::open(&index_path, index_identity())?;
    let statistics = store.statistics(&PathScope::new(&root))?;
    let working_directory = std::env::current_dir()?;
    write_stdout(&format!(
        "{}: {} files ({} images), {} chunks indexed\nindex: {}\n",
        display_path(&root, &working_directory),
        statistics.files,
        statistics.images,
        statistics.chunks,
        index_path.display()
    ))
}

fn summary_line(root: &Path, summary: &IndexSummary, dimensions: usize) -> String {
    let skipped: Vec<String> = [
        (summary.files_skipped_binary, "binary"),
        (summary.files_skipped_too_large, "too large"),
        (summary.images_skipped_too_small, "tiny images"),
        (summary.images_unreadable, "unreadable images"),
    ]
    .into_iter()
    .filter(|(count, _)| *count > 0)
    .map(|(count, reason)| format!("{count} {reason}"))
    .collect();
    let skipped = if skipped.is_empty() { String::new() } else { format!(", skipped {}", skipped.join(", ")) };
    let images =
        if summary.images_embedded > 0 { format!(" ({} images)", summary.images_embedded) } else { String::new() };
    format!(
        "indexed {}: {} files ({} unchanged, {} embedded{images} into {} chunks, {} removed{skipped}) in {:.1}s [{dimensions}d]",
        root.display(),
        summary.files_seen,
        summary.files_unchanged,
        summary.files_embedded,
        summary.chunks_embedded,
        summary.files_removed,
        summary.elapsed.as_secs_f64(),
    )
}

/// Writes to stdout, treating a closed pipe (`sems ... | head`) as success rather than a panic.
fn write_stdout(text: &str) -> Result<()> {
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => other.context("failed to write output"),
    }
}

/// Single-line progress on stderr, redrawn in place.
struct TerminalProgress {
    started: Instant,
    files_total: usize,
    enabled: bool,
}

impl TerminalProgress {
    fn new() -> Self {
        Self { started: Instant::now(), files_total: 0, enabled: std::io::stderr().is_terminal() }
    }

    fn finish(&self) {
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
