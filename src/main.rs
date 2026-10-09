use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};

use sems::av::Ffmpeg;
use sems::device::{DeviceChoice, RuntimeLayout, nvidia_driver_supports_cuda, plan_runtime};
use sems::discovery::{DiscoveryOptions, FileKind};
use sems::download::DownloadProgress;
use sems::embedding::{EmbeddingModel, ExecutionDevice, ModelDirectory, ModelSource, OnnxRuntime};
use sems::encoder::{Encoder, GemmaEncoder, GemmaEncoderConfig};
use sems::indexer::{IndexOptions, IndexProgress, IndexSummary, index_directory};
use sems::model_download::DownloadedModel;
use sems::render::{OutputFormat, Style, display_path, render};
use sems::runtime_download::{self, CUDA_FLAVOR};
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
    /// Manage the CUDA pack, which runs indexing on NVIDIA GPUs (Windows and Linux).
    Gpu {
        #[command(subcommand)]
        action: GpuAction,
    },
}

#[derive(Subcommand)]
enum GpuAction {
    /// Download the CUDA pack (about 1 GB, NVIDIA driver 580 or newer); `sems index` then uses it.
    Install,
    /// Delete the CUDA pack.
    Remove,
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
    /// Show weak results too, instead of only those that stand out from the rest of the index
    #[arg(long)]
    all: bool,
    /// Make result paths clickable links (auto: when the terminal is known to support them)
    #[arg(long, value_enum, env = "SEMS_HYPERLINKS", default_value = "auto")]
    hyperlinks: HyperlinkMode,
    /// Only return this kind of content
    #[arg(long, value_enum)]
    kind: Option<KindFilter>,
}

#[derive(Clone, Copy, ValueEnum)]
enum HyperlinkMode {
    Auto,
    Always,
    Never,
}

impl HyperlinkMode {
    fn enabled(self, stdout_is_terminal: bool) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => stdout_is_terminal && terminal_supports_hyperlinks(),
        }
    }
}

/// OSC 8 hyperlinks are ignored by many terminals but printed as garbage by some (old conhost, some
/// multiplexers), so `auto` only enables them for terminals that identify themselves as supporting
/// them. `--hyperlinks always` covers the rest.
fn terminal_supports_hyperlinks() -> bool {
    let set = |name: &str| std::env::var_os(name).is_some();
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    let vte_version: u32 = std::env::var("VTE_VERSION").ok().and_then(|version| version.parse().ok()).unwrap_or(0);
    set("WT_SESSION") // Windows Terminal
        || set("KITTY_WINDOW_ID")
        || set("WEZTERM_EXECUTABLE")
        || set("KONSOLE_VERSION")
        || matches!(term_program.as_str(), "vscode" | "iTerm.app" | "WezTerm" | "ghostty")
        || vte_version >= 5_000 // GNOME Terminal, Tilix, and other VTE terminals
}

#[derive(Clone, Copy, ValueEnum)]
enum KindFilter {
    Text,
    Image,
    Pdf,
    Audio,
    Video,
}

impl From<KindFilter> for FileKind {
    fn from(filter: KindFilter) -> Self {
        match filter {
            KindFilter::Text => FileKind::Text,
            KindFilter::Image => FileKind::Image,
            KindFilter::Pdf => FileKind::Pdf,
            KindFilter::Audio => FileKind::Audio,
            KindFilter::Video => FileKind::Video,
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
    /// Where to run the model while indexing: auto picks CUDA, then DirectML, then the CPU.
    /// Searches always use the CPU, which starts faster.
    #[arg(long, env = "SEMS_DEVICE", default_value = "auto", value_parser = parse_device_choice)]
    device: DeviceChoice,
    /// Skip text files larger than this many bytes
    #[arg(long, default_value_t = 1024 * 1024)]
    max_file_size: u64,
    /// Skip images and PDFs larger than this many bytes
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    max_media_size: u64,
    /// Do not index images (each takes ~5 s on a CPU, well under 1 s with --device cuda)
    #[arg(long)]
    skip_images: bool,
    /// Do not index audio files (each 25 s of audio takes ~3 s on a CPU, ~0.5 s with CUDA)
    #[arg(long)]
    skip_audio: bool,
    /// Do not index videos (each 9 s of video takes ~8 s on a CPU, under 1 s with CUDA)
    #[arg(long)]
    skip_video: bool,
}

/// Where sems keeps its index and finds its model; flags override environment variables.
#[derive(Args)]
struct Locations {
    /// Index database [default: <local data dir>/sems/index.db]
    #[arg(long, global = true, env = "SEMS_INDEX")]
    index: Option<PathBuf>,
    /// Directory with an exported model, used as is [default: downloaded into <local data dir>/sems/model]
    #[arg(long, global = true, env = "SEMS_MODEL_DIR")]
    model_dir: Option<PathBuf>,
    /// ONNX Runtime library [default: downloaded into <local data dir>/sems/runtime, see `--device`]
    #[arg(long, global = true, env = "SEMS_ONNXRUNTIME")]
    onnxruntime: Option<PathBuf>,
}

impl Locations {
    fn index_path(&self) -> Result<PathBuf> {
        match &self.index {
            Some(path) => Ok(path.clone()),
            None => Ok(data_directory()?.join("index.db")),
        }
    }

    /// An explicit model directory is used as is (a local export); otherwise the pinned model is
    /// downloaded into the data directory as its files are first needed.
    fn model_source(&self) -> Result<Box<dyn ModelSource>> {
        match &self.model_dir {
            Some(directory) => Ok(Box::new(ModelDirectory(directory.clone()))),
            None => Ok(Box::new(DownloadedModel::new(
                data_directory()?.join("model"),
                Box::new(TerminalDownloadProgress::new()),
            ))),
        }
    }

    fn runtime_layout(&self) -> Result<RuntimeLayout> {
        let executable = std::env::current_exe().context("cannot locate the sems executable")?;
        Ok(RuntimeLayout {
            executable_directory: executable.parent().context("executable has no directory")?.to_path_buf(),
            runtimes_directory: data_directory()?.join("runtime"),
        })
    }
}

fn data_directory() -> Result<PathBuf> {
    Ok(dirs::data_local_dir().context("cannot determine the local data directory")?.join("sems"))
}

fn parse_device_choice(value: &str) -> Result<DeviceChoice, String> {
    value.parse()
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = match &cli.command {
        Some(Command::Index(arguments)) => run_index(arguments, &cli.locations),
        Some(Command::Status { path }) => run_status(path.as_deref(), &cli.locations),
        Some(Command::Gpu { action: GpuAction::Install }) => run_gpu_install(&cli.locations),
        Some(Command::Gpu { action: GpuAction::Remove }) => run_gpu_remove(&cli.locations),
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
    IndexIdentity::current(config.identity(), config.dimensions)
}

/// Loads the model on the first device in the plan that works, and reports which one it was.
fn load_encoder(locations: &Locations, choice: DeviceChoice) -> Result<(GemmaEncoder, ExecutionDevice)> {
    let layout = locations.runtime_layout()?;
    let default_runtime = || runtime_download::default_runtime(&layout, &TerminalDownloadProgress::new());
    let plan = plan_runtime(
        choice,
        locations.onnxruntime.as_deref(),
        &layout,
        &nvidia_driver_supports_cuda,
        &default_runtime,
    )?;
    let runtime = OnnxRuntime::load(&plan.library)?;
    let mut failures = Vec::new();
    for &device in &plan.devices {
        match EmbeddingModel::load_from(runtime, locations.model_source()?, device) {
            Ok(model) => return Ok((GemmaEncoder::new(model, GemmaEncoderConfig::default())?, device)),
            Err(error) => failures.push(format!("{}: {error}", device_name(device))),
        }
    }
    bail!("could not load the model on any device ({})", failures.join("; "))
}

fn device_name(device: ExecutionDevice) -> &'static str {
    match device {
        ExecutionDevice::Cpu => "cpu",
        ExecutionDevice::Cuda => "cuda",
        ExecutionDevice::DirectMl => "directml",
    }
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

    let (mut encoder, _) = load_encoder(locations, DeviceChoice::Exactly(ExecutionDevice::Cpu))?;
    let options = SearchOptions {
        limit: arguments.limit,
        one_result_per_file: arguments.files_with_matches,
        kind: arguments.kind.map(FileKind::from),
        relevance_cutoff: !arguments.all,
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
    let stdout_is_terminal = std::io::stdout().is_terminal();
    let style = Style { color: stdout_is_terminal, hyperlinks: arguments.hyperlinks.enabled(stdout_is_terminal) };
    let working_directory = std::env::current_dir()?;
    write_stdout(&render(&results, format, style, &working_directory))
}

fn run_index(arguments: &IndexArguments, locations: &Locations) -> Result<()> {
    let root = resolve_scope_root(arguments.path.as_deref())?;
    if !root.is_dir() {
        bail!("{} is not a directory", root.display());
    }
    // Load the model first: a missing runtime should fail before the index is created or cleared.
    let (mut encoder, device) = load_encoder(locations, arguments.device)?;
    let index_path = locations.index_path()?;
    let mut store = if arguments.rebuild {
        IndexStore::recreate(&index_path, index_identity())?
    } else {
        IndexStore::open(&index_path, index_identity())?
    };
    let options = IndexOptions {
        discovery: DiscoveryOptions {
            max_text_size: arguments.max_file_size,
            max_media_size: arguments.max_media_size,
            include_images: !arguments.skip_images,
            include_audio: !arguments.skip_audio,
            include_video: !arguments.skip_video,
        },
        ffmpeg: Ffmpeg::locate(),
        ..IndexOptions::default()
    };
    let mut progress = TerminalProgress::new();
    let summary = index_directory(&mut store, &mut encoder, &root, &options, &mut progress)?;
    progress.finish();
    eprintln!("{}", summary_line(&root, &summary, encoder.dimensions(), device));
    Ok(())
}

fn run_status(path: Option<&Path>, locations: &Locations) -> Result<()> {
    let root = resolve_scope_root(path)?;
    let index_path = locations.index_path()?;
    let store = IndexStore::open(&index_path, index_identity())?;
    let statistics = store.statistics(&PathScope::new(&root))?;
    let working_directory = std::env::current_dir()?;
    write_stdout(&format!(
        "{}: {} files ({} images, {} audio and video), {} chunks indexed\nindex: {}\n",
        display_path(&root, &working_directory),
        statistics.files,
        statistics.images,
        statistics.recordings,
        statistics.chunks,
        index_path.display()
    ))
}

/// Checks the driver before downloading: the pack is about 1 GB and useless without it.
fn run_gpu_install(locations: &Locations) -> Result<()> {
    let layout = locations.runtime_layout()?;
    let download_size = runtime_download::cuda_pack_download_size()?;
    if runtime_download::is_cuda_pack_installed(&layout) {
        eprintln!("the CUDA pack is already installed in {}", layout.runtimes_directory.join(CUDA_FLAVOR).display());
        return Ok(());
    }
    if !nvidia_driver_supports_cuda() {
        bail!(
            "no NVIDIA GPU with a driver that supports CUDA 13 was found; the CUDA pack needs an NVIDIA GPU and \
             driver 580 or newer (other GPUs are used through DirectML on Windows)"
        );
    }
    eprintln!("downloading the CUDA pack ({})", readable_size(download_size));
    let library = runtime_download::install_cuda_pack(&layout, &TerminalDownloadProgress::new())?;
    eprintln!("installed the CUDA pack in {}; `sems index` now uses your NVIDIA GPU", display_parent(&library));
    Ok(())
}

fn run_gpu_remove(locations: &Locations) -> Result<()> {
    let directory = locations.runtime_layout()?.runtimes_directory.join(CUDA_FLAVOR);
    if !directory.exists() {
        eprintln!("the CUDA pack is not installed");
        return Ok(());
    }
    std::fs::remove_dir_all(&directory).with_context(|| format!("failed to delete {}", directory.display()))?;
    eprintln!("removed the CUDA pack from {}", directory.display());
    Ok(())
}

fn display_parent(path: &Path) -> String {
    path.parent().unwrap_or(path).display().to_string()
}

fn summary_line(root: &Path, summary: &IndexSummary, dimensions: usize, device: ExecutionDevice) -> String {
    let skipped: Vec<String> = [
        (summary.files_skipped_binary, "binary"),
        (summary.files_skipped_too_large, "too large"),
        (summary.images_skipped_too_small, "tiny images"),
        (summary.images_unreadable, "unreadable images"),
        (summary.pdfs_without_text, "PDFs without text (scans need OCR)"),
        (summary.pdfs_unreadable, "unreadable PDFs"),
        (summary.recordings_without_ffmpeg, "audio/video files (install ffmpeg to index them)"),
        (summary.recordings_unreadable, "unreadable audio/video files"),
        (summary.recordings_without_content, "silent audio/video files"),
    ]
    .into_iter()
    .filter(|(count, _)| *count > 0)
    .map(|(count, reason)| format!("{count} {reason}"))
    .collect();
    let skipped = if skipped.is_empty() { String::new() } else { format!(", skipped {}", skipped.join(", ")) };
    let media: Vec<String> = [(summary.images_embedded, "images"), (summary.recordings_embedded, "audio/video")]
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .map(|(count, kind)| format!("{count} {kind}"))
        .collect();
    let media = if media.is_empty() { String::new() } else { format!(" ({})", media.join(", ")) };
    format!(
        "indexed {}: {} files ({} unchanged, {} embedded{media} into {} chunks, {} removed{skipped}) in {:.1}s [{}, {dimensions}d]",
        root.display(),
        summary.files_seen,
        summary.files_unchanged,
        summary.files_embedded,
        summary.chunks_embedded,
        summary.files_removed,
        summary.elapsed.as_secs_f64(),
        device_name(device),
    )
}

/// Writes to stdout, treating a closed pipe (`sems ... | head`) as success rather than a panic.
fn write_stdout(text: &str) -> Result<()> {
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => other.context("failed to write output"),
    }
}

/// Download progress (model and ONNX Runtime) on stderr: a line redrawn in place on a terminal,
/// otherwise one line per file so logs stay readable.
struct TerminalDownloadProgress {
    redraw: bool,
}

impl TerminalDownloadProgress {
    fn new() -> Self {
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

/// `4 KB`, `274 MB`: whole units, decimal as download sizes usually are.
fn readable_size(bytes: u64) -> String {
    if bytes < 1_000_000 { format!("{} KB", bytes.div_ceil(1_000)) } else { format!("{} MB", bytes / 1_000_000) }
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
