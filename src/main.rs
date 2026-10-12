mod commands;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

use sems::device::{DeviceChoice, RuntimeLayout};
use sems::discovery::FileKind;
use sems::embedding::{ModelDirectory, ModelSource};
use sems::model_download::DownloadedModel;

use commands::progress::TerminalDownloadProgress;

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
    /// Find files like a given one: photos like a photo, notes like a note, clips like a clip.
    Similar(SimilarArguments),
    /// List the indexed folders, or show what is and is not indexed under one.
    Status {
        /// Folder to report on [default: every indexed folder; with --list, the current directory]
        path: Option<PathBuf>,
        /// List every file that is not indexed, not just the first few of each kind
        #[arg(long)]
        list: bool,
    },
    /// Remove a folder (or file) from the index, for example one that was deleted or moved.
    Forget {
        /// Folder or file to remove from the index
        path: PathBuf,
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
    #[command(flatten)]
    results: ResultArguments,
}

#[derive(Args)]
struct SimilarArguments {
    /// The example: any file sems can index, in the index or not
    file: PathBuf,
    /// Restrict results to this file or directory [default: current directory]
    path: Option<PathBuf>,
    #[command(flatten)]
    results: ResultArguments,
}

/// How many results to show, which, and how.
#[derive(Args)]
struct ResultArguments {
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
    let locations = &cli.locations;
    let outcome = match &cli.command {
        Some(Command::Index(arguments)) => commands::index::run(arguments, locations),
        Some(Command::Similar(arguments)) => commands::search::run_similar(arguments, locations),
        Some(Command::Status { path, list }) => commands::status::run_status(path.as_deref(), *list, locations),
        Some(Command::Forget { path }) => commands::status::run_forget(path, locations),
        Some(Command::Gpu { action: GpuAction::Install }) => commands::gpu::install(locations),
        Some(Command::Gpu { action: GpuAction::Remove }) => commands::gpu::remove(locations),
        None => commands::search::run_search(&cli.search, locations),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sems: {error:#}");
            ExitCode::FAILURE
        }
    }
}
