//! Searching by query and by example (`sems similar`).

use std::io::IsTerminal;
use std::path::Path;

use anyhow::{Context, Result, bail};

use sems::av::Ffmpeg;
use sems::device::DeviceChoice;
use sems::discovery::FileKind;
use sems::embedding::ExecutionDevice;
use sems::encoder::{Encoder, GemmaEncoder};
use sems::example::example_embedding;
use sems::freshness::{Freshness, refresh_before_search};
use sems::indexer::{IndexOptions, SilentProgress};
use sems::render::{OutputFormat, Style, render};
use sems::search::{SearchOptions, SearchResult, search, search_similar};
use sems::segments::MediaSegmentation;
use sems::store::{IndexStore, PathScope};

use super::progress::files;
use super::{load_encoder, open_store, resolve_existing_path, write_stdout};
use crate::{Locations, ResultArguments, SearchArguments, SimilarArguments};

pub fn run_search(arguments: &SearchArguments, locations: &Locations) -> Result<()> {
    let query = arguments.query.as_deref().context("a query is required")?;
    let root = resolve_existing_path(arguments.path.as_deref())?;
    let (store, mut encoder) = prepare(locations, &root)?;
    let options = search_options(&arguments.results);
    let results = search(&store, &mut encoder, &PathScope::new(&root), query, options)?;
    print_results(&results, &arguments.results)
}

pub fn run_similar(arguments: &SimilarArguments, locations: &Locations) -> Result<()> {
    let example = resolve_existing_path(Some(&arguments.file))?;
    if !example.is_file() {
        bail!("{} is not a file", example.display());
    }
    let root = resolve_existing_path(arguments.path.as_deref())?;
    let (store, mut encoder) = prepare(locations, &root)?;
    let ffmpeg = Ffmpeg::locate();
    let vector = example_embedding(&store, &mut encoder, &example, ffmpeg.as_ref(), &MediaSegmentation::default())?;
    let results =
        search_similar(&store, &PathScope::new(&root), &vector, &example, search_options(&arguments.results))?;
    print_results(&results, &arguments.results)
}

/// Opens the index, checks there is something to search under `root`, loads the model on the
/// CPU (faster to start than a GPU), and brings the index for `root` up to date if that is cheap.
fn prepare(locations: &Locations, root: &Path) -> Result<(IndexStore, GemmaEncoder)> {
    let mut store = open_store(locations)?;
    if store.statistics(&PathScope::new(root))?.files == 0 {
        bail!("nothing is indexed under {}; run `sems index {}` first", root.display(), root.display());
    }
    let (mut encoder, _) = load_encoder(locations, DeviceChoice::Exactly(ExecutionDevice::Cpu))?;
    refresh(&mut store, &mut encoder, root)?;
    Ok((store, encoder))
}

/// Applies small text changes under `root`, or says that `sems index` should run, using the
/// options its folder was indexed with. Folders indexed before sems recorded them are skipped.
fn refresh(store: &mut IndexStore, encoder: &mut dyn Encoder, root: &Path) -> Result<()> {
    let Some(indexed) = store.indexed_root_containing(root)? else {
        return Ok(());
    };
    let options = IndexOptions { discovery: indexed.options, ..IndexOptions::default() };
    match refresh_before_search(store, encoder, root, &options, &mut SilentProgress)? {
        Freshness::UpToDate => {}
        Freshness::Updated(changes) => eprintln!("sems: updated {} before searching", files(changes.changed_files())),
        Freshness::Stale(changes) => eprintln!(
            "sems: {} changed since the last index; `sems index {}` includes them",
            files(changes.changed_files()),
            indexed.path.display()
        ),
    }
    Ok(())
}

fn search_options(arguments: &ResultArguments) -> SearchOptions {
    SearchOptions {
        limit: arguments.limit,
        one_result_per_file: arguments.files_with_matches,
        kind: arguments.kind.map(FileKind::from),
        relevance_cutoff: !arguments.all,
        ..SearchOptions::default()
    }
}

fn print_results(results: &[SearchResult], arguments: &ResultArguments) -> Result<()> {
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
    write_stdout(&render(results, format, style, &working_directory))
}
