//! SQLite persistence: files, chunks with their embeddings, and an FTS5 keyword index.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

use crate::chunking::Chunk;

/// Bumped whenever the schema changes incompatibly.
const SCHEMA_VERSION: u32 = 1;

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS metadata (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS files (
        id                    INTEGER PRIMARY KEY,
        path                  TEXT NOT NULL UNIQUE,
        size                  INTEGER NOT NULL,
        modified_nanoseconds  INTEGER NOT NULL,
        content_hash          BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id          INTEGER PRIMARY KEY,
        file_id     INTEGER NOT NULL REFERENCES files(id),
        start_line  INTEGER NOT NULL,
        end_line    INTEGER NOT NULL,
        text        TEXT NOT NULL,
        embedding   BLOB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_id);

    -- External-content FTS5 index over chunks.text; '_' is a token character so snake_case
    -- identifiers match as whole words.
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text, content='chunks', content_rowid='id', tokenize=\"unicode61 tokenchars '_'\"
    );
    CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_after_delete AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;
";

/// What produced the vectors and chunks in an index. Mixing identities would make similarity
/// scores meaningless, so opening an index built differently is an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexIdentity {
    pub encoder: String,
    pub dimensions: usize,
    pub chunker_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRecord {
    pub size: u64,
    pub modified_nanoseconds: i64,
    pub content_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecord {
    pub path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeStatistics {
    pub files: usize,
    pub chunks: usize,
}

/// Restricts queries to one file or everything under one directory.
#[derive(Debug, Clone)]
pub struct PathScope {
    exact: String,
    /// `exact` plus a trailing separator, so `C:\work` does not match `C:\workshop`.
    directory_prefix: String,
    /// SQLite's substr counts characters, not bytes.
    directory_prefix_characters: i64,
}

impl PathScope {
    pub fn new(root: &Path) -> Self {
        let exact = path_key(root);
        let directory_prefix = if exact.ends_with(std::path::MAIN_SEPARATOR) {
            exact.clone()
        } else {
            format!("{exact}{}", std::path::MAIN_SEPARATOR)
        };
        let directory_prefix_characters = directory_prefix.chars().count() as i64;
        Self { exact, directory_prefix, directory_prefix_characters }
    }

    /// SQL condition on `files.path` (aliased `f`) using parameters `:scope_exact`,
    /// `:scope_prefix` and `:scope_prefix_length`.
    const CONDITION: &str = "(f.path = :scope_exact OR substr(f.path, 1, :scope_prefix_length) = :scope_prefix)";

    fn parameters(&self) -> [(&'static str, &dyn rusqlite::ToSql); 3] {
        [
            (":scope_exact", &self.exact),
            (":scope_prefix", &self.directory_prefix),
            (":scope_prefix_length", &self.directory_prefix_characters),
        ]
    }
}

pub struct IndexStore {
    connection: Connection,
    identity: IndexIdentity,
}

impl IndexStore {
    pub fn open(path: &Path, identity: IndexIdentity) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let connection = Connection::open(path).with_context(|| format!("failed to open index {}", path.display()))?;
        Self::initialize(connection, identity)
    }

    pub fn open_in_memory(identity: IndexIdentity) -> Result<Self> {
        Self::initialize(Connection::open_in_memory()?, identity)
    }

    fn initialize(connection: Connection, identity: IndexIdentity) -> Result<Self> {
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.execute_batch(SCHEMA).context("failed to create index schema")?;
        let store = Self { connection, identity };
        store.verify_identity()?;
        Ok(store)
    }

    fn expected_metadata(&self) -> [(&'static str, String); 4] {
        [
            ("schema_version", SCHEMA_VERSION.to_string()),
            ("encoder", self.identity.encoder.clone()),
            ("dimensions", self.identity.dimensions.to_string()),
            ("chunker_version", self.identity.chunker_version.to_string()),
        ]
    }

    fn verify_identity(&self) -> Result<()> {
        for (key, expected) in self.expected_metadata() {
            let stored: Option<String> = self
                .connection
                .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| row.get(0))
                .optional()?;
            match stored {
                None => {
                    self.connection.execute("INSERT INTO metadata(key, value) VALUES (?1, ?2)", [key, &expected])?;
                }
                Some(stored) if stored == expected => {}
                Some(stored) => bail!(
                    "the index was built with {key} = {stored}, but this version of sems uses {expected}; \
                     run `sems index --rebuild` to recreate it"
                ),
            }
        }
        Ok(())
    }

    /// Deletes every file and chunk, keeping the schema. Used by `--rebuild`.
    pub fn clear(&mut self) -> Result<()> {
        let transaction = self.connection.transaction()?;
        transaction.execute_batch(
            "DELETE FROM chunks; DELETE FROM files; DELETE FROM metadata;
             INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');",
        )?;
        transaction.commit()?;
        self.verify_identity()
    }

    pub fn files_in_scope(&self, scope: &PathScope) -> Result<HashMap<PathBuf, FileRecord>> {
        let sql = format!(
            "SELECT f.path, f.size, f.modified_nanoseconds, f.content_hash FROM files f WHERE {}",
            PathScope::CONDITION
        );
        let mut statement = self.connection.prepare(&sql)?;
        let rows = statement.query_map(scope.parameters().as_slice(), |row| {
            Ok((
                PathBuf::from(row.get::<_, String>(0)?),
                FileRecord {
                    size: row.get::<_, i64>(1)? as u64,
                    modified_nanoseconds: row.get(2)?,
                    content_hash: row.get(3)?,
                },
            ))
        })?;
        rows.collect::<rusqlite::Result<_>>().context("failed to list indexed files")
    }

    /// Replaces a file's chunks atomically (inserting the file if it is new).
    pub fn replace_file(&mut self, path: &Path, record: &FileRecord, chunks: &[(Chunk, Vec<f32>)]) -> Result<()> {
        for (_, embedding) in chunks {
            ensure!(
                embedding.len() == self.identity.dimensions,
                "embedding has {} dimensions, index expects {}",
                embedding.len(),
                self.identity.dimensions
            );
        }
        let transaction = self.connection.transaction()?;
        delete_chunks_of(&transaction, path)?;
        let file_id: i64 = transaction.query_row(
            "INSERT INTO files(path, size, modified_nanoseconds, content_hash) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET size = excluded.size,
                 modified_nanoseconds = excluded.modified_nanoseconds, content_hash = excluded.content_hash
             RETURNING id",
            params![path_key(path), record.size as i64, record.modified_nanoseconds, record.content_hash],
            |row| row.get(0),
        )?;
        {
            let mut insert = transaction.prepare(
                "INSERT INTO chunks(file_id, start_line, end_line, text, embedding) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (chunk, embedding) in chunks {
                insert.execute(params![
                    file_id,
                    chunk.start_line as i64,
                    chunk.end_line as i64,
                    chunk.text,
                    encode_vector(embedding)
                ])?;
            }
        }
        transaction.commit().with_context(|| format!("failed to store {}", path.display()))
    }

    /// Records new size/mtime for a file whose content hash did not change.
    pub fn update_file_metadata(&self, path: &Path, record: &FileRecord) -> Result<()> {
        self.connection.execute(
            "UPDATE files SET size = ?2, modified_nanoseconds = ?3 WHERE path = ?1",
            params![path_key(path), record.size as i64, record.modified_nanoseconds],
        )?;
        Ok(())
    }

    pub fn remove_file(&mut self, path: &Path) -> Result<()> {
        let transaction = self.connection.transaction()?;
        delete_chunks_of(&transaction, path)?;
        transaction.execute("DELETE FROM files WHERE path = ?1", [path_key(path)])?;
        transaction.commit().with_context(|| format!("failed to remove {} from the index", path.display()))
    }

    /// Exact nearest neighbours by cosine similarity (vectors are unit length, so a dot product).
    pub fn nearest_chunks(&self, scope: &PathScope, query: &[f32], limit: usize) -> Result<Vec<(i64, f32)>> {
        ensure!(query.len() == self.identity.dimensions, "query vector has the wrong dimensions");
        let sql = format!(
            "SELECT c.id, c.embedding FROM chunks c JOIN files f ON f.id = c.file_id WHERE {}",
            PathScope::CONDITION
        );
        let mut statement = self.connection.prepare(&sql)?;
        let mut rows = statement.query(scope.parameters().as_slice())?;
        let mut best = TopK::new(limit);
        while let Some(row) = rows.next()? {
            let embedding = row.get_ref(1)?.as_blob()?;
            best.offer(row.get(0)?, dot_product_with_encoded(query, embedding)?);
        }
        Ok(best.into_sorted())
    }

    /// Best keyword matches by BM25. Higher scores are better.
    pub fn keyword_chunks(&self, scope: &PathScope, fts_query: &str, limit: usize) -> Result<Vec<(i64, f32)>> {
        let sql = format!(
            "SELECT c.id, bm25(chunks_fts) AS rank
             FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN files f ON f.id = c.file_id
             WHERE chunks_fts MATCH :query AND {}
             ORDER BY rank LIMIT :limit",
            PathScope::CONDITION
        );
        let limit = limit as i64;
        let mut parameters: Vec<(&str, &dyn rusqlite::ToSql)> = vec![(":query", &fts_query), (":limit", &limit)];
        parameters.extend(scope.parameters());
        let mut statement = self.connection.prepare(&sql)?;
        // SQLite's bm25() is "lower is better"; negate so callers can treat all scores alike.
        let rows =
            statement.query_map(parameters.as_slice(), |row| Ok((row.get(0)?, -row.get::<_, f64>(1)? as f32)))?;
        rows.collect::<rusqlite::Result<_>>().context("keyword search failed")
    }

    pub fn chunk(&self, chunk_id: i64) -> Result<ChunkRecord> {
        let (path, start_line, end_line, text, embedding): (String, i64, i64, String, Vec<u8>) = self
            .connection
            .query_row(
                "SELECT f.path, c.start_line, c.end_line, c.text, c.embedding
                 FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id = ?1",
                [chunk_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .with_context(|| format!("chunk {chunk_id} not found"))?;
        Ok(ChunkRecord {
            path: PathBuf::from(path),
            start_line: start_line as usize,
            end_line: end_line as usize,
            text,
            embedding: decode_vector(&embedding)?,
        })
    }

    pub fn statistics(&self, scope: &PathScope) -> Result<ScopeStatistics> {
        let sql = format!(
            "SELECT COUNT(DISTINCT f.id), COUNT(c.id) FROM files f LEFT JOIN chunks c ON c.file_id = f.id WHERE {}",
            PathScope::CONDITION
        );
        let (files, chunks): (i64, i64) =
            self.connection.query_row(&sql, scope.parameters().as_slice(), |row| Ok((row.get(0)?, row.get(1)?)))?;
        Ok(ScopeStatistics { files: files as usize, chunks: chunks as usize })
    }
}

fn delete_chunks_of(transaction: &Transaction<'_>, path: &Path) -> Result<()> {
    transaction
        .execute("DELETE FROM chunks WHERE file_id = (SELECT id FROM files WHERE path = ?1)", [path_key(path)])?;
    Ok(())
}

/// The string form paths are stored and compared in.
fn path_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn encode_vector(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn decode_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    let (values, remainder) = bytes.as_chunks::<4>();
    ensure!(remainder.is_empty(), "corrupt embedding blob of {} bytes", bytes.len());
    Ok(values.iter().map(|&value| f32::from_le_bytes(value)).collect())
}

fn dot_product_with_encoded(query: &[f32], encoded: &[u8]) -> Result<f32> {
    let (values, remainder) = encoded.as_chunks::<4>();
    ensure!(remainder.is_empty() && values.len() == query.len(), "stored embedding has the wrong dimensions");
    Ok(query.iter().zip(values).map(|(left, &right)| left * f32::from_le_bytes(right)).sum())
}

/// Keeps the `capacity` highest-scoring ids without sorting everything.
struct TopK {
    capacity: usize,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<ScoredId>>,
}

#[derive(PartialEq)]
struct ScoredId(f32, i64);

impl Eq for ScoredId {}
impl PartialOrd for ScoredId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl TopK {
    fn new(capacity: usize) -> Self {
        Self { capacity, heap: std::collections::BinaryHeap::with_capacity(capacity + 1) }
    }

    fn offer(&mut self, id: i64, score: f32) {
        self.heap.push(std::cmp::Reverse(ScoredId(score, id)));
        if self.heap.len() > self.capacity {
            self.heap.pop();
        }
    }

    fn into_sorted(self) -> Vec<(i64, f32)> {
        let mut items: Vec<(i64, f32)> =
            self.heap.into_iter().map(|std::cmp::Reverse(ScoredId(score, id))| (id, score)).collect();
        items.sort_by(|left, right| right.1.total_cmp(&left.1));
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> IndexIdentity {
        IndexIdentity { encoder: "test-encoder".into(), dimensions: 2, chunker_version: 1 }
    }

    fn record(seed: u8) -> FileRecord {
        FileRecord { size: 10, modified_nanoseconds: 1, content_hash: [seed; 32] }
    }

    fn chunk(text: &str, embedding: [f32; 2]) -> (Chunk, Vec<f32>) {
        (Chunk { start_line: 1, end_line: 1, text: text.into() }, embedding.to_vec())
    }

    fn root() -> PathBuf {
        std::env::temp_dir().join("sems-store-test")
    }

    #[test]
    fn replacing_a_file_swaps_its_chunks_and_keyword_entries() {
        let mut store = IndexStore::open_in_memory(identity()).unwrap();
        let path = root().join("notes.txt");
        store.replace_file(&path, &record(1), &[chunk("old_term", [1.0, 0.0])]).unwrap();
        store.replace_file(&path, &record(2), &[chunk("new_term", [0.0, 1.0])]).unwrap();

        let scope = PathScope::new(&root());
        assert_eq!(store.statistics(&scope).unwrap(), ScopeStatistics { files: 1, chunks: 1 });
        assert!(store.keyword_chunks(&scope, "\"old_term\"", 10).unwrap().is_empty());
        assert_eq!(store.keyword_chunks(&scope, "\"new_term\"", 10).unwrap().len(), 1);
        assert_eq!(store.files_in_scope(&scope).unwrap()[&path].content_hash, [2; 32]);
    }

    #[test]
    fn nearest_chunks_ranks_by_similarity_within_scope() {
        let mut store = IndexStore::open_in_memory(identity()).unwrap();
        let inside = root().join("project");
        store.replace_file(&inside.join("a.txt"), &record(1), &[chunk("a", [1.0, 0.0])]).unwrap();
        store.replace_file(&inside.join("b.txt"), &record(2), &[chunk("b", [0.6, 0.8])]).unwrap();
        store.replace_file(&root().join("project-other/c.txt"), &record(3), &[chunk("c", [1.0, 0.0])]).unwrap();

        let results = store.nearest_chunks(&PathScope::new(&inside), &[1.0, 0.0], 10).unwrap();
        let paths: Vec<PathBuf> = results.iter().map(|(id, _)| store.chunk(*id).unwrap().path).collect();
        assert_eq!(paths, [inside.join("a.txt"), inside.join("b.txt")]);
        assert!((results[1].1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn scope_can_be_a_single_file() {
        let mut store = IndexStore::open_in_memory(identity()).unwrap();
        let path = root().join("one.txt");
        store.replace_file(&path, &record(1), &[chunk("x", [1.0, 0.0])]).unwrap();
        store.replace_file(&root().join("one.txt.bak"), &record(2), &[chunk("x", [1.0, 0.0])]).unwrap();
        assert_eq!(store.statistics(&PathScope::new(&path)).unwrap().files, 1);
    }

    #[test]
    fn removing_a_file_removes_its_chunks() {
        let mut store = IndexStore::open_in_memory(identity()).unwrap();
        let path = root().join("gone.txt");
        store.replace_file(&path, &record(1), &[chunk("vanishing", [1.0, 0.0])]).unwrap();
        store.remove_file(&path).unwrap();
        let scope = PathScope::new(&root());
        assert_eq!(store.statistics(&scope).unwrap(), ScopeStatistics { files: 0, chunks: 0 });
        assert!(store.keyword_chunks(&scope, "\"vanishing\"", 10).unwrap().is_empty());
    }

    #[test]
    fn opening_with_a_different_identity_fails() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index.db");
        drop(IndexStore::open(&path, identity()).unwrap());
        let other = IndexIdentity { dimensions: 3, ..identity() };
        let error = IndexStore::open(&path, other).err().expect("identity mismatch must fail").to_string();
        assert!(error.contains("--rebuild"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_embeddings_of_the_wrong_size() {
        let mut store = IndexStore::open_in_memory(identity()).unwrap();
        let bad = (Chunk { start_line: 1, end_line: 1, text: "x".into() }, vec![1.0, 0.0, 0.0]);
        assert!(store.replace_file(&root().join("bad.txt"), &record(1), &[bad]).is_err());
    }
}
