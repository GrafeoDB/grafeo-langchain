# Changelog

All notable changes to grafeo-langchain, LangChain graph and vector stores backed by the Grafeo embedded graph database.

## [0.3.0] - 2026-11-14

Storage and correctness release: grafeo-langchain runs on grafeo 0.6 (`grafeo>=0.6.2,<0.7`) and its single-file database format, and repeated ingestion, deletes and vector search are fixed. Before upgrading: a database written by grafeo 0.5.x is migrated on first open and grafeo 0.5.x cannot open it afterwards (back it up first, see "Upgrading from grafeo 0.5" in the README), a `db_path` that is an existing directory raises `grafeo.GrafeoError`, and `GrafeoGraphStore.query` returns results with grafeo 0.6 semantics. A store that stays on grafeo 0.5 stays on grafeo-langchain 0.2 and pins `grafeo<0.6`.

### Added

- `decimal.Decimal` metadata values are stored (grafeo 0.6 keeps them exact, with their scale) and
  returned, by both stores and by `GrafeoAdapter`, and work as filter values: `Decimal("1.5")`
  matches a stored `Decimal("1.50")`, not the float `1.5`. They were dropped before.
- Returned documents carry their id in `Document.id`, besides `metadata["id"]`.
- `tests/test_grafeo_contract.py`: the grafeo behavior this package relies on (replacing upserts,
  vector index sync after writes and deletes, filter semantics, decimals, null ordering,
  `DETACH DELETE`, result counters, traversal ordering, schema shape, single-file persistence), each
  with a message naming the code that depends on it.
- `tests/test_migration.py`: stores written by grafeo-langchain 0.2.0 on grafeo 0.5.44 (a file and
  a WAL directory, in `tests/fixtures/`, made by `tests/fixtures/make_grafeo_0_5_fixtures.py`) open
  with grafeo 0.6 with their documents, vectors and links, and keep the old files as `.pre-0.6`.
- Tests for repeated ingestion, deletes with links, persistence, generated ids, dimension checks,
  traversal order under shuffled row order, decimals in results and metadata, and the `[retriever]`
  extra with Eager traversal.
- Weekly CI workflow (`grafeo-latest.yml`, also run by hand) that tests against the newest grafeo on
  PyPI, pre-releases included, both within the supported range and past it.
- Python classifiers for 3.12 to 3.14. The suite passes on Python 3.15, which CI also runs but
  allows to fail: zstandard, which langchain-core needs through langsmith, has no 3.15 wheels yet
  and installs only where it can be compiled.

### Changed

The first four entries change behavior on upgrade and may need action.

- Requires grafeo 0.6 (`grafeo>=0.6.2,<0.7`; was `>=0.5`). The first time grafeo 0.6 opens a
  store written by grafeo 0.5.x, it migrates the database in place and keeps the old files as
  `<file>.pre-0.6` (a WAL directory as `<path>.pre-0.6/`). grafeo 0.5.x cannot open the migrated
  database: stop every process that uses it with 0.5.x and back it up before upgrading. See
  "Upgrading from grafeo 0.5" in the README.
- A `db_path` without the `.grafeo` extension is a single file too: grafeo 0.5 made a WAL
  directory there. An existing WAL directory becomes a file at the same path on first open.
- A `db_path` that is an existing directory, an empty one included, raises `grafeo.GrafeoError`
  (grafeo 0.5 created a database in it): pass a file path inside it.
- `GrafeoGraphStore.query` runs GQL with grafeo 0.6 semantics: a number with a point (`9.99`) is
  an exact decimal and comes back as `decimal.Decimal`, also once stored as a property; integer and
  decimal overflow and division by zero raise `grafeo.GrafeoError` instead of returning null;
  `ORDER BY ... DESC` puts nulls last; a `WHERE` right after `INSERT`, `CREATE`, `MERGE` or `DELETE`
  is an error. Queries that relied on the old behavior need changing.
- `add_texts` writes each batch with one replacing `upsert_nodes` call: all documents of a batch are
  stored, with the metadata keys a replaced document no longer has removed, or none is.
- The vector index is no longer rebuilt after every write, nor after deletes: grafeo keeps it in
  sync.
- README: single-file databases, upgrading from grafeo 0.5 (migration, backups, `.pre-0.6`), GQL
  results under grafeo 0.6, metadata types, re-ingestion and upsert semantics, type-strict filters.
- CI: `actions/checkout` v7, `astral-sh/setup-uv` v10, `codecov/codecov-action` v7; the ruff
  pre-commit hook is v0.17.1.
- Development dependencies moved from `[tool.uv] dev-dependencies` to `[dependency-groups]`.
- `examples/graph_retriever_demo.py` passes an explicit `Eager` strategy.

### Fixed

- `GrafeoGraphVectorStore.delete` failed with `GrafeoError` ("Cannot delete node with N connected
  edge(s)") for a document with graph links, since grafeo 0.5.44 refuses to delete a node that
  still has edges. Documents are now deleted with their links in one `DETACH DELETE`.
- Searching an empty `GrafeoGraphVectorStore` raised "No vector index found" instead of returning
  no documents. The vector index is now created when the store opens.
- A search vector of the wrong size crashed the process with a Rust panic, and an embedding model
  that returns vectors of another size than its probe stored unusable documents. Both now raise
  `ValueError` before anything is written or searched.
- Vector index errors were swallowed (`contextlib.suppress(RuntimeError)`): reopening a database
  with an embedding model of another dimensionality now raises `ValueError` and leaves the
  database closed, instead of failing later with "No vector index found".
- Documents could drop out of every vector search after a delete or a replaced embedding, because
  grafeo 0.5 removed a vector from its HNSW graph without reconnecting its neighbors (grafeo #600).
  grafeo 0.6 fixes it; the store relies on that and does not rebuild the index.
- `add_texts` with an id that is already stored created a second document with that id (returned
  twice by searches, and linked twice). It now replaces the stored document in place, keeping its
  graph links, and a repeated id within one batch keeps the last text.
- Generated ids could repeat a stored id after deletes and a reopen. They now skip ids in use.
- `add_documents` stored documents without `Document.id` with no id; they now get a generated id.
- `from_texts(..., ids=...)` raised `TypeError`, and `from_documents` ignored `Document.id`.
- The metadata keys `text`, `doc_id` and `embedding` overwrote the document's text, id or vector.
  They are now reserved and ignored.
- Graph links and `GrafeoGraphStore` relationships were duplicated each time the same data was
  ingested again, as were `MENTIONED_IN` edges. Edges are now merged by source, target and type,
  and their properties updated.
- `SourceDocument` ids used Python's `hash()` of the first 200 characters, which changes between
  processes and collides for documents with a common prefix. They now use the `id` metadata,
  `Document.id`, or an MD5 hash of the full text (as `langchain-neo4j` does).
- `traversal_search` and `mmr_traversal_search` returned linked documents in an unspecified order
  before cutting the result to `2 * k`, so a two-hop neighbor could push out a direct one. They now
  return the closest linked documents first, in a stable order.
- The `filter` of `traversal_search` and `mmr_traversal_search` now applies to linked documents,
  as documented, not only to the seeds.
- `GrafeoGraphStore.query` did not clear the cached schema after a write. The schema also listed
  labels and relationship types with no nodes or edges left, and its order changed between calls;
  it is now sorted.
- `traversal_search` and `mmr_traversal_search` reject a `depth` that is not an integer instead of
  formatting it into the query text.

### Dependencies

- **grafeo 0.6.2** (`grafeo>=0.6.2,<0.7`; 0.2.0 required `>=0.5`).
- **Lockfile**: grafeo 0.6.2, langchain-core 1.8.0, langchain 1.5.0 and langchain-graph-retriever 0.8.0; development tools pytest 9.2, ruff 0.17 and ty 0.0.104.

## [0.2.0] - 2026-04-12

Feature release: aligned with grafeo 0.5.37, with new features, test coverage and documentation improvements.

### Added

- Auto-detect embedding dimensions: no more need to specify `embedding_dimensions`
  manually. The store probes the embedding model at init time. When provided, the
  value is validated against the model's actual output.
- `filter` parameter on `similarity_search`, `similarity_search_by_vector`,
  `traversal_search`, and `mmr_traversal_search` for property-based metadata filtering.
- `delete(ids)` method on `GrafeoGraphVectorStore`.
- Consistent metadata: vector results include `"source": "vector"`, traversal results
  include `"score": None`, so downstream code can rely on both keys being present.
- `GrafeoAdapter` for `langchain-graph-retriever` integration (install with
  `uv add grafeo-langchain[retriever]`).
- `py.typed` marker for PEP 561 type checking support.
- `__version__` attribute on the package.
- Persistence round-trip tests: data and graph links survive close/reopen
- Filter tests: parametrized across str, int, float, bool; unsupported types handled gracefully
- Circular graph link test: A->B->C->A traversal at depth 5 returns finite results
- Self-link test: node linking to itself handled without crash
- Embedding dimension mismatch test: `ValueError` on wrong dimensions
- Delete and rebuild test: similarity search correct after partial deletion
- Large batch test: 500 documents added and verified

### Changed

- Requires grafeo >=0.5 (tested against 0.5.37)
- README: added Persistence section with `db_path` usage and reopen example
- README: documented exact-match filter semantics and supported types
- README: added `[retriever]` extra note in quickstart
- README: added `__graph_links__` metadata key format reference table
- 79 tests passing, 98% coverage

### Fixed

- Graph link ordering: links to target documents now resolve correctly regardless
  of insertion order within the same `add_texts` batch (two-pass approach).
- Persistent store reopen: existing Document nodes now detected on constructor, vector index rebuilt before first search
- `langchain-graph-retriever` optional floor bumped from `>=0.3` to `>=0.8` to match dev pin

## [0.1.1] - 2026-02-11

- Initial release with `GrafeoGraphStore` and `GrafeoGraphVectorStore`.

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
