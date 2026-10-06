# Changelog

## Unreleased

Aligned with grafeo 0.5.44 and prepared for grafeo 0.6.0. Fixes for repeated ingestion, deletes
and vector search, an engine contract test suite and a weekly run against the newest grafeo.

### Added

- Returned documents carry their id in `Document.id`, besides `metadata["id"]`.
- `tests/test_grafeo_contract.py`: the grafeo behavior this package relies on (upserts, vector
  index sync, filter semantics, `DETACH DELETE`, result counters, traversal ordering, schema shape,
  persistence of single-file and WAL-directory databases), each with a message naming the code that
  depends on it.
- Tests for repeated ingestion, deletes with links, persistence in both storage formats, generated
  ids, dimension checks, traversal order under shuffled row order and the `[retriever]` extra with
  Eager traversal.
- Weekly CI workflow (`grafeo-latest.yml`, also run by hand) that tests against the newest grafeo on
  PyPI, pre-releases included, both within the supported range and past it.

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
- Documents could drop out of every vector search after a delete, because grafeo removes a vector
  from its HNSW graph without reconnecting its neighbors. The store rebuilds the index before the
  next search after a delete.
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

### Changed

- Requires `grafeo>=0.5.44,<0.7` (tested against 0.5.44); the lock file resolves grafeo 0.5.44.
- `add_texts` writes each batch with one `upsert_nodes` call: all documents of a batch are stored,
  or none.
- The vector index is no longer rebuilt after every write: grafeo keeps it in sync.
- README: `.grafeo` paths for single-file databases (other paths create a WAL directory, deprecated
  as of grafeo 0.6), upgrading to grafeo 0.6, re-ingestion and upsert semantics, type-strict filters.
- Development dependencies moved from `[tool.uv] dev-dependencies` to `[dependency-groups]`.
- `examples/graph_retriever_demo.py` passes an explicit `Eager` strategy.

## 0.2.0 - 2026-04-12

Aligned with grafeo 0.5.37. New features, test coverage and documentation improvements.

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

### Fixed

- Graph link ordering: links to target documents now resolve correctly regardless
  of insertion order within the same `add_texts` batch (two-pass approach).
- Persistent store reopen: existing Document nodes now detected on constructor, vector index rebuilt before first search
- `langchain-graph-retriever` optional floor bumped from `>=0.3` to `>=0.8` to match dev pin

### Changed

- Requires grafeo >=0.5 (tested against 0.5.37)
- README: added Persistence section with `db_path` usage and reopen example
- README: documented exact-match filter semantics and supported types
- README: added `[retriever]` extra note in quickstart
- README: added `__graph_links__` metadata key format reference table
- 79 tests passing, 98% coverage

## 0.1.1

- Initial release with `GrafeoGraphStore` and `GrafeoGraphVectorStore`.
