# Changelog

## 0.2.1

### Added

- Persistence round-trip tests: data and graph links survive close/reopen
- Filter tests: parametrized across str, int, float, bool; unsupported types handled gracefully
- Circular graph link test: A->B->C->A traversal at depth 5 returns finite results
- Self-link test: node linking to itself handled without crash
- Embedding dimension mismatch test: `ValueError` on wrong dimensions
- Delete and rebuild test: similarity search correct after partial deletion
- Large batch test: 500 documents added and verified

### Fixed

- Persistent store reopen: existing Document nodes now detected on constructor, vector index rebuilt before first search
- `langchain-graph-retriever` optional floor bumped from `>=0.3` to `>=0.8` to match dev pin

### Changed

- README: added Persistence section with `db_path` usage and reopen example
- README: documented exact-match filter semantics and supported types
- README: added `[retriever]` extra note in quickstart
- README: added `__graph_links__` metadata key format reference table
- 79 tests passing, 98% coverage

## 0.2.0

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

### Fixed

- Graph link ordering: links to target documents now resolve correctly regardless
  of insertion order within the same `add_texts` batch (two-pass approach).

## 0.1.1

- Initial release with `GrafeoGraphStore` and `GrafeoGraphVectorStore`.
