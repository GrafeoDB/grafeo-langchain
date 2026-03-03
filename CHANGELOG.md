# Changelog

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
