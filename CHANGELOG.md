# Changelog

All notable changes to grafeo-langchain, LangChain graph and vector stores backed by the Grafeo embedded graph database.

## [0.3.0] - 2026-11-16

Storage and question-answering release: grafeo-langchain runs on grafeo 0.6 and its single-file database format, `GrafeoQAChain` answers questions over a knowledge graph, and the standard LangChain retrievers work. Before upgrading: a database written by grafeo 0.5.x is migrated on first open and grafeo 0.5.x cannot open it afterwards (back it up first, see "Upgrading from grafeo 0.5" in the README), and `query()` results, the `get_schema` text and search scores change (see Changed). A store that stays on grafeo 0.5 pins both `grafeo-langchain<0.3` and `grafeo<0.6`.

### Added

- **Questions are answered over the graph**: `GrafeoQAChain.from_llm(llm, graph=store)` has an LLM write a GQL query from the schema, runs it and answers from the rows, with only `langchain-core` and the input and output keys of `GraphCypherQAChain`. Generated queries run read-only (a write is rolled back and raises `ValueError`), and a failing query goes back to the LLM with grafeo's error once (`max_retries`).
- **Questions work on LLM-extracted graphs**: the query prompt matches names by a distinctive word, leaves node labels out and lists alternative relationship types, a query that matches nothing gets one looser try (`retry_on_empty`), and a reply that is not a query raises `ValueError` instead of a grafeo syntax error. With Mistral Small 4 on three short documents, extraction plus questions went from 2 to 11 right answers of 15.
- **Questions work on graphs the store did not write**: `GrafeoQAChain` names entities by `node_id` when the graph has it and by `name` otherwise, as in the `grafeo-datasets` Silicon Valley graph; `id_property` sets another.
- **The README's extraction recipe gives a consistent graph**: fixed entity and relation types, one name per entity across documents, and amounts as properties; Mistral (`langchain-mistralai`) is the example model.
- **Queries can run read-only**: `GrafeoGraphStore.query(..., read_only=True)` rolls back every write and raises `ValueError` if the query wrote anything.
- **The structured schema has the langchain-neo4j keys**: `get_structured_schema` adds `node_props` and `rel_props` (properties with their types) and `relationships` (start label, type, end label).
- **The standard retrievers work**: `similarity_search_with_score` (cosine distance), `max_marginal_relevance_search` (Grafeo's native MMR) and `get_by_ids` are implemented, so `as_retriever(search_type="mmr")` and `as_retriever(search_type="similarity_score_threshold")` work. They raised `NotImplementedError` before.
- **Search results carry `_depth`**: `metadata["_depth"]` is `0` for a search hit and the number of hops for a linked document, as in langchain-graph-retriever. The key is reserved, so it still tells hits from linked documents when a loader's own `source` metadata replaces the `"vector"` / `"graph_traversal"` marker.
- **Decimal metadata is kept**: `decimal.Decimal` values are stored exactly, returned by both stores and by `GrafeoAdapter`, and work as filters; `Decimal("1.5")` matches a stored `Decimal("1.50")`, not the float `1.5`. They were dropped before.
- **Documents carry their id**: returned documents set `Document.id`, besides `metadata["id"]`.
- **Python 3.12 to 3.14 are declared**: the package classifiers list them.
- **A QA chain example runs without an API key**: `examples/graph_qa_chain.py`.
- **The examples follow Pied Piper**: the README and the four `examples/*.py` scripts use the Silicon Valley story and the `grafeo-datasets` graph, and were run with Mistral Small 4 and `mistral-embed`. The scripts also run without an API key, with an offline stand-in for the model.

### Changed

The first six entries change behavior on upgrade and may need action.

- **grafeo 0.6 is required**: `grafeo>=0.6.2,<0.7` (was `>=0.5`); the first open migrates a grafeo 0.5.x database in place and keeps the old files as `<path>.pre-0.6`. grafeo 0.5.x cannot open the migrated database, so stop its processes and back it up first (see "Upgrading from grafeo 0.5" in the README).
- **Every `db_path` is a single file**: a path without the `.grafeo` extension is a file too, where grafeo 0.5 made a WAL directory. An existing WAL directory becomes a file at the same path on first open.
- **A directory as `db_path` raises**: an existing directory, an empty one included, raises `grafeo.GrafeoError`; pass a file path inside it.
- **`query()` follows grafeo 0.6 GQL**: a number with a point (`9.99`) is an exact `decimal.Decimal`, overflow and division by zero raise `grafeo.GrafeoError` instead of returning null, `ORDER BY ... DESC` puts nulls last, and a `WHERE` right after `INSERT`, `CREATE`, `MERGE` or `DELETE` is an error. Queries that relied on the old behavior need changing.
- **`get_schema` describes properties and patterns**: it lists each label and relationship type with its properties and their types, then patterns such as `(:Person)-[:WORKS_AT]->(:Company)`, merged into label unions where only a label differs and with one `()-[:MENTIONED_IN]->(:SourceDocument)` line, instead of three name lists. Code that parses the old text needs changing.
- **Search scores run from 0 to 1**: `metadata["score"]` is `1 - distance / 2` (1 identical, 0.5 unrelated, 0 opposite), in both stores and `GrafeoAdapter`; it was `1 - distance`, which went below 0. Score thresholds need adjusting.
- **A batch is written whole or not at all**: `add_texts` stores every document of a call or none, and a replaced document loses the metadata keys it no longer has.
- **grafeo keeps the vector index in sync**: writes and deletes no longer rebuild it.
- **The README covers the 0.3 behavior**: single-file databases, the upgrade from grafeo 0.5, `GrafeoQAChain`, LLM extraction with structured output, result metadata, score ranges and type-strict filters.
- **The sdist holds the package, tests, examples and docs**: plus the `.gitignore` that hatchling always adds.
- **The retriever example names its strategy**: `examples/graph_retriever_demo.py` passes `Eager` explicitly.
- **CI actions are current**: `actions/checkout` v7, `astral-sh/setup-uv` v10, `codecov/codecov-action` v7, and the ruff pre-commit hook v0.17.1.
- **Development dependencies are a dependency group**: they moved from `[tool.uv] dev-dependencies` to `[dependency-groups]`.

### Fixed

- **Deleting a linked document works**: `GrafeoGraphVectorStore.delete` removes a document with its links in one `DETACH DELETE`; it failed with "Cannot delete node with N connected edge(s)".
- **An empty store returns no results**: searching it raised "No vector index found". The vector index is now created when the store opens.
- **Wrong vector sizes raise `ValueError`**: a search vector of the wrong size crashed the process with a Rust panic, and an embedding model whose vectors differ from its probe stored unusable documents. Both now fail before anything is written or searched.
- **Reopening with another embedding size raises**: a database opened with an embedding model of another dimensionality raises `ValueError` and stays closed, instead of failing later with "No vector index found".
- **Documents stay findable after deletes**: a delete or a replaced embedding could drop documents out of every vector search (grafeo #600). grafeo 0.6 fixes it.
- **Adding a stored id replaces the document**: `add_texts` created a second document with the same id; it now replaces the stored one and keeps its graph links. A repeated id within one batch keeps the last text.
- **Generated ids are unique**: after deletes and a reopen they could repeat a stored id; they now skip ids in use.
- **`add_documents` assigns ids**: a document without `Document.id` gets a generated id instead of none.
- **The factory methods take ids**: `from_texts(..., ids=...)` raised `TypeError`, and `from_documents` now uses `Document.id`.
- **Reserved metadata keys are ignored**: `text`, `doc_id`, `embedding` and `_depth` in metadata no longer overwrite the store's own values.
- **Re-ingestion does not duplicate edges**: graph links, `GrafeoGraphStore` relationships and `MENTIONED_IN` edges are merged by source, target and type, with their properties updated.
- **`SourceDocument` ids are stable**: they use the `id` metadata, `Document.id` or an MD5 hash of the full text, as langchain-neo4j does, instead of Python's per-process `hash()` of the first 200 characters.
- **Traversal returns the closest documents first**: `traversal_search` and `mmr_traversal_search` order linked documents by distance before cutting the result to `2 * k`, so a two-hop neighbor no longer pushes out a direct one.
- **Traversal filters apply to linked documents**: the `filter` of both traversal searches now applies to them too, as documented, not only to the seeds.
- **The schema is current and stable**: a `query()` that writes clears the cached schema, labels and types with nothing left are dropped, and the lists are sorted.
- **`depth` must be an integer**: both traversal searches reject another type instead of formatting it into the query text.

### Dependencies

- **grafeo 0.6.2** (requires `>=0.6.2,<0.7`; was `>=0.5`).
- **grafeo-datasets 0.6.3** (development group only, for the examples).
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
