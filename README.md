<picture>
  <source media="(max-width: 556px) and (prefers-color-scheme: dark), (min-width: 768px) and (max-width: 844px) and (prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-narrow-dark.svg">
  <source media="(max-width: 556px), (min-width: 768px) and (max-width: 844px)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-narrow-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-dark.svg">
  <img alt="grafeo-langchain: LangChain graph and vector stores backed by Grafeo, in one file." src="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-light.svg" width="100%">
</picture>

[![CI](https://github.com/GrafeoDB/grafeo-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-langchain/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-langchain/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-langchain)
[![PyPI](https://img.shields.io/pypi/v/grafeo-langchain.svg)](https://pypi.org/project/grafeo-langchain/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

LangChain graph store and vector store backed by [GrafeoDB](https://github.com/GrafeoDB/grafeo): an embedded graph database with native vector search.

No servers, no Docker, no configuration. Just `uv add` and go.

## Install

```bash
uv add grafeo-langchain

# Optional: langchain-graph-retriever integration (requires >=0.8)
uv add "grafeo-langchain[retriever]"
```

## Quick Start

### Knowledge Graph (GraphStore)

Store LLM-extracted triples and query them with GQL/Cypher:

```python
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from grafeo_langchain import GrafeoGraphStore

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
transformer = LLMGraphTransformer(llm=llm)

documents = [
    Document(page_content="Alice works at Microsoft. Bob works at Google. Alice knows Bob."),
]
graph_documents = transformer.convert_to_graph_documents(documents)

store = GrafeoGraphStore(db_path="./knowledge.grafeo")
store.add_graph_documents(graph_documents, include_source=True)

results = store.query("MATCH (p:Person)-[:WORKS_AT]->(c) RETURN p.node_id, c.node_id")
# [{"p.node_id": "Alice", "c.node_id": "Microsoft"}]: a column without an alias is named after its expression
print(store.get_schema)
```

Ingestion is idempotent: nodes are matched by id, and relationships by source, target and type, so
adding the same graph documents again updates their properties instead of duplicating them. With
`include_source=True`, each source document becomes one `SourceDocument` node, identified by its `id`
metadata, its `Document.id`, or an MD5 hash of its text.

A `query()` that writes to the graph clears the cached schema. After writing through `store.client`,
call `store.refresh_schema()`.

`query()` runs GQL (grafeo 0.6). Results to expect:

- A number with a point is an exact decimal: `RETURN 9.99 * 3` gives `Decimal("29.97")`, and a
  property written as `price: 9.99` reads back as a `Decimal`. Write `9.99e0` for a float.
- An integer or decimal overflow and a division by zero (`1 / 0`) raise `grafeo.GrafeoError` instead
  of returning null.
- `ORDER BY` puts nulls last, ascending and descending, unless the query says `NULLS FIRST`.
- A `WHERE` right after `INSERT`, `CREATE`, `MERGE` or `DELETE` is an error: put the condition
  before the write.

### Vector + Graph Retrieval (GraphVectorStore)

Combine vector similarity search with graph traversal for Graph RAG:

```python
from langchain_openai import OpenAIEmbeddings
from grafeo_langchain import GrafeoGraphVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = GrafeoGraphVectorStore(
    embedding=embeddings,
    db_path="./doc_graph.grafeo",
    # embedding_dimensions auto-detected from the model
)

store.add_texts(
    texts=["Python is a programming language...", "Guido van Rossum...", "ABC influenced..."],
    metadatas=[
        {"id": "python", "__graph_links__": [{"target_id": "abc", "type": "INFLUENCED_BY"}]},
        {"id": "guido"},
        {"id": "abc", "__graph_links__": [{"target_id": "python", "type": "INFLUENCED"}]},
    ],
    ids=["python", "guido", "abc"],
)

# Standard vector search
docs = store.similarity_search("What programming languages exist?", k=2)

# Vector search + graph traversal
docs = store.traversal_search("What programming languages exist?", k=4, depth=2)

# MMR-diversified graph traversal
docs = store.mmr_traversal_search("programming history", k=4, depth=2, lambda_mult=0.7)

# Filtered search (only documents with matching metadata)
docs = store.similarity_search("languages", k=4, filter={"category": "systems"})

# Delete documents (and their graph links)
store.delete(["python", "abc"])
```

Adding a text whose id is already stored replaces that document: its text, embedding and metadata
are overwritten, its graph links are kept, and new links are added once. Texts added without an id
get a generated id (`"0"`, `"1"`, ...) that no stored document uses. `add_documents` and
`from_documents` use `Document.id` when it is set. The metadata keys `doc_id`, `text` and
`embedding` are reserved for the store and ignored.

`traversal_search` returns the seed documents first, then the linked documents closest to a seed
first, up to `2 * k` documents in total.

### Persistence

Pass a `db_path` to keep everything in a single file. Close the store, reopen it later, and your documents, embeddings, and graph links are all still there:

```python
from langchain_openai import OpenAIEmbeddings
from grafeo_langchain import GrafeoGraphVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Write phase
store = GrafeoGraphVectorStore(embedding=embeddings, db_path="./my_store.grafeo")
store.add_texts(["Python is great", "Rust is fast"], ids=["py", "rs"])
store.close()

# Later: reopen and query
store = GrafeoGraphVectorStore(embedding=embeddings, db_path="./my_store.grafeo")
docs = store.similarity_search("programming languages", k=2)
store.close()
```

Omit `db_path` (or pass `None`) for a purely in-memory store that is discarded when the process exits.

The database is one file at `db_path`, whatever its extension (`.grafeo` is the usual one). While
the store is open, grafeo keeps its write-ahead log in a `<db_path>.wal/` directory next to it, and
`close()` folds the log into the file. The directory that holds the file must be writable.

`db_path` must not be an existing directory: grafeo 0.6 refuses a directory that is not a grafeo
0.5.x database, an empty one included, and raises `grafeo.GrafeoError`. Use a file path inside it,
such as `./data/my_store.grafeo`.

Reopening a store with an embedding model of another dimensionality raises `ValueError`.

#### Upgrading from grafeo 0.5 (grafeo-langchain 0.2)

grafeo-langchain 0.3 requires grafeo 0.6, which has a new database file format. The first time grafeo
0.6 opens a database written by grafeo 0.5.x for writing (opening a store does), it migrates the
database in place and keeps the old files, unchanged, next to it:

| Written by 0.5.x as | After the first open with 0.6 | Old files kept as |
| --- | --- | --- |
| a file, such as `./my_store.grafeo` | a file at the same path | `./my_store.grafeo.pre-0.6` (and `./my_store.grafeo.pre-0.6.wal` if it had a log) |
| a WAL directory, any path without `.grafeo`, such as `./my_store.db` | a file at the same path | the directory, as `./my_store.db.pre-0.6/` |

grafeo 0.5.x cannot open a migrated database, so:

1. Stop every process that uses the database with grafeo 0.5.x (grafeo-langchain 0.2 or another
   grafeo program).
2. Back it up: copy the file or the WAL directory somewhere else.
3. Upgrade: `uv add "grafeo-langchain>=0.3"`.
4. Open the store once and check your documents, for example with `similarity_search`.
5. Keep the `.pre-0.6` copy until you no longer need to go back. To go back to 0.5.x, delete the
   migrated file and rename the `.pre-0.6` file or directory to its old name.

grafeo 0.7 will no longer read 0.5.x databases, so open them with 0.6 before then. A store that
stays on grafeo-langchain 0.2 must pin `grafeo<0.6` itself: 0.2.0 requires `grafeo>=0.5` without an
upper bound, so a new install of it gets grafeo 0.6. See
[Upgrading from 0.5](https://docs.grafeo.dev/operate/persistence/persistent/#upgrading-from-05).

### Graph Retriever Integration

> **Note:** The `[retriever]` extra is required for this feature. Install with
> `uv add "grafeo-langchain[retriever]"` (requires `langchain-graph-retriever>=0.8`).

Use `GrafeoAdapter` with [langchain-graph-retriever](https://github.com/datastax/graph-rag)
for advanced traversal strategies (Eager, BFS, MMR) via metadata edges:

```python
from grafeo_langchain import GrafeoGraphVectorStore
from grafeo_langchain.adapter import GrafeoAdapter
from langchain_graph_retriever import GraphRetriever

store = GrafeoGraphVectorStore(embedding=embeddings)
store.add_texts(
    texts=["Python is a language", "Rust is a language"],
    metadatas=[{"topic": "python"}, {"topic": "rust"}],
    ids=["py", "rs"],
)

adapter = GrafeoAdapter(vector_store=store)
retriever = GraphRetriever(store=adapter, edges=[("topic", "topic")])
docs = retriever.invoke("programming")
```

## Filters

All filter parameters use **exact-match equality**. Pass a dict where each key is a metadata field name and the value is the expected value. Only documents whose metadata matches every key-value pair are returned:

```python
docs = store.similarity_search("query", k=4, filter={"category": "science", "year": 2024})
```

Supported value types: `str`, `int`, `float`, `bool` and `decimal.Decimal`. Compound types (lists, dicts) are not supported as filter values.

A value matches only a stored value of the same type: `{"year": 2024}` does not match `2024.0`,
`{"active": 1}` does not match `True`, and `{"price": Decimal("1.5")}` matches a stored
`Decimal("1.50")` but not the float `1.5`. In `traversal_search` and `mmr_traversal_search` the
filter applies to the linked documents as well as to the seeds.

## Metadata

The stores keep metadata values of type `str`, `int`, `float`, `bool` and `decimal.Decimal`
(grafeo 0.6 stores a `Decimal` exactly, with its scale) and leave out other values, such as lists,
dicts and `None`. Documents come back with the metadata as stored. langchain-graph-retriever 0.8
follows edges on `str`, `bool`, `int` and `float` values only: it skips a `Decimal` field with a
warning, so use another type for a field that `GrafeoAdapter` traversal should follow.

## Graph Links Format

Graph links between documents are specified via the `__graph_links__` metadata key. Each link is a dict with the following fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `target_id` | `str` | Yes | The `id` of the target document |
| `type` | `str` | No | Edge label (defaults to `LINKS_TO`) |
| `properties` | `dict` | No | Additional properties stored on the edge |

Example:

```python
store.add_texts(
    texts=["Source document", "Target document"],
    metadatas=[
        {
            "__graph_links__": [
                {"target_id": "target", "type": "CITES"},
                {"target_id": "other", "type": "RELATES_TO", "properties": {"weight": 0.9}},
            ]
        },
        {},
    ],
    ids=["source", "target"],
)
```

The `__graph_links__` key is consumed during ingestion and is not stored as document metadata.

Links are created after all documents of the same `add_texts` call are stored, so a link may point to
a document later in the same batch. A link to an id that is not stored yet is skipped: add target
documents first, or in the same batch. Adding the same link again keeps one edge and updates its
properties.

## Why Grafeo?

| Feature | Neo4j | Grafeo |
| --- | --- | --- |
| Requires server | Yes (Docker/Cloud) | **No** (embedded, `uv add`) |
| GraphStore | Yes | **Yes** |
| GraphVectorStore | Community package | **Built-in** (native HNSW) |
| Query language | Cypher | **GQL + Cypher + Gremlin** |
| Graph algorithms | GDS plugin ($$$) | **Built-in** (PageRank, Louvain, ...) |
| Deployment | Docker container | **Single .grafeo file** |
| Offline/edge | No | **Yes** |

## API Reference

### `GrafeoGraphStore`

- `GrafeoGraphStore(db_path=None)`: in-memory or persistent graph store
- `.add_graph_documents(docs, include_source=False)`: ingest LLM-extracted graph documents
- `.query(query, params=None)`: execute GQL/Cypher queries
- `.get_schema` / `.get_structured_schema`: inspect the graph schema
- `.refresh_schema()`: refresh the cached schema
- `.client`: access the underlying `GrafeoDB` instance
- `.close()`: close the database; with a `db_path`, this folds the write-ahead log into the file. The store is also a context manager (`with GrafeoGraphStore(...) as store:`) that closes on exit

### `GrafeoGraphVectorStore`

- `GrafeoGraphVectorStore(embedding, db_path=None, embedding_dimensions=None)`: vector store with graph links (dimensions auto-detected from the model)
- `.add_texts(texts, metadatas=None, ids=None)`: add documents with embeddings and optional graph links
- `.similarity_search(query, k=4, filter=None)`: standard vector similarity search
- `.similarity_search_by_vector(embedding, k=4, filter=None)`: search by pre-computed vector
- `.traversal_search(query, k=4, depth=1, filter=None)`: vector search + graph traversal
- `.mmr_traversal_search(query, k=4, depth=2, fetch_k=100, lambda_mult=0.5, filter=None)`: MMR-diversified traversal
- `.delete(ids)`: remove documents by ID, with their graph links
- `.from_texts(...)` / `.from_documents(...)`: factory methods
- `.close()`: close the database; with a `db_path`, this folds the write-ahead log into the file. The store is also a context manager that closes on exit

### `GrafeoAdapter`

Requires `uv add grafeo-langchain[retriever]`.

- `GrafeoAdapter(vector_store)`: adapter for `langchain-graph-retriever`
- Works with `GraphRetriever(store=adapter, edges=[...])` for Eager/BFS strategies

## Requirements

- Python 3.12+
- grafeo 0.6 (`grafeo>=0.6.2,<0.7`). Databases written by grafeo 0.5.x
  are migrated on first open: see [Upgrading from grafeo 0.5](#upgrading-from-grafeo-05-grafeo-langchain-02).
- langchain-core 1.2 or later
- langchain-graph-retriever 0.8 or later, for `GrafeoAdapter` (the `[retriever]` extra)

## License

Apache-2.0
