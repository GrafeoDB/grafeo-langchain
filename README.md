[![CI](https://github.com/GrafeoDB/grafeo-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-langchain/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-langchain/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-langchain)
[![PyPI](https://img.shields.io/pypi/v/grafeo-langchain.svg)](https://pypi.org/project/grafeo-langchain/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

# grafeo-langchain

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

Pass a `db_path` ending in `.grafeo` to keep everything in a single file. Close the store, reopen it later, and your documents, embeddings, and graph links are all still there:

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

A `db_path` without the `.grafeo` extension (such as `./my_store.db`) creates a directory that
holds a write-ahead log instead of a single file. That storage is deprecated as of grafeo 0.6, so
prefer `.grafeo` paths for new stores.

Reopening a store with an embedding model of another dimensionality raises `ValueError`.

#### Upgrading to grafeo 0.6

grafeo 0.6 changes the database file format. The first time 0.6 opens a database written by grafeo
0.5.x for writing, it migrates the database and keeps the old files next to it (for example
`my_store.grafeo.pre-0.6`). After that, grafeo 0.5.x cannot open the database, so stop every process
that still uses 0.5.x first, and keep a backup if you may need to go back. See
[Upgrading from 0.5](https://grafeo.dev/user-guide/persistence/persistent/#upgrading-from-05).

### Graph Retriever Integration

> **Note:** The `[retriever]` extra is required for this feature. Install with
> `uv add "grafeo-langchain[retriever]"` (requires `langchain-graph-retriever>=0.8`).

Use `GrafeoAdapter` with [langchain-graph-retriever](https://github.com/datastax/langchain-graph-retriever)
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

Supported value types: `str`, `int`, `float`, `bool`. Compound types (lists, dicts) are not supported as filter values.

A value matches only a stored value of the same type: `{"year": 2024}` does not match `2024.0`, and
`{"active": 1}` does not match `True`. In `traversal_search` and `mmr_traversal_search` the filter
applies to the linked documents as well as to the seeds.

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

### `GrafeoGraphVectorStore`

- `GrafeoGraphVectorStore(embedding, db_path=None, embedding_dimensions=None)`: vector store with graph links (dimensions auto-detected from the model)
- `.add_texts(texts, metadatas=None, ids=None)`: add documents with embeddings and optional graph links
- `.similarity_search(query, k=4, filter=None)`: standard vector similarity search
- `.similarity_search_by_vector(embedding, k=4, filter=None)`: search by pre-computed vector
- `.traversal_search(query, k=4, depth=1, filter=None)`: vector search + graph traversal
- `.mmr_traversal_search(query, k=4, depth=2, fetch_k=100, lambda_mult=0.5, filter=None)`: MMR-diversified traversal
- `.delete(ids)`: remove documents by ID, with their graph links
- `.from_texts(...)` / `.from_documents(...)`: factory methods

### `GrafeoAdapter`

Requires `uv add grafeo-langchain[retriever]`.

- `GrafeoAdapter(vector_store)`: adapter for `langchain-graph-retriever`
- Works with `GraphRetriever(store=adapter, edges=[...])` for Eager/BFS strategies

## Requirements

- Python 3.12+
- grafeo 0.5.44 or later, below 0.7 (tested against 0.5.44)

## License

Apache-2.0
