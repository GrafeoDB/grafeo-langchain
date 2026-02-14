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

store = GrafeoGraphStore(db_path="./knowledge.db")
store.add_graph_documents(graph_documents, include_source=True)

results = store.query("MATCH (p:Person)-[:WORKS_AT]->(c) RETURN p.node_id, c.node_id")
print(store.get_schema)
```

### Vector + Graph Retrieval (GraphVectorStore)

Combine vector similarity search with graph traversal for Graph RAG:

```python
from langchain_openai import OpenAIEmbeddings
from grafeo_langchain import GrafeoGraphVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = GrafeoGraphVectorStore(
    embedding=embeddings,
    db_path="./doc_graph.db",
    embedding_dimensions=1536,
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
```

## Why Grafeo?

| Feature | Neo4j | Grafeo |
| --- | --- | --- |
| Requires server | Yes (Docker/Cloud) | **No** (embedded, pip install) |
| GraphStore | Yes | **Yes** |
| GraphVectorStore | Community package | **Built-in** (native HNSW) |
| Query language | Cypher | **GQL + Cypher + Gremlin** |
| Graph algorithms | GDS plugin ($$$) | **Built-in** (PageRank, Louvain, ...) |
| Deployment | Docker container | **Single .db file** |
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

- `GrafeoGraphVectorStore(embedding, db_path=None, embedding_dimensions=1536)`: vector store with graph links
- `.add_texts(texts, metadatas=None, ids=None)`: add documents with embeddings and optional graph links
- `.similarity_search(query, k=4)`: standard vector similarity search
- `.traversal_search(query, k=4, depth=1)`: vector search + graph traversal
- `.mmr_traversal_search(query, k=4, depth=2, fetch_k=100, lambda_mult=0.5)`: MMR-diversified traversal
- `.from_texts(...)` / `.from_documents(...)`: factory methods

## Requirements

- Python 3.12+

## License

Apache-2.0
