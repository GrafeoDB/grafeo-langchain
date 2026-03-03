"""Graph Retriever adapter demo with Eager strategy.

Demonstrates using GrafeoAdapter with langchain-graph-retriever
for advanced graph traversal strategies via metadata edges.

Requires:
    uv add grafeo-langchain[retriever]

Run:
    uv run python examples/graph_retriever_demo.py
"""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings

from grafeo_langchain import GrafeoGraphVectorStore
from grafeo_langchain.adapter import GrafeoAdapter

DIMS = 32


class DemoEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for demo purposes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:DIMS]]


# ── Create store and add documents ───────────────────────────────────────────

embeddings = DemoEmbeddings()
store = GrafeoGraphVectorStore(embedding=embeddings)

store.add_texts(
    texts=[
        "Python is a high-level programming language.",
        "Guido van Rossum created Python.",
        "ABC influenced Python's design.",
        "Rust is a systems programming language.",
        "TypeScript is a typed superset of JavaScript.",
    ],
    metadatas=[
        {"topic": "python"},
        {"topic": "python"},
        {"topic": "python"},
        {"topic": "rust"},
        {"topic": "typescript"},
    ],
    ids=["python", "guido", "abc", "rust", "typescript"],
)

# ── Create adapter and retriever ─────────────────────────────────────────────

from langchain_graph_retriever import GraphRetriever  # noqa: E402

adapter = GrafeoAdapter(vector_store=store)

retriever = GraphRetriever(
    store=adapter,
    edges=[("topic", "topic")],
    k=5,
    start_k=2,
)

# ── Invoke the retriever ─────────────────────────────────────────────────────

print("=== Eager Strategy (topic edges) ===")
docs = retriever.invoke("programming languages")
for doc in docs:
    print(f"  [{doc.metadata.get('id', '?')}] {doc.page_content[:80]}")
print()

# ── With metadata filter ─────────────────────────────────────────────────────

print("=== Filtered (topic=python) ===")
docs = retriever.invoke("programming", filter={"topic": "python"})
for doc in docs:
    print(f"  [{doc.metadata.get('id', '?')}] {doc.page_content[:80]}")

store.close()
