"""GraphVectorStore with traversal, filtering, and deletion.

Demonstrates:
- Auto-detected embedding dimensions (no manual config needed)
- Adding documents with embeddings and graph links
- Similarity search (pure vector)
- Traversal search (vector seeds + graph walk)
- MMR traversal search (diversified + graph walk)
- Metadata filtering
- Document deletion

Uses a simple hash-based embedding so no API key is needed.

Run:
    uv run python examples/graph_vector_rag.py
"""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings

from grafeo_langchain import GrafeoGraphVectorStore

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


# ── Create store and add documents with graph links ────────────────────────────

embeddings = DemoEmbeddings()
store = GrafeoGraphVectorStore(embedding=embeddings)  # dimensions auto-detected

store.add_texts(
    texts=[
        "Python is a high-level programming language created by Guido van Rossum.",
        "Guido van Rossum started developing Python in the late 1980s.",
        "ABC was a programming language that heavily influenced Python's design.",
        "Rust is a systems programming language focused on safety and performance.",
        "TypeScript is a typed superset of JavaScript developed by Microsoft.",
    ],
    metadatas=[
        {"__graph_links__": [{"target_id": "abc", "type": "INFLUENCED_BY"}]},
        {},
        {"__graph_links__": [{"target_id": "python", "type": "INFLUENCED"}]},
        {},
        {},
    ],
    ids=["python", "guido", "abc", "rust", "typescript"],
)

# ── Similarity search (pure vector) ───────────────────────────────────────────

print("=== Similarity Search ===")
docs = store.similarity_search("programming languages", k=3)
for doc in docs:
    print(f"  [{doc.metadata.get('id', '?')}] {doc.page_content[:80]}...")
print()

# ── Traversal search (vector + graph links) ───────────────────────────────────

print("=== Traversal Search (depth=1) ===")
docs = store.traversal_search("Python programming", k=2, depth=1)
for doc in docs:
    source = doc.metadata.get("source", "vector")
    print(f"  [{source}] {doc.page_content[:80]}...")
print()

# ── MMR traversal search (diversified + graph) ────────────────────────────────

print("=== MMR Traversal Search ===")
docs = store.mmr_traversal_search("programming", k=3, depth=1, lambda_mult=0.5)
for doc in docs:
    score = doc.metadata.get("score", "n/a")
    print(f"  [score={score}] {doc.page_content[:80]}...")

# ── Filtered search ──────────────────────────────────────────────────────────

print()
print("=== Filtered Search ===")
store2 = GrafeoGraphVectorStore(embedding=embeddings)
store2.add_texts(
    ["Python is a scripting language", "Rust is a systems language", "Go is compiled"],
    metadatas=[{"category": "scripting"}, {"category": "systems"}, {"category": "systems"}],
    ids=["py", "rs", "go"],
)
docs = store2.similarity_search("language", k=3, filter={"category": "systems"})
for doc in docs:
    print(f"  [{doc.metadata.get('id')}] {doc.page_content}")

# ── Delete ───────────────────────────────────────────────────────────────────

print()
print("=== Delete ===")
print(f"  Before delete: {len(store2.similarity_search('language', k=10))} docs")
store2.delete(["py"])
print(f"  After deleting 'py': {len(store2.similarity_search('language', k=10))} docs")

store.close()
store2.close()
