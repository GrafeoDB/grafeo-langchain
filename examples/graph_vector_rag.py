"""GrafeoGraphVectorStore over the Silicon Valley dataset: search, traversal, filters and deletes.

Demonstrates:
- The 64 dataset descriptions as documents, its 149 edges as graph links
- Auto-detected embedding dimensions
- Similarity search, traversal search (search hits, then linked documents) and MMR traversal
- Metadata filters, the MMR retriever, and deleting a document with its links

Uses Mistral embeddings when MISTRAL_API_KEY is set, and an offline keyword stand-in otherwise. Run:
    uv run python examples/graph_vector_rag.py                                # no API key
    uv run --with langchain-mistralai python examples/graph_vector_rag.py    # with MISTRAL_API_KEY
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re

from grafeo_datasets import silicon_valley
from langchain_core.embeddings import Embeddings

from grafeo_langchain import GrafeoGraphVectorStore


class KeywordEmbeddings(Embeddings):
    """Offline stand-in for an embedding model: texts that share words come out close."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in re.findall(r"[a-z]{3,}", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        return vector if any(vector) else [1.0] + [0.0] * (self.dims - 1)


def pick_embeddings() -> Embeddings:
    if os.environ.get("MISTRAL_API_KEY"):
        # optional (uv run --with langchain-mistralai): imported by name, so type checkers do not require it
        MistralAIEmbeddings = importlib.import_module("langchain_mistralai").MistralAIEmbeddings
        print("Embeddings: Mistral (mistral-embed)")
        return MistralAIEmbeddings(model="mistral-embed")
    print("Embeddings: offline keyword stand-in (set MISTRAL_API_KEY to use Mistral)")
    return KeywordEmbeddings()


def show(title: str, docs: list) -> None:
    print(f"\n=== {title} ===")
    for doc in docs:
        hop = "hit" if doc.metadata["_depth"] == 0 else f"{doc.metadata['_depth']} hop"
        print(f"  [{doc.id}] ({hop}) {doc.page_content[:70]}...")


# ── Documents and links from the dataset ───────────────────────────────────────

nodes = silicon_valley.nodes()
links: dict[str, list[dict[str, str]]] = {}
for edge in silicon_valley.edges():
    links.setdefault(edge["source"], []).append({"target_id": edge["target"], "type": edge["type"]})

store = GrafeoGraphVectorStore(embedding=pick_embeddings())
store.add_texts(
    texts=[node["description"] for node in nodes],
    metadatas=[
        {"label": node["label"], "name": node["name"], "__graph_links__": links.get(node["key"], [])} for node in nodes
    ],
    ids=[node["key"] for node in nodes],
)
print(f"{len(nodes)} documents")

show("An AI trained to write and fix software", store.similarity_search("an AI trained to write and fix software", k=2))
show(
    "Pied Piper's systems architect, then what he built",
    store.traversal_search("Pied Piper's systems architect who trusts his server", k=1, depth=1),
)
show(
    "Apps that recognise food in photos (MMR)",
    store.mmr_traversal_search("apps that recognise food in photos", k=2, depth=1),
)
show(
    "A machine with a voice, AIs only",
    store.similarity_search("a machine with a screen and a voice", k=2, filter={"label": "AI"}),
)
show("MMR retriever: Hooli", store.as_retriever(search_type="mmr", search_kwargs={"k": 2}).invoke("Hooli's products"))

store.delete(["russ-car"])
print("\n=== Delete ===")
print("  russ-car left:", store.get_by_ids(["russ-car"]) != [])
