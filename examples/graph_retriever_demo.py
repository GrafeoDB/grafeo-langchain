"""GrafeoAdapter with langchain-graph-retriever on the Silicon Valley dataset.

Demonstrates:
- The dataset's people as documents, with their company and city as metadata
- GraphRetriever with the Eager strategy, following metadata edges: colleagues share a company
- A metadata filter on the retriever

Uses Mistral embeddings when MISTRAL_API_KEY is set, and an offline keyword stand-in otherwise.

Requires:
    uv add "grafeo-langchain[retriever]"

Run:
    uv run python examples/graph_retriever_demo.py                                # no API key
    uv run --with langchain-mistralai python examples/graph_retriever_demo.py    # with MISTRAL_API_KEY
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re

from grafeo_datasets import silicon_valley
from graph_retriever.strategies import Eager
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever import GraphRetriever

from grafeo_langchain import GrafeoGraphVectorStore
from grafeo_langchain.adapter import GrafeoAdapter


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
        place = f"{doc.metadata.get('company', '-')}, {doc.metadata.get('city', '-')}"
        print(f"  [{doc.id}] {place}: {doc.page_content[:60]}...")


# ── People, with their company and city ───────────────────────────────────────

names = {node["key"]: node["name"] for node in silicon_valley.nodes()}
company: dict[str, str] = {}
city: dict[str, str] = {}
for edge in silicon_valley.edges():
    if edge["type"] == "WORKS_AT":
        company.setdefault(edge["source"], names[edge["target"]])
    elif edge["type"] == "LIVES_IN":
        city.setdefault(edge["source"], names[edge["target"]])

people = [node for node in silicon_valley.nodes() if node["label"] == "Person"]
store = GrafeoGraphVectorStore(embedding=pick_embeddings())
store.add_texts(
    texts=[person["description"] for person in people],
    metadatas=[
        {
            "name": person["name"],
            **({"company": company[person["key"]]} if person["key"] in company else {}),
            **({"city": city[person["key"]]} if person["key"] in city else {}),
        }
        for person in people
    ],
    ids=[person["key"] for person in people],
)

# ── The retriever: the best match, then the people who share its company ───

retriever = GraphRetriever(
    store=GrafeoAdapter(vector_store=store),
    edges=[("company", "company")],
    strategy=Eager(select_k=4, start_k=1, max_depth=1),
)

show(
    "The systems architect who trusts his server, then his colleagues",
    retriever.invoke("the systems architect who trusts his own server"),
)
show(
    "The incubator owner, only among people who live in Palo Alto",
    retriever.invoke("the owner of the incubator in his own house", filter={"city": "Palo Alto"}),
)

store.close()
