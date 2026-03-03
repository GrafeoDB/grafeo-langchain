"""Tests for the langchain-graph-retriever adapter."""

from __future__ import annotations

import pytest

from grafeo_langchain import GrafeoGraphVectorStore

from .conftest import DIMS, FakeEmbeddings

try:
    from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY

    from grafeo_langchain.adapter import GrafeoAdapter

    HAS_ADAPTER = True
except ImportError:
    HAS_ADAPTER = False

pytestmark = pytest.mark.skipif(not HAS_ADAPTER, reason="langchain-graph-retriever not installed")


@pytest.fixture
def embedding() -> FakeEmbeddings:
    return FakeEmbeddings(dims=DIMS)


@pytest.fixture
def store(embedding: FakeEmbeddings) -> GrafeoGraphVectorStore:
    return GrafeoGraphVectorStore(embedding)


@pytest.fixture
def adapter(store: GrafeoGraphVectorStore) -> GrafeoAdapter:
    return GrafeoAdapter(vector_store=store)


TEXTS = ["Alpha document", "Beta document", "Gamma document"]
METAS: list[dict] = [{"category": "first"}, {"category": "second"}, {"category": "third"}]
IDS = ["alpha", "beta", "gamma"]


# ── _search ──────────────────────────────────────────────────────────────────


class TestAdapterSearch:
    def test_search_returns_documents_with_embedding(
        self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings
    ) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        vec = embedding.embed_query(TEXTS[0])
        docs = adapter._search(vec, k=2)
        assert len(docs) == 2
        for doc in docs:
            assert METADATA_EMBEDDING_KEY in doc.metadata
            assert doc.metadata[METADATA_EMBEDDING_KEY] is not None
            assert doc.id is not None

    def test_search_with_filter(
        self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings
    ) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        vec = embedding.embed_query("document")
        docs = adapter._search(vec, k=3, filter={"category": "first"})
        assert all(d.metadata.get("category") == "first" for d in docs)


# ── _get ─────────────────────────────────────────────────────────────────────


class TestAdapterGet:
    def test_get_by_id(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        docs = adapter._get(["alpha", "gamma"])
        assert len(docs) == 2
        returned_ids = {d.id for d in docs}
        assert "alpha" in returned_ids
        assert "gamma" in returned_ids
        for doc in docs:
            assert METADATA_EMBEDDING_KEY in doc.metadata

    def test_get_missing_id(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        docs = adapter._get(["nonexistent"])
        assert docs == []

    def test_get_with_filter(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        docs = adapter._get(["alpha", "beta"], filter={"category": "first"})
        assert len(docs) == 1
        assert docs[0].id == "alpha"


# ── GraphRetriever integration ───────────────────────────────────────────────


class TestAdapterIntegration:
    def test_graph_retriever_invoke(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        try:
            from langchain_graph_retriever import GraphRetriever
        except ImportError:
            pytest.skip("langchain-graph-retriever GraphRetriever not available")

        store.add_texts(
            ["cats are pets", "dogs are pets", "fish swim in water"],
            metadatas=[
                {"animal": "cat"},
                {"animal": "dog"},
                {"animal": "fish"},
            ],
            ids=["cat", "dog", "fish"],
        )
        retriever = GraphRetriever(
            store=adapter,
            edges=[("animal", "animal")],
        )
        docs = retriever.invoke("pets")
        assert len(docs) >= 1
