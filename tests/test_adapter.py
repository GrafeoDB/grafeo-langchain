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


# ── Blind spots: dimensions, filters, deletes, traversal ─────────────────────


class TestAdapterEdgeCases:
    def test_search_wrong_dimensions(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        with pytest.raises(ValueError, match="dimensional"):
            adapter._search([1.0, 0.0], k=2)

    def test_search_empty_store(self, adapter: GrafeoAdapter, embedding: FakeEmbeddings) -> None:
        assert adapter._search(embedding.embed_query("anything"), k=4) == []

    def test_get_filter_requires_same_type(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["typed"], metadatas=[{"n": 1}], ids=["typed"])
        assert [d.id for d in adapter._get(["typed"], filter={"n": 1})] == ["typed"]
        assert adapter._get(["typed"], filter={"n": True}) == []
        assert adapter._get(["typed"], filter={"n": 1.0}) == []

    def test_get_and_search_skip_deleted(
        self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings
    ) -> None:
        store.add_texts(TEXTS, metadatas=METAS, ids=IDS)
        store.delete(["beta"])
        assert [d.id for d in adapter._get(["alpha", "beta"])] == ["alpha"]
        found = {d.id for d in adapter._search(embedding.embed_query("document"), k=10)}
        assert found == {"alpha", "gamma"}

    def test_get_after_replacing_a_document(self, adapter: GrafeoAdapter, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["old"], metadatas=[{"category": "old"}], ids=["doc"])
        store.add_texts(["new"], metadatas=[{"category": "new"}], ids=["doc"])
        docs = adapter._get(["doc"])
        assert [(d.page_content, d.metadata["category"]) for d in docs] == [("new", "new")]


class TestEagerTraversal:
    """The retriever extra end to end: Eager traversal over metadata edges."""

    @pytest.fixture(autouse=True)
    def _populate(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(
            ["cats purr", "dogs bark", "fish swim", "lions roar"],
            metadatas=[
                {"family": "felidae", "habitat": "home"},
                {"family": "canidae", "habitat": "home"},
                {"family": "fish", "habitat": "water"},
                {"family": "felidae", "habitat": "savanna"},
            ],
            ids=["cat", "dog", "fish", "lion"],
        )

    def test_follows_metadata_edges(self, adapter: GrafeoAdapter) -> None:
        from graph_retriever.strategies import Eager
        from langchain_graph_retriever import GraphRetriever

        retriever = GraphRetriever(
            store=adapter,
            edges=[("family", "family")],
            strategy=Eager(select_k=10, start_k=1, adjacent_k=10, max_depth=1),
        )
        docs = retriever.invoke("cats purr")
        ids = [d.id for d in docs]
        assert ids[0] == "cat"
        assert "lion" in ids, "a same-family document must be reached through the metadata edge"
        assert "fish" not in ids

    def test_depth_zero_returns_start_documents_only(self, adapter: GrafeoAdapter) -> None:
        from graph_retriever.strategies import Eager
        from langchain_graph_retriever import GraphRetriever

        retriever = GraphRetriever(
            store=adapter,
            edges=[("habitat", "habitat")],
            strategy=Eager(select_k=10, start_k=1, adjacent_k=10, max_depth=0),
        )
        assert [d.id for d in retriever.invoke("dogs bark")] == ["dog"]

    def test_filter_applies_to_traversal(self, adapter: GrafeoAdapter) -> None:
        from graph_retriever.strategies import Eager
        from langchain_graph_retriever import GraphRetriever

        retriever = GraphRetriever(
            store=adapter,
            edges=[("habitat", "habitat")],
            strategy=Eager(select_k=10, start_k=2, adjacent_k=10, max_depth=2),
        )
        docs = retriever.invoke("dogs bark", filter={"habitat": "home"})
        assert {d.id for d in docs} <= {"cat", "dog"}
        assert docs
