from __future__ import annotations

import grafeo
import pytest
from langchain_core.documents import Document

from grafeo_langchain import GrafeoGraphVectorStore

from .conftest import DIMS, FakeEmbeddings

HAS_MMR = hasattr(grafeo.GrafeoDB, "mmr_search")


@pytest.fixture
def embedding() -> FakeEmbeddings:
    return FakeEmbeddings(dims=DIMS)


@pytest.fixture
def store(embedding: FakeEmbeddings) -> GrafeoGraphVectorStore:
    return GrafeoGraphVectorStore(embedding, embedding_dimensions=DIMS)


TEXTS = [
    "Alice works at Acme Corp",
    "Bob works at Acme Corp",
    "Charlie works at Beta Inc",
    "The weather is sunny today",
]


# ── add_texts ──────────────────────────────────────────────────────────────────


class TestAddTexts:
    def test_returns_ids(self, store: GrafeoGraphVectorStore) -> None:
        ids = store.add_texts(TEXTS)
        assert len(ids) == 4

    def test_custom_ids(self, store: GrafeoGraphVectorStore) -> None:
        ids = store.add_texts(TEXTS[:2], ids=["doc-a", "doc-b"])
        assert ids == ["doc-a", "doc-b"]

    def test_metadata_stored(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["hello"], metadatas=[{"source": "test", "page": 1}], ids=["m1"])
        docs = store.similarity_search("hello", k=1)
        assert docs[0].metadata["source"] == "test"
        assert docs[0].metadata["page"] == 1

    def test_graph_links_create_edges(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["target doc"], ids=["t1"])
        store.add_texts(
            ["linked doc"],
            metadatas=[{"__graph_links__": [{"target_id": "t1", "type": "CITES"}]}],
            ids=["l1"],
        )
        results = store._db.execute(
            "MATCH ()-[r:CITES]->() RETURN r",
        )
        assert len(list(results)) == 1

    def test_metadata_not_mutated(self, store: GrafeoGraphVectorStore) -> None:
        """add_texts should not mutate the caller's metadata dicts."""
        meta = {"__graph_links__": [{"target_id": "x", "type": "LINKS_TO"}], "extra": "kept"}
        store.add_texts(["doc"], metadatas=[meta], ids=["d1"])
        assert "__graph_links__" in meta


# ── Similarity search ─────────────────────────────────────────────────────────


class TestSimilaritySearch:
    def test_returns_documents(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.similarity_search("Alice works at Acme", k=2)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_page_content_preserved(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert docs[0].page_content == TEXTS[0]

    def test_score_in_metadata(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert "score" in docs[0].metadata
        assert 0.0 <= docs[0].metadata["score"] <= 1.0

    def test_k_limits_results(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.similarity_search("anything", k=1)
        assert len(docs) == 1

    def test_by_vector(self, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings) -> None:
        store.add_texts(TEXTS)
        vec = embedding.embed_query(TEXTS[0])
        docs = store.similarity_search_by_vector(vec, k=1)
        assert len(docs) == 1
        assert docs[0].page_content == TEXTS[0]

    def test_exact_match_is_top_result(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.similarity_search(TEXTS[0], k=4)
        assert docs[0].page_content == TEXTS[0]


# ── Traversal search ──────────────────────────────────────────────────────────


class TestTraversalSearch:
    def test_finds_linked_documents(self, store: GrafeoGraphVectorStore) -> None:
        """Traversal follows outgoing edges from seed nodes."""
        store.add_texts(["neighbor document"], ids=["neighbor"])
        # seed links OUT to neighbor — traversal from seed follows outgoing edges
        store.add_texts(
            ["seed document"],
            metadatas=[{"__graph_links__": [{"target_id": "neighbor", "type": "RELATES_TO"}]}],
            ids=["seed"],
        )
        docs = store.traversal_search("seed document", k=1, depth=1)
        texts = {d.page_content for d in docs}
        assert "seed document" in texts
        assert "neighbor document" in texts

    def test_depth_zero_returns_seeds_only(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["linked"], ids=["l1"])
        store.add_texts(
            ["seed"],
            metadatas=[{"__graph_links__": [{"target_id": "l1", "type": "LINKS_TO"}]}],
            ids=["s1"],
        )
        docs = store.traversal_search("seed", k=1, depth=0)
        assert all(d.page_content != "linked" or d.metadata.get("source") != "graph_traversal" for d in docs)


# ── MMR traversal search ──────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_MMR, reason="grafeo build lacks mmr_search")
class TestMmrTraversalSearch:
    def test_returns_documents(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.mmr_traversal_search("Alice", k=2, depth=0)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_finds_linked_documents(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["mmr neighbor"], ids=["mmr_nb"])
        # seed links OUT to neighbor — traversal from seed follows outgoing edges
        store.add_texts(
            ["seed for mmr"],
            metadatas=[{"__graph_links__": [{"target_id": "mmr_nb", "type": "RELATES_TO"}]}],
            ids=["mmr_seed"],
        )
        docs = store.mmr_traversal_search("seed for mmr", k=1, depth=1)
        texts = {d.page_content for d in docs}
        assert "seed for mmr" in texts
        assert "mmr neighbor" in texts


# ── Factory methods ────────────────────────────────────────────────────────────


class TestFactoryMethods:
    def test_from_texts(self, embedding: FakeEmbeddings) -> None:
        store = GrafeoGraphVectorStore.from_texts(TEXTS, embedding, embedding_dimensions=DIMS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert docs[0].page_content == TEXTS[0]

    def test_from_documents(self, embedding: FakeEmbeddings) -> None:
        lc_docs = [Document(page_content=t, metadata={"idx": i}) for i, t in enumerate(TEXTS)]
        store = GrafeoGraphVectorStore.from_documents(lc_docs, embedding, embedding_dimensions=DIMS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert docs[0].page_content == TEXTS[0]


# ── Lifecycle ──────────────────────────────────────────────────────────────────


class TestVectorStoreLifecycle:
    def test_embeddings_property(self, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings) -> None:
        assert store.embeddings is embedding

    def test_close(self, store: GrafeoGraphVectorStore) -> None:
        store.close()  # should not raise

    def test_context_manager(self, embedding: FakeEmbeddings) -> None:
        with GrafeoGraphVectorStore(embedding, embedding_dimensions=DIMS) as store:
            store.add_texts(TEXTS)
            docs = store.similarity_search(TEXTS[0], k=1)
            assert docs[0].page_content == TEXTS[0]
