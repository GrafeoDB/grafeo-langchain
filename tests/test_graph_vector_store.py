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
    return GrafeoGraphVectorStore(embedding)


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
        with GrafeoGraphVectorStore(embedding) as store:
            store.add_texts(TEXTS)
            docs = store.similarity_search(TEXTS[0], k=1)
            assert docs[0].page_content == TEXTS[0]


# ── Auto dimension detection ─────────────────────────────────────────────────


class TestAutoDimensionDetection:
    def test_auto_detects_dimensions(self, embedding: FakeEmbeddings) -> None:
        """Store works without specifying embedding_dimensions."""
        store = GrafeoGraphVectorStore(embedding)
        store.add_texts(TEXTS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert docs[0].page_content == TEXTS[0]

    def test_explicit_dimensions_accepted(self, embedding: FakeEmbeddings) -> None:
        """Explicit dimensions matching the model are accepted."""
        store = GrafeoGraphVectorStore(embedding, embedding_dimensions=DIMS)
        store.add_texts(TEXTS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert len(docs) == 1

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched explicit dimensions raise ValueError."""
        emb = FakeEmbeddings(dims=DIMS)
        with pytest.raises(ValueError, match="does not match"):
            GrafeoGraphVectorStore(emb, embedding_dimensions=9999)


# ── Filter support ───────────────────────────────────────────────────────────


class TestFilterSupport:
    def test_similarity_search_filter(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(
            ["doc A", "doc B"],
            metadatas=[{"category": "alpha"}, {"category": "beta"}],
            ids=["a", "b"],
        )
        docs = store.similarity_search("doc", k=2, filter={"category": "alpha"})
        assert all(d.metadata.get("category") == "alpha" for d in docs)

    def test_similarity_search_by_vector_filter(self, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings) -> None:
        store.add_texts(
            ["doc A", "doc B"],
            metadatas=[{"category": "alpha"}, {"category": "beta"}],
            ids=["a", "b"],
        )
        vec = embedding.embed_query("doc")
        docs = store.similarity_search_by_vector(vec, k=2, filter={"category": "alpha"})
        assert all(d.metadata.get("category") == "alpha" for d in docs)

    @pytest.mark.skipif(not HAS_MMR, reason="grafeo build lacks mmr_search")
    def test_mmr_traversal_search_filter(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(
            ["doc A", "doc B"],
            metadatas=[{"category": "alpha"}, {"category": "beta"}],
            ids=["a", "b"],
        )
        docs = store.mmr_traversal_search("doc", k=2, depth=0, filter={"category": "alpha"})
        assert all(d.metadata.get("category") == "alpha" for d in docs)


# ── Delete ───────────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_removes_documents(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["doc to keep", "doc to delete"], ids=["keep", "del"])
        result = store.delete(["del"])
        assert result is True
        docs = store.similarity_search("doc", k=10)
        doc_ids = {d.metadata.get("id") for d in docs}
        assert "del" not in doc_ids
        assert "keep" in doc_ids

    def test_delete_empty_ids(self, store: GrafeoGraphVectorStore) -> None:
        result = store.delete([])
        assert result is False

    def test_delete_none_ids(self, store: GrafeoGraphVectorStore) -> None:
        result = store.delete(None)
        assert result is False

    def test_delete_nonexistent_id(self, store: GrafeoGraphVectorStore) -> None:
        result = store.delete(["nonexistent"])
        assert result is False


# ── Graph link ordering ──────────────────────────────────────────────────────


class TestGraphLinkOrdering:
    def test_forward_reference_in_same_batch(self, store: GrafeoGraphVectorStore) -> None:
        """Source added before target in the same batch: links should still work."""
        store.add_texts(
            ["source doc", "target doc"],
            metadatas=[
                {"__graph_links__": [{"target_id": "target", "type": "CITES"}]},
                {},
            ],
            ids=["source", "target"],
        )
        results = store._db.execute("MATCH ()-[r:CITES]->() RETURN r")
        assert len(list(results)) == 1

    def test_backward_reference_in_same_batch(self, store: GrafeoGraphVectorStore) -> None:
        """Target added before source in the same batch: links should work too."""
        store.add_texts(
            ["target doc", "source doc"],
            metadatas=[
                {},
                {"__graph_links__": [{"target_id": "target", "type": "REFS"}]},
            ],
            ids=["target", "source"],
        )
        results = store._db.execute("MATCH ()-[r:REFS]->() RETURN r")
        assert len(list(results)) == 1


# ── Score metadata consistency ───────────────────────────────────────────────


class TestScoreMetadata:
    def test_vector_search_has_source_vector(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        docs = store.similarity_search(TEXTS[0], k=1)
        assert docs[0].metadata.get("source") == "vector"

    def test_traversal_neighbors_have_score_none(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["neighbor"], ids=["nb"])
        store.add_texts(
            ["seed"],
            metadatas=[{"__graph_links__": [{"target_id": "nb", "type": "LINKS_TO"}]}],
            ids=["seed"],
        )
        docs = store.traversal_search("seed", k=1, depth=1)
        traversed = [d for d in docs if d.metadata.get("source") == "graph_traversal"]
        assert len(traversed) >= 1
        for d in traversed:
            assert "score" in d.metadata
            assert d.metadata["score"] is None
