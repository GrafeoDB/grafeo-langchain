from __future__ import annotations

from typing import cast

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


# ── Empty/whitespace inputs (T2) ────────────────────────────────────────────


class TestEmptyInputs:
    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_or_whitespace_text(self, store: GrafeoGraphVectorStore, text: str) -> None:
        """Empty or whitespace-only texts should not produce ghost nodes or should raise."""
        try:
            ids = store.add_texts([text])
            # If it succeeds, verify we can search without error
            docs = store.similarity_search("anything", k=10)
            # The added text should be retrievable (it was accepted)
            assert len(ids) == 1
            assert isinstance(docs, list)
        except (ValueError, RuntimeError):
            pass  # raising is also acceptable


# ── Circular graph links (T3) ───────────────────────────────────────────────


class TestCircularLinks:
    def test_circular_traversal_terminates(self, store: GrafeoGraphVectorStore) -> None:
        """A->B->C->A cycle should not cause infinite loop in traversal."""
        store.add_texts(
            ["node A", "node B", "node C"],
            metadatas=[
                {"__graph_links__": [{"target_id": "b", "type": "LINKS_TO"}]},
                {"__graph_links__": [{"target_id": "c", "type": "LINKS_TO"}]},
                {"__graph_links__": [{"target_id": "a", "type": "LINKS_TO"}]},
            ],
            ids=["a", "b", "c"],
        )
        docs = store.traversal_search("node A", k=4, depth=5)
        assert isinstance(docs, list)
        # Should return finite results (no duplicates from looping)
        doc_ids = [d.metadata.get("id") for d in docs]
        assert len(doc_ids) == len(set(doc_ids))


# ── Self-links (T4) ─────────────────────────────────────────────────────────


class TestSelfLinks:
    def test_self_link_traversal(self, store: GrafeoGraphVectorStore) -> None:
        """A node linking to itself should not crash traversal."""
        store.add_texts(
            ["self-referencing node"],
            metadatas=[{"__graph_links__": [{"target_id": "self", "type": "SELF_REF"}]}],
            ids=["self"],
        )
        docs = store.traversal_search("self-referencing", k=4, depth=2)
        assert isinstance(docs, list)
        assert len(docs) >= 1


# ── Embedding dimension mismatch (T5) ───────────────────────────────────────


class TestEmbeddingDimensionMismatch:
    def test_mismatched_dimensions_at_construction(self) -> None:
        """Explicit dimension mismatch at construction should raise ValueError."""
        emb = FakeEmbeddings(dims=4)
        with pytest.raises(ValueError, match="does not match"):
            GrafeoGraphVectorStore(emb, embedding_dimensions=128)


# ── Delete and rebuild (T7) ─────────────────────────────────────────────────


class TestDeleteAndRebuild:
    def test_search_after_partial_delete(self, store: GrafeoGraphVectorStore) -> None:
        """After deleting half the documents, similarity_search returns only the remaining."""
        ids = store.add_texts(
            [f"document {i}" for i in range(10)],
            ids=[f"d{i}" for i in range(10)],
        )
        to_delete = ids[:5]
        store.delete(to_delete)
        docs = store.similarity_search("document", k=10)
        returned_ids = {d.metadata["id"] for d in docs}
        assert len(returned_ids) == 5
        for did in to_delete:
            assert did not in returned_ids


# ── Large batch (T8) ────────────────────────────────────────────────────────


class TestLargeBatch:
    def test_add_500_documents(self, store: GrafeoGraphVectorStore) -> None:
        """Adding 500 documents should not truncate or corrupt."""
        texts = [f"large batch document number {i}" for i in range(500)]
        ids = [f"batch-{i}" for i in range(500)]
        metadatas = [{"batch_index": i} for i in range(500)]
        returned_ids = store.add_texts(texts, metadatas=metadatas, ids=ids)
        assert len(returned_ids) == 500

        # Verify search returns correct results
        docs = store.similarity_search("large batch document number 0", k=10)
        assert len(docs) == 10
        assert all(isinstance(d, Document) for d in docs)


# ── Helpers for the tests below ─────────────────────────────────────────────


def _edges(store: GrafeoGraphVectorStore) -> list[tuple[str, str, str]]:
    rows = store._db.execute("MATCH (s)-[r]->(t) RETURN s.doc_id AS s, type(r) AS type, t.doc_id AS t")
    return sorted((row["s"], row["type"], row["t"]) for row in rows)


def _doc_ids(store: GrafeoGraphVectorStore) -> list[str]:
    rows = store._db.execute("MATCH (n:Document) RETURN n.doc_id AS doc_id")
    return sorted(row["doc_id"] for row in rows)


def _link(target: str, edge_type: str = "LINKS_TO", **props: object) -> dict:
    link: dict = {"target_id": target, "type": edge_type}
    if props:
        link["properties"] = props
    return {"__graph_links__": [link]}


class WrongSizeEmbeddings(FakeEmbeddings):
    """Probes with DIMS dimensions, then embeds documents with one dimension too many."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[*vec, 0.0] for vec in super().embed_documents(texts)]


# ── Re-adding an id replaces the document ───────────────────────────────────


class TestUpsertById:
    def test_same_id_twice_keeps_one_document(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["first version"], metadatas=[{"rev": 1}], ids=["doc"])
        store.add_texts(["second version"], metadatas=[{"rev": 2}], ids=["doc"])
        docs = store.similarity_search("second version", k=10)
        assert [(d.page_content, d.metadata["rev"]) for d in docs] == [("second version", 2)]

    def test_stale_metadata_removed(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["text"], metadatas=[{"old": "x"}], ids=["doc"])
        store.add_texts(["text"], metadatas=[{"new": "y"}], ids=["doc"])
        meta = store.similarity_search("text", k=1)[0].metadata
        assert "old" not in meta
        assert meta["new"] == "y"

    def test_embedding_replaced_in_index(self, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings) -> None:
        store.add_texts(["alpha"], ids=["doc"])
        store.add_texts(["omega"], ids=["doc"])
        hit = store.similarity_search_by_vector(embedding.embed_query("omega"), k=1)[0]
        assert hit.page_content == "omega"
        assert hit.metadata["score"] == pytest.approx(1.0, abs=1e-5)

    def test_duplicate_id_within_one_batch(self, store: GrafeoGraphVectorStore) -> None:
        ids = store.add_texts(["first", "last"], ids=["doc", "doc"])
        assert ids == ["doc", "doc"]
        assert _doc_ids(store) == ["doc"]
        assert store.similarity_search("last", k=10)[0].page_content == "last"

    def test_links_kept_and_not_duplicated(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["target", "other"], ids=["t", "o"])
        store.add_texts(["source"], metadatas=[_link("t")], ids=["s"])
        store.add_texts(["source again"], metadatas=[_link("t")], ids=["s"])
        store.add_texts(["source, new link"], metadatas=[_link("o", "CITES")], ids=["s"])
        assert _edges(store) == [("s", "CITES", "o"), ("s", "LINKS_TO", "t")]

    def test_incoming_links_survive_replacement(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["target"], ids=["t"])
        store.add_texts(["source"], metadatas=[_link("t")], ids=["s"])
        store.add_texts(["target, replaced"], ids=["t"])
        assert _edges(store) == [("s", "LINKS_TO", "t")]
        docs = store.traversal_search("source", k=1, depth=1)
        assert "target, replaced" in {d.page_content for d in docs}

    def test_link_properties_updated(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["target"], ids=["t"])
        store.add_texts(["source"], metadatas=[_link("t", weight=1)], ids=["s"])
        store.add_texts(["source"], metadatas=[_link("t", weight=2)], ids=["s"])
        rows = list(store._db.execute("MATCH ()-[r:LINKS_TO]->() RETURN r.weight AS weight"))
        assert rows == [{"weight": 2}]


# ── Generated ids ───────────────────────────────────────────────────────────


class TestGeneratedIds:
    def test_sequential_ids_for_a_new_store(self, store: GrafeoGraphVectorStore) -> None:
        assert store.add_texts(["a", "b"]) == ["0", "1"]
        assert store.add_texts(["c"]) == ["2"]

    def test_generated_ids_skip_ids_in_use(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["explicit"], ids=["1"])
        assert store.add_texts(["a", "b"]) == ["0", "2"]
        assert _doc_ids(store) == ["0", "1", "2"]

    def test_none_entries_get_generated_ids(self, store: GrafeoGraphVectorStore) -> None:
        given: list = ["x", None, "0"]
        ids = store.add_texts(["a", "b", "c"], ids=given)
        assert ids == ["x", "1", "0"]
        assert _doc_ids(store) == ["0", "1", "x"]

    def test_add_documents_with_and_without_ids(self, store: GrafeoGraphVectorStore) -> None:
        ids = store.add_documents([Document(page_content="a", id="x"), Document(page_content="b")])
        assert ids[0] == "x"
        assert ids[1] is not None
        assert _doc_ids(store) == sorted(ids)

    def test_returned_documents_carry_their_id(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["hello"], metadatas=[{"id": "user-meta-id"}], ids=["doc-1"])
        doc = store.similarity_search("hello", k=1)[0]
        assert doc.id == "doc-1"


# ── Validation ──────────────────────────────────────────────────────────────


class TestAddTextsValidation:
    def test_empty_input(self, store: GrafeoGraphVectorStore) -> None:
        assert store.add_texts([]) == []

    @pytest.mark.parametrize(
        ("metadatas", "ids"),
        [([{}], None), (None, ["only-one"])],
        ids=["metadatas", "ids"],
    )
    def test_length_mismatch(self, store: GrafeoGraphVectorStore, metadatas: list | None, ids: list | None) -> None:
        with pytest.raises(ValueError, match="lengths must match"):
            store.add_texts(["a", "b"], metadatas=metadatas, ids=ids)
        assert _doc_ids(store) == []

    def test_reserved_metadata_keys_ignored(self, store: GrafeoGraphVectorStore) -> None:
        meta = {"text": "not the text", "doc_id": "not the id", "embedding": "not a vector", "kept": 1}
        store.add_texts(["the text"], metadatas=[meta], ids=["the-id"])
        doc = store.similarity_search("the text", k=1)[0]
        assert doc.page_content == "the text"
        assert doc.id == "the-id"
        assert doc.metadata["kept"] == 1
        assert _doc_ids(store) == ["the-id"]

    def test_unsupported_metadata_values_dropped(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["t"], metadatas=[{"tags": ["a"], "nested": {"k": 1}, "none": None, "ok": "yes"}], ids=["d"])
        meta = store.similarity_search("t", k=1)[0].metadata
        assert meta["ok"] == "yes"
        assert not {"tags", "nested", "none"} & set(meta)

    def test_wrong_size_document_embeddings_rejected(self) -> None:
        store = GrafeoGraphVectorStore(WrongSizeEmbeddings())
        with pytest.raises(ValueError, match="dimensional"):
            store.add_texts(["a", "b"], ids=["a", "b"])
        assert _doc_ids(store) == []


class TestQueryDimensions:
    """Grafeo panics on a search vector of the wrong size: the store must raise ValueError first."""

    def test_similarity_search_by_vector(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        with pytest.raises(ValueError, match="dimensional"):
            store.similarity_search_by_vector([1.0, 0.0], k=1)

    @pytest.mark.skipif(not HAS_MMR, reason="grafeo build lacks mmr_search")
    def test_mmr_traversal_search(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS)
        store._embedding = FakeEmbeddings(dims=DIMS + 1)
        with pytest.raises(ValueError, match="dimensional"):
            store.mmr_traversal_search("query", k=1)


# ── Empty store ─────────────────────────────────────────────────────────────


class TestEmptyStore:
    def test_similarity_search(self, store: GrafeoGraphVectorStore) -> None:
        assert store.similarity_search("anything", k=4) == []

    def test_traversal_search(self, store: GrafeoGraphVectorStore) -> None:
        assert store.traversal_search("anything", k=4, depth=2) == []

    @pytest.mark.skipif(not HAS_MMR, reason="grafeo build lacks mmr_search")
    def test_mmr_traversal_search(self, store: GrafeoGraphVectorStore) -> None:
        assert store.mmr_traversal_search("anything", k=4) == []

    def test_after_deleting_everything(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(TEXTS, ids=["a", "b", "c", "d"])
        store.delete(["a", "b", "c", "d"])
        assert store.similarity_search("anything", k=4) == []


# ── Search after delete ─────────────────────────────────────────────────────


class TestSearchAfterDelete:
    """Grafeo's HNSW index can lose live vectors when one is removed: the store rebuilds it."""

    @pytest.mark.parametrize("deleted", [["d1"], ["d2"], ["d1", "d2"], ["d0", "d4"]])
    def test_every_remaining_document_found(self, store: GrafeoGraphVectorStore, deleted: list[str]) -> None:
        ids = [f"d{i}" for i in range(5)]
        store.add_texts([f"document {i}" for i in range(5)], ids=ids)
        store.delete(deleted)
        found = {d.id for d in store.similarity_search("document", k=10)}
        assert found == set(ids) - set(deleted)

    def test_adapter_search_after_delete(self, store: GrafeoGraphVectorStore, embedding: FakeEmbeddings) -> None:
        pytest.importorskip("langchain_graph_retriever")
        from grafeo_langchain.adapter import GrafeoAdapter

        store.add_texts([f"document {i}" for i in range(5)], ids=[f"d{i}" for i in range(5)])
        store.delete(["d1"])
        docs = GrafeoAdapter(vector_store=store)._search(embedding.embed_query("document"), k=10)
        assert {d.id for d in docs} == {"d0", "d2", "d3", "d4"}

    def test_replacing_documents_keeps_others_searchable(self, store: GrafeoGraphVectorStore) -> None:
        ids = [f"d{i}" for i in range(5)]
        store.add_texts([f"document {i}" for i in range(5)], ids=ids)
        store.add_texts(["replaced one", "replaced two"], ids=["d1", "d2"])
        assert {d.id for d in store.similarity_search("document", k=10)} == set(ids)


# ── Delete with graph links ─────────────────────────────────────────────────


class TestDeleteLinkedDocuments:
    @pytest.fixture(autouse=True)
    def _chain(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(
            ["first", "middle", "last"],
            metadatas=[_link("middle"), _link("last"), {}],
            ids=["first", "middle", "last"],
        )

    def test_delete_link_target(self, store: GrafeoGraphVectorStore) -> None:
        assert store.delete(["last"]) is True
        assert _doc_ids(store) == ["first", "middle"]
        assert _edges(store) == [("first", "LINKS_TO", "middle")]

    def test_delete_node_with_incoming_and_outgoing_links(self, store: GrafeoGraphVectorStore) -> None:
        assert store.delete(["middle"]) is True
        assert _edges(store) == []
        texts = {d.page_content for d in store.traversal_search("first", k=1, depth=3)}
        assert "middle" not in texts
        assert "last" not in texts

    def test_delete_mixed_existing_and_missing(self, store: GrafeoGraphVectorStore) -> None:
        assert store.delete(["missing", "first"]) is True
        assert _doc_ids(store) == ["last", "middle"]

    def test_deleted_document_not_returned(self, store: GrafeoGraphVectorStore) -> None:
        store.delete(["first"])
        assert "first" not in {d.id for d in store.similarity_search("first", k=10)}


# ── Traversal ordering and filtering ────────────────────────────────────────

CHAIN_IDS = ["seed", "hop1", "near", "far1", "far2", "far3"]


def _build_tree(store: GrafeoGraphVectorStore) -> None:
    # seed -> hop1 -> far1, far2, far3 ; seed -> near
    store.add_texts(
        CHAIN_IDS,
        metadatas=[
            {"__graph_links__": [{"target_id": "hop1"}, {"target_id": "near"}]},
            {"__graph_links__": [{"target_id": t} for t in ("far1", "far2", "far3")]},
            {},
            {},
            {},
            {},
        ],
        ids=CHAIN_IDS,
    )


class TestTraversalOrdering:
    @pytest.mark.usefixtures("shuffled_rows")
    def test_closest_neighbors_first(self, embedding: FakeEmbeddings) -> None:
        """With k=1 only two documents come back: the seed and a direct neighbor."""
        store = GrafeoGraphVectorStore(embedding)
        _build_tree(store)
        for _ in range(10):
            docs = store.traversal_search("seed", k=1, depth=2, filter=None)
            assert docs[0].id == "seed"
            assert len(docs) == 2
            assert docs[1].id in {"hop1", "near"}, "a direct neighbor must win over a two-hop one"

    @pytest.mark.usefixtures("shuffled_rows")
    def test_neighbors_by_distance_in_a_stable_order(self, embedding: FakeEmbeddings) -> None:
        store = GrafeoGraphVectorStore(embedding)
        _build_tree(store)
        seed = store._vector_hits(embedding.embed_query("seed"), 1, {"text": "seed"})
        orders = set()
        for _ in range(10):
            ids = [d.id for d in store._expand(seed, depth=2, filter=None, limit=5)]
            assert ids[0] == "seed"
            assert set(ids[1:3]) == {"hop1", "near"}
            assert set(ids[3:]) <= {"far1", "far2", "far3"}
            orders.add(tuple(ids))
        assert len(orders) == 1, f"traversal order changes between calls: {orders}"

    def test_depth_must_be_an_integer(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["seed"], ids=["seed"])
        depth = cast("int", "1] MATCH (x) DETACH DELETE x //")
        with pytest.raises(TypeError):
            store.traversal_search("seed", k=1, depth=depth)
        assert _doc_ids(store) == ["seed"]


class TestTraversalFilter:
    @pytest.fixture(autouse=True)
    def _populate(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["off-topic neighbor"], metadatas=[{"cat": "b"}], ids=["nb_b"])
        store.add_texts(["on-topic neighbor"], metadatas=[{"cat": "a"}], ids=["nb_a"])
        store.add_texts(
            ["seed"],
            metadatas=[{"cat": "a", "__graph_links__": [{"target_id": "nb_b"}, {"target_id": "nb_a"}]}],
            ids=["seed"],
        )

    def test_traversal_search_filters_neighbors(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.traversal_search("seed", k=1, depth=1, filter={"cat": "a"})
        assert {d.id for d in docs} <= {"seed", "nb_a"}
        assert "nb_b" not in {d.id for d in docs}

    @pytest.mark.skipif(not HAS_MMR, reason="grafeo build lacks mmr_search")
    def test_mmr_traversal_search_filters_neighbors(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.mmr_traversal_search("seed", k=2, depth=1, filter={"cat": "a"})
        assert {d.metadata["cat"] for d in docs} == {"a"}

    def test_filter_type_must_match(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["typed"], metadatas=[{"n": 1, "__graph_links__": [{"target_id": "nb_a"}]}], ids=["typed"])
        assert store.traversal_search("typed", k=1, depth=1, filter={"n": True}) == []

    def test_without_filter_all_neighbors(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.traversal_search("seed", k=3, depth=1, filter={"cat": "a"})
        assert {d.metadata["cat"] for d in docs} == {"a"}
        unfiltered = store.traversal_search("seed", k=3, depth=1)
        assert {"nb_a", "nb_b"} <= {d.id for d in unfiltered}


# ── Factory ids ─────────────────────────────────────────────────────────────


class TestFactoryIds:
    def test_from_texts_with_ids(self, embedding: FakeEmbeddings) -> None:
        store = GrafeoGraphVectorStore.from_texts(TEXTS[:2], embedding, ids=["a", "b"])
        assert _doc_ids(store) == ["a", "b"]

    def test_from_documents_uses_document_ids(self, embedding: FakeEmbeddings) -> None:
        docs = [Document(page_content=t, id=f"doc-{i}") for i, t in enumerate(TEXTS[:2])]
        store = GrafeoGraphVectorStore.from_documents(docs, embedding)
        assert _doc_ids(store) == ["doc-0", "doc-1"]

    def test_from_documents_explicit_ids_win(self, embedding: FakeEmbeddings) -> None:
        docs = [Document(page_content=t, id=f"doc-{i}") for i, t in enumerate(TEXTS[:2])]
        store = GrafeoGraphVectorStore.from_documents(docs, embedding, ids=["x", "y"])
        assert _doc_ids(store) == ["x", "y"]
