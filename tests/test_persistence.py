from __future__ import annotations

from pathlib import Path

import pytest

from grafeo_langchain import GrafeoGraphVectorStore

from .conftest import DIMS, FakeEmbeddings

# A ".grafeo" path is a single-file database; any other path a WAL directory,
# which keeps no indexes across a reopen before grafeo 0.6.0.
pytestmark = pytest.mark.parametrize("suffix", [".grafeo", ".db"], ids=["single-file", "wal-directory"])


def _ids(store: GrafeoGraphVectorStore) -> list[str]:
    rows = store._db.execute("MATCH (n:Document) RETURN n.doc_id AS doc_id")
    return sorted(row["doc_id"] for row in rows)


class TestPersistenceRoundTrip:
    """T1: Data survives close/reopen when using db_path."""

    def test_data_survives_reopen(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"test{suffix}")
        embedding = FakeEmbeddings(dims=DIMS)

        texts = [f"document number {i}" for i in range(10)]
        metadatas = [{"category": "even" if i % 2 == 0 else "odd", "index": i} for i in range(10)]
        ids = [f"doc-{i}" for i in range(10)]

        # Write phase
        store = GrafeoGraphVectorStore(embedding, db_path=db_path)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        store.close()

        # Reopen phase
        store2 = GrafeoGraphVectorStore(embedding, db_path=db_path)
        docs = store2.similarity_search("document number 0", k=10)
        assert len(docs) == 10

        # Verify metadata round-trips
        found = {d.metadata["id"]: d for d in docs}
        assert "doc-0" in found
        assert found["doc-0"].metadata["category"] == "even"
        assert found["doc-0"].metadata["index"] == 0
        assert docs[0].page_content == "document number 0"
        store2.close()

    def test_graph_links_survive_reopen(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"links{suffix}")
        embedding = FakeEmbeddings(dims=DIMS)

        store = GrafeoGraphVectorStore(embedding, db_path=db_path)
        store.add_texts(["target doc"], ids=["target"])
        store.add_texts(
            ["source doc"],
            metadatas=[{"__graph_links__": [{"target_id": "target", "type": "CITES"}]}],
            ids=["source"],
        )
        store.close()

        store2 = GrafeoGraphVectorStore(embedding, db_path=db_path)
        docs = store2.traversal_search("source doc", k=1, depth=1)
        texts = {d.page_content for d in docs}
        assert "target doc" in texts
        store2.close()


class TestReopenedStoreWrites:
    def test_index_follows_writes_after_reopen(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"writes{suffix}")
        embedding = FakeEmbeddings(dims=DIMS)
        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            store.add_texts(["old one", "old two"], ids=["old1", "old2"])

        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            store.add_texts(["new one"], ids=["new1"])
            store.add_texts(["old two, replaced"], ids=["old2"])
            store.delete(["old1"])
            docs = store.similarity_search("anything", k=10)
            assert sorted(d.id for d in docs if d.id) == ["new1", "old2"]
            assert store.similarity_search("old two, replaced", k=1)[0].id == "old2"

        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            assert _ids(store) == ["new1", "old2"]

    def test_generated_ids_after_delete_and_reopen(self, tmp_path: Path, suffix: str) -> None:
        """Generated ids never reuse a stored id (which would replace that document)."""
        db_path = str(tmp_path / f"ids{suffix}")
        embedding = FakeEmbeddings(dims=DIMS)
        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            first = store.add_texts(["a", "b", "c"])
            store.delete([first[0]])

        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            new = store.add_texts(["d"])
            assert new[0] not in first[1:]
            assert len(_ids(store)) == 3
            assert {d.page_content for d in store.similarity_search("x", k=10)} == {"b", "c", "d"}

    def test_deleted_links_stay_deleted(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"del{suffix}")
        embedding = FakeEmbeddings(dims=DIMS)
        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            store.add_texts(
                ["source", "target"],
                metadatas=[{"__graph_links__": [{"target_id": "target"}]}, {}],
                ids=["source", "target"],
            )
            store.delete(["target"])

        with GrafeoGraphVectorStore(embedding, db_path=db_path) as store:
            assert _ids(store) == ["source"]
            assert store._db.edge_count == 0


class TestReopenWithOtherModel:
    def test_other_dimensions_rejected(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"dims{suffix}")
        with GrafeoGraphVectorStore(FakeEmbeddings(dims=DIMS), db_path=db_path) as store:
            store.add_texts(["a"], ids=["a"])

        with pytest.raises(ValueError, match="dimensional embeddings"):
            GrafeoGraphVectorStore(FakeEmbeddings(dims=DIMS * 2), db_path=db_path)

        # the database is left as it was
        with GrafeoGraphVectorStore(FakeEmbeddings(dims=DIMS), db_path=db_path) as store:
            assert _ids(store) == ["a"]
