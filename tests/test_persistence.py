from __future__ import annotations

from pathlib import Path

from grafeo_langchain import GrafeoGraphVectorStore

from .conftest import DIMS, FakeEmbeddings


class TestPersistenceRoundTrip:
    """T1: Data survives close/reopen when using db_path."""

    def test_data_survives_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
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
        store2.close()

    def test_graph_links_survive_reopen(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "links.db")
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
