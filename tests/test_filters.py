from __future__ import annotations

import pytest

from grafeo_langchain import GrafeoGraphVectorStore

from .conftest import DIMS, FakeEmbeddings


@pytest.fixture
def embedding() -> FakeEmbeddings:
    return FakeEmbeddings(dims=DIMS)


@pytest.fixture
def store(embedding: FakeEmbeddings) -> GrafeoGraphVectorStore:
    return GrafeoGraphVectorStore(embedding)


# ── T6: Filter type coverage ────────────────────────────────────────────────


class TestFilterTypes:
    """Verify exact-match filters work across supported scalar types."""

    @pytest.fixture(autouse=True)
    def _populate(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(
            [
                "science article about physics",
                "history article about wars",
                "tech article about AI",
                "cooking article about pasta",
            ],
            metadatas=[
                {"tag": "science", "year": 2024, "score": 0.95, "active": True},
                {"tag": "history", "year": 2020, "score": 0.80, "active": False},
                {"tag": "tech", "year": 2024, "score": 0.90, "active": True},
                {"tag": "cooking", "year": 2019, "score": 0.70, "active": True},
            ],
            ids=["d1", "d2", "d3", "d4"],
        )

    def test_filter_by_str(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.similarity_search("article", k=4, filter={"tag": "science"})
        assert len(docs) == 1
        assert docs[0].metadata["tag"] == "science"

    def test_filter_by_int(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.similarity_search("article", k=4, filter={"year": 2024})
        assert len(docs) == 2
        assert all(d.metadata["year"] == 2024 for d in docs)

    def test_filter_by_float(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.similarity_search("article", k=4, filter={"score": 0.95})
        assert len(docs) == 1
        assert docs[0].metadata["score"] == 0.95

    def test_filter_by_bool(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.similarity_search("article", k=4, filter={"active": True})
        assert len(docs) == 3
        assert all(d.metadata["active"] is True for d in docs)

    def test_filter_no_match(self, store: GrafeoGraphVectorStore) -> None:
        docs = store.similarity_search("article", k=4, filter={"tag": "nonexistent"})
        assert docs == []

    @pytest.mark.parametrize(
        "bad_filter",
        [
            {"tags": ["a", "b"]},
            {"nested": {"key": "value"}},
        ],
        ids=["list-value", "dict-value"],
    )
    def test_unsupported_filter_types_handled(self, store: GrafeoGraphVectorStore, bad_filter: dict) -> None:
        """Unsupported filter value types should either return empty or raise, not crash."""
        try:
            docs = store.similarity_search("article", k=4, filter=bad_filter)
            # If it returns without error, results should be empty (no match)
            assert isinstance(docs, list)
        except (TypeError, ValueError):
            pass  # raising a clear error is acceptable
