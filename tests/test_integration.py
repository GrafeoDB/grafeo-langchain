"""Integration tests with real embedding models.

These tests require an OpenAI API key and are skipped by default.
Run with:
    uv run pytest -m integration -v

Requires:
    OPENAI_API_KEY environment variable
    pip install langchain-openai
"""

from __future__ import annotations

import os

import pytest

from grafeo_langchain import GrafeoGraphVectorStore

pytestmark = pytest.mark.integration

HAVE_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore[unresolved-import]

    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

skip_reason = (
    "langchain-openai not installed" if not HAVE_OPENAI else "OPENAI_API_KEY not set" if not HAVE_OPENAI_KEY else ""
)
requires_openai = pytest.mark.skipif(not (HAVE_OPENAI and HAVE_OPENAI_KEY), reason=skip_reason or "n/a")


@requires_openai
class TestOpenAIIntegration:
    @pytest.fixture
    def store(self) -> GrafeoGraphVectorStore:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return GrafeoGraphVectorStore(embedding=embeddings, embedding_dimensions=1536)

    def test_end_to_end_similarity_search(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(
            texts=[
                "The capital of France is Paris.",
                "Machine learning is a subset of artificial intelligence.",
                "The Great Wall of China is visible from space.",
            ],
            ids=["france", "ml", "wall"],
        )
        docs = store.similarity_search("What is the capital of France?", k=1)
        assert len(docs) == 1
        assert "Paris" in docs[0].page_content

    def test_end_to_end_traversal_search(self, store: GrafeoGraphVectorStore) -> None:
        store.add_texts(["Neural networks are used in deep learning."], ids=["nn"])
        store.add_texts(
            ["Deep learning revolutionized computer vision."],
            metadatas=[{"__graph_links__": [{"target_id": "nn", "type": "BUILDS_ON"}]}],
            ids=["dl"],
        )
        docs = store.traversal_search("deep learning", k=1, depth=1)
        texts = {d.page_content for d in docs}
        assert any("neural" in t.lower() or "deep learning" in t.lower() for t in texts)
