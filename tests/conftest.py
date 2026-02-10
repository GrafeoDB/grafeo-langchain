from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from grafeo_langchain import GraphDocument, Node, Relationship

DIMS = 4


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing — no model required."""

    def __init__(self, dims: int = DIMS) -> None:
        self.dims = dims

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dims]]


# ── Shared test data ──────────────────────────────────────────────────────────

ALICE = Node(id="alice", type="Person", properties={"name": "Alice", "age": 30})
BOB = Node(id="bob", type="Person", properties={"name": "Bob", "age": 25})
ACME = Node(id="acme", type="Company", properties={"name": "Acme Corp"})

ALICE_WORKS_AT_ACME = Relationship(source=ALICE, target=ACME, type="WORKS_AT")
BOB_WORKS_AT_ACME = Relationship(source=BOB, target=ACME, type="WORKS_AT")
ALICE_KNOWS_BOB = Relationship(source=ALICE, target=BOB, type="KNOWS")

SOURCE_DOC = Document(page_content="Alice and Bob work at Acme Corp.", metadata={"id": "doc1"})

SAMPLE_GRAPH_DOC = GraphDocument(
    nodes=[ALICE, BOB, ACME],
    relationships=[ALICE_WORKS_AT_ACME, BOB_WORKS_AT_ACME, ALICE_KNOWS_BOB],
    source=SOURCE_DOC,
)
