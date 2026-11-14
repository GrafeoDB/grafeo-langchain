"""Write the grafeo 0.5.44 databases that ``tests/test_migration.py`` opens with grafeo 0.6.

The databases are written by the previous release of this package (grafeo-langchain 0.2.0) on
grafeo 0.5.44, as a user upgrading to 0.3.0 has them. Run it in an empty scratch directory, outside
the project, and copy the output into ``tests/fixtures/``::

    uv run --no-project --with grafeo==0.5.44 --with grafeo-langchain==0.2.0 \\
        python tests/fixtures/make_grafeo_0_5_fixtures.py <scratch dir>

It writes:

- ``vectors-0.5.44.grafeo``: a ``GrafeoGraphVectorStore`` in a single file;
- ``vectors-0.5.44.db/``: the same store in a 0.5.x WAL directory (a path without ``.grafeo``);
- ``knowledge-0.5.44.grafeo``: a ``GrafeoGraphStore`` with a source document.

The embeddings are the ``FakeEmbeddings`` of ``tests/conftest.py``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import grafeo
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from grafeo_langchain import GrafeoGraphStore, GrafeoGraphVectorStore, GraphDocument, Node, Relationship

DIMS = 4

TEXTS = ["alpha document", "beta document", "gamma document"]
IDS = ["alpha", "beta", "gamma"]
METADATAS = [
    {
        "category": "even",
        "rank": 1,
        "weight": 0.25,
        "active": True,
        "__graph_links__": [{"target_id": "beta", "type": "CITES", "properties": {"weight": 0.5}}],
    },
    {"category": "odd", "rank": 2, "weight": 0.5, "active": False, "__graph_links__": [{"target_id": "gamma"}]},
    {"category": "even", "rank": 3, "weight": 0.75, "active": True},
]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [b / 255.0 for b in hashlib.sha256(text.encode()).digest()[:DIMS]]


def write_vector_store(path: Path) -> None:
    store = GrafeoGraphVectorStore(FakeEmbeddings(), db_path=str(path))
    store.add_texts(TEXTS, metadatas=METADATAS, ids=IDS)
    store.close()


def write_graph_store(path: Path) -> None:
    alice = Node(id="alice", type="Person", properties={"name": "Alice", "age": 30})
    acme = Node(id="acme", type="Company", properties={"name": "Acme Corp"})
    source = Document(page_content="Alice works at Acme Corp.", metadata={"id": "doc1"})
    doc = GraphDocument(nodes=[alice, acme], relationships=[Relationship(alice, acme, "WORKS_AT")], source=source)
    store = GrafeoGraphStore(db_path=str(path))
    store.add_graph_documents([doc], include_source=True)
    store.close()


def main(out_dir: str) -> None:
    assert grafeo.__version__ == "0.5.44", f"run with grafeo 0.5.44, not {grafeo.__version__}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_vector_store(out / "vectors-0.5.44.grafeo")
    write_vector_store(out / "vectors-0.5.44.db")
    write_graph_store(out / "knowledge-0.5.44.grafeo")


if __name__ == "__main__":
    main(sys.argv[1])
