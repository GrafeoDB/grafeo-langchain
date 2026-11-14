"""Opening a store written by grafeo-langchain 0.2.0 on grafeo 0.5.44.

grafeo 0.6 migrates a 0.5.x database to its new file format on the first read-write open and keeps
the old files next to it: ``<file>.pre-0.6`` for a ``.grafeo`` file, ``<path>.pre-0.6/`` for a WAL
directory.  The fixtures come from ``tests/fixtures/make_grafeo_0_5_fixtures.py``.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document

from grafeo_langchain import GrafeoGraphStore, GrafeoGraphVectorStore, GraphDocument, Node, Relationship

from .conftest import ALICE, DIMS, FakeEmbeddings

FIXTURES = Path(__file__).parent / "fixtures"


def _copy(name: str, tmp_path: Path) -> Path:
    source = FIXTURES / name
    target = tmp_path / name
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def _same_tree(left: Path, right: Path) -> bool:
    comparison = filecmp.dircmp(left, right)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(left, right, comparison.common_files, shallow=False)
    return not mismatch and not errors and all(_same_tree(left / d, right / d) for d in comparison.common_dirs)


@pytest.mark.parametrize("name", ["vectors-0.5.44.grafeo", "vectors-0.5.44.db"], ids=["file", "wal-directory"])
class TestVectorStoreMigration:
    def test_documents_vectors_and_links(self, tmp_path: Path, name: str) -> None:
        path = _copy(name, tmp_path)
        with GrafeoGraphVectorStore(FakeEmbeddings(dims=DIMS), db_path=str(path)) as store:
            hits = store.similarity_search("beta document", k=3)
            assert hits[0].id == "beta"
            assert hits[0].metadata["score"] == pytest.approx(1.0, abs=1e-5)
            docs = {d.id: d for d in hits}
            assert set(docs) == {"alpha", "beta", "gamma"}
            alpha = docs["alpha"]
            assert alpha.page_content == "alpha document"
            meta = {k: alpha.metadata[k] for k in ("category", "rank", "weight", "active")}
            assert meta == {"category": "even", "rank": 1, "weight": 0.25, "active": True}
            assert docs["beta"].metadata["active"] is False

            linked = store.traversal_search("alpha document", k=1, depth=2)
            assert [d.id for d in linked] == ["alpha", "beta"], "k=1 keeps the seed and its closest link"
            assert [d.id for d in store.traversal_search("alpha document", k=2, depth=2)][-2:] == ["beta", "gamma"]
            query = "MATCH (a)-[r]->(b) RETURN a.doc_id AS a, type(r) AS t, r.weight AS w, b.doc_id AS b"
            edges = store._db.execute(query)
            assert sorted((row["a"], row["t"], row["w"], row["b"]) for row in edges) == [
                ("alpha", "CITES", 0.5, "beta"),
                ("beta", "LINKS_TO", None, "gamma"),
            ]

    def test_old_files_kept_and_new_database_is_one_file(self, tmp_path: Path, name: str) -> None:
        path = _copy(name, tmp_path)
        with GrafeoGraphVectorStore(FakeEmbeddings(dims=DIMS), db_path=str(path)):
            pass
        backup = tmp_path / f"{name}.pre-0.6"
        assert path.is_file(), "the migrated database is a single file at the same path"
        if (FIXTURES / name).is_dir():
            assert backup.is_dir() and _same_tree(backup, FIXTURES / name), "the WAL directory is kept unchanged"
        else:
            assert backup.is_file() and filecmp.cmp(backup, FIXTURES / name, shallow=False)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted([name, backup.name])

    def test_writes_after_migration_survive_reopen(self, tmp_path: Path, name: str) -> None:
        path = _copy(name, tmp_path)
        embedding = FakeEmbeddings(dims=DIMS)
        with GrafeoGraphVectorStore(embedding, db_path=str(path)) as store:
            new = store.add_texts(["delta document"], metadatas=[{"__graph_links__": [{"target_id": "alpha"}]}])
            store.delete(["beta"])
        backup = tmp_path / f"{name}.pre-0.6"
        backup_state = sorted(p.name for p in backup.rglob("*")) if backup.is_dir() else backup.stat().st_size

        with GrafeoGraphVectorStore(embedding, db_path=str(path)) as store:
            found = {d.id for d in store.similarity_search("anything", k=10)}
            assert found == {"alpha", "gamma", new[0]}
            assert new[0] not in {"alpha", "beta", "gamma"}
            assert [d.id for d in store.traversal_search("delta document", k=1, depth=1)] == [new[0], "alpha"]
        assert (sorted(p.name for p in backup.rglob("*")) if backup.is_dir() else backup.stat().st_size) == backup_state


class TestGraphStoreMigration:
    def test_graph_and_reingest(self, tmp_path: Path) -> None:
        name = "knowledge-0.5.44.grafeo"
        path = _copy(name, tmp_path)
        with GrafeoGraphStore(db_path=str(path)) as store:
            schema = store.get_structured_schema
            assert schema["labels"] == ["Company", "Person", "SourceDocument"]
            assert schema["edge_types"] == ["MENTIONED_IN", "WORKS_AT"]
            rows = store.query(
                "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name AS p, p.age AS age, c.name AS c"
            )
            assert rows == [{"p": "Alice", "age": 30, "c": "Acme Corp"}]
            text = store.query("MATCH (s:SourceDocument) RETURN s.node_id AS id, s.text AS text")
            assert text == [{"id": "source_doc1", "text": "Alice works at Acme Corp."}]

            # ingesting the same facts again finds the migrated nodes and edges
            acme = Node(id="acme", type="Company")
            again = GraphDocument(
                nodes=[ALICE, acme],
                relationships=[Relationship(ALICE, acme, "WORKS_AT")],
                source=Document(page_content="Alice works at Acme Corp.", metadata={"id": "doc1"}),
            )
            store.add_graph_documents([again], include_source=True)
            assert store.query("MATCH (n) RETURN count(n) AS n")[0]["n"] == 3
            assert store.query("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"] == 3
        assert filecmp.cmp(tmp_path / f"{name}.pre-0.6", FIXTURES / name, shallow=False)
