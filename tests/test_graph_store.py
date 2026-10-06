from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.documents import Document

from grafeo_langchain import GrafeoGraphStore, GraphDocument, Node, Relationship

from .conftest import (
    ALICE,
    ALICE_KNOWS_BOB,
    BOB,
    SAMPLE_GRAPH_DOC,
    SOURCE_DOC,
)


@pytest.fixture
def store() -> GrafeoGraphStore:
    return GrafeoGraphStore()


# ── Schema ─────────────────────────────────────────────────────────────────────


class TestSchema:
    def test_empty_schema_string(self, store: GrafeoGraphStore) -> None:
        assert store.get_schema == "Empty graph"

    def test_empty_structured_schema(self, store: GrafeoGraphStore) -> None:
        schema = store.get_structured_schema
        assert schema["labels"] == []
        assert schema["edge_types"] == []

    def test_schema_after_ingestion(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        store.refresh_schema()
        schema = store.get_structured_schema
        assert "Person" in schema["labels"]
        assert "Company" in schema["labels"]
        assert "WORKS_AT" in schema["edge_types"]
        assert "KNOWS" in schema["edge_types"]

    def test_schema_string_after_ingestion(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        store.refresh_schema()
        text = store.get_schema
        assert "Node labels:" in text
        assert "Person" in text
        assert "Relationship types:" in text

    def test_schema_invalidated_after_add(self, store: GrafeoGraphStore) -> None:
        """add_graph_documents should clear the cached schema."""
        _ = store.get_structured_schema  # prime cache
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        # Internal cache should be cleared; next access re-fetches
        assert store._schema is None


# ── Ingestion ──────────────────────────────────────────────────────────────────


class TestAddGraphDocuments:
    def test_creates_nodes(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        results = store.query("MATCH (n:Person) RETURN n")
        assert len(results) == 2

    def test_creates_edges(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        results = store.query("MATCH ()-[r:WORKS_AT]->() RETURN r")
        assert len(results) == 2

    def test_edge_type_normalized(self, store: GrafeoGraphStore) -> None:
        """Spaces and dashes in relationship types become underscored uppercase."""
        node_a = Node(id="a", type="X")
        node_b = Node(id="b", type="X")
        rel = Relationship(source=node_a, target=node_b, type="works-at here")
        doc = GraphDocument(nodes=[node_a, node_b], relationships=[rel], source=SOURCE_DOC)
        store.add_graph_documents([doc])
        results = store.query("MATCH ()-[r:WORKS_AT_HERE]->() RETURN r")
        assert len(results) == 1

    def test_include_source(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        results = store.query("MATCH (n:SourceDocument) RETURN n")
        assert len(results) == 1
        results = store.query("MATCH ()-[r:MENTIONED_IN]->() RETURN r")
        assert len(results) == 3  # alice, bob, acme all linked to source

    def test_node_properties_stored(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        gids = store.client.find_nodes_by_property("node_id", "alice")
        assert len(gids) == 1
        node = store.client.get_node(gids[0])
        props = node.properties()
        assert props["name"] == "Alice"
        assert props["age"] == 30


# ── Upsert ─────────────────────────────────────────────────────────────────────


class TestUpsertBehavior:
    def test_duplicate_node_id_not_duplicated(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        results = store.query("MATCH (n:Person) RETURN n")
        assert len(results) == 2  # not 4

    def test_upsert_updates_properties(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        updated_alice = Node(id="alice", type="Person", properties={"name": "Alice Updated", "age": 31})
        doc2 = GraphDocument(nodes=[updated_alice], relationships=[], source=SOURCE_DOC)
        store.add_graph_documents([doc2])

        gids = store.client.find_nodes_by_property("node_id", "alice")
        node = store.client.get_node(gids[0])
        props = node.properties()
        assert props["name"] == "Alice Updated"
        assert props["age"] == 31


# ── Query ──────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_parameterized_query(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        results = store.query("MATCH (n {node_id: $nid}) RETURN n", {"nid": "alice"})
        assert len(results) == 1

    def test_empty_results(self, store: GrafeoGraphStore) -> None:
        results = store.query("MATCH (n:NonExistent) RETURN n")
        assert results == []

    def test_relationship_with_new_nodes(self, store: GrafeoGraphStore) -> None:
        """Relationships can reference nodes not in the nodes list — they get created."""
        eve = Node(id="eve", type="Person")
        rel = Relationship(source=eve, target=ALICE, type="FOLLOWS")
        doc = GraphDocument(nodes=[ALICE], relationships=[rel], source=SOURCE_DOC)
        store.add_graph_documents([doc])
        results = store.query("MATCH (e {node_id: 'eve'})-[:FOLLOWS]->(a {node_id: 'alice'}) RETURN e, a")
        assert len(results) == 1


# ── Lifecycle ──────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_client_property(self, store: GrafeoGraphStore) -> None:
        assert store.client is store._db

    def test_close(self, store: GrafeoGraphStore) -> None:
        store.close()  # should not raise

    def test_context_manager(self) -> None:
        with GrafeoGraphStore() as store:
            store.add_graph_documents([SAMPLE_GRAPH_DOC])
            results = store.query("MATCH (n:Person) RETURN n")
            assert len(results) == 2


# ── Invalid label fallback ─────────────────────────────────────────────────────


class TestLabelNormalization:
    def test_non_identifier_label_falls_back(self, store: GrafeoGraphStore) -> None:
        node = Node(id="x", type="123-bad!")
        doc = GraphDocument(nodes=[node], relationships=[], source=SOURCE_DOC)
        store.add_graph_documents([doc])
        results = store.query("MATCH (n:Node {node_id: 'x'}) RETURN n")
        assert len(results) == 1

    def test_label_with_spaces_normalized(self, store: GrafeoGraphStore) -> None:
        node = Node(id="y", type="My Type")
        doc = GraphDocument(nodes=[node], relationships=[], source=SOURCE_DOC)
        store.add_graph_documents([doc])
        results = store.query("MATCH (n:My_Type {node_id: 'y'}) RETURN n")
        assert len(results) == 1


# ── Delete behavior (T9) ────────────────────────────────────────────────────


class TestGraphStoreDelete:
    def test_delete_not_supported(self, store: GrafeoGraphStore) -> None:
        """GraphStore has no delete method; attempting to call it should signal clearly."""
        assert not hasattr(store, "delete"), (
            "GrafeoGraphStore does not implement delete; "
            "if it does, this test should be updated to verify correct behavior"
        )


# ── Schema refresh after mutation (T10) ─────────────────────────────────────


class TestSchemaRefreshAfterMutation:
    def test_schema_grows_with_new_labels(self, store: GrafeoGraphStore) -> None:
        """Adding documents with new labels should update the schema."""
        person = Node(id="p1", type="Person", properties={"name": "Alice"})
        doc1 = GraphDocument(nodes=[person], relationships=[], source=SOURCE_DOC)
        store.add_graph_documents([doc1])
        store.refresh_schema()
        schema1 = store.get_structured_schema
        assert "Person" in schema1["labels"]

        company = Node(id="c1", type="Company", properties={"name": "Acme"})
        doc2 = GraphDocument(nodes=[company], relationships=[], source=SOURCE_DOC)
        store.add_graph_documents([doc2])
        store.refresh_schema()
        schema2 = store.get_structured_schema
        assert "Person" in schema2["labels"]
        assert "Company" in schema2["labels"]

    def test_schema_grows_with_new_edge_types(self, store: GrafeoGraphStore) -> None:
        """Adding relationships with new types should update the schema."""
        alice = Node(id="a", type="Person")
        bob = Node(id="b", type="Person")
        rel_knows = Relationship(source=alice, target=bob, type="KNOWS")
        doc1 = GraphDocument(nodes=[alice, bob], relationships=[rel_knows], source=SOURCE_DOC)
        store.add_graph_documents([doc1])
        store.refresh_schema()
        assert "KNOWS" in store.get_structured_schema["edge_types"]

        acme = Node(id="acme", type="Company")
        rel_employs = Relationship(source=acme, target=alice, type="EMPLOYS")
        doc2 = GraphDocument(nodes=[acme], relationships=[rel_employs], source=SOURCE_DOC)
        store.add_graph_documents([doc2])
        store.refresh_schema()
        schema = store.get_structured_schema
        assert "KNOWS" in schema["edge_types"]
        assert "EMPLOYS" in schema["edge_types"]


# ── Idempotent re-ingestion ─────────────────────────────────────────────────


def _count(store: GrafeoGraphStore, pattern: str) -> int:
    return len(store.query(f"MATCH {pattern} RETURN 1 AS one"))


class TestIdempotentIngestion:
    def test_relationships_not_duplicated(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        assert _count(store, "()-[:WORKS_AT]->()") == 2
        assert _count(store, "()-[:KNOWS]->()") == 1

    def test_mentioned_in_not_duplicated(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        assert _count(store, "(:SourceDocument)") == 1
        assert _count(store, "()-[:MENTIONED_IN]->()") == 3

    def test_same_relationship_twice_in_one_document(self, store: GrafeoGraphStore) -> None:
        doc = GraphDocument(nodes=[ALICE], relationships=[ALICE_KNOWS_BOB, ALICE_KNOWS_BOB], source=SOURCE_DOC)
        store.add_graph_documents([doc])
        assert _count(store, "()-[:KNOWS]->()") == 1

    def test_relationship_properties_updated(self, store: GrafeoGraphStore) -> None:
        first = Relationship(source=ALICE, target=BOB, type="KNOWS", properties={"since": 2020})
        second = Relationship(source=ALICE, target=BOB, type="KNOWS", properties={"since": 2024})
        store.add_graph_documents([GraphDocument(nodes=[], relationships=[first], source=SOURCE_DOC)])
        store.add_graph_documents([GraphDocument(nodes=[], relationships=[second], source=SOURCE_DOC)])
        rows = store.query("MATCH ()-[r:KNOWS]->() RETURN r.since AS since")
        assert rows == [{"since": 2024}]

    def test_other_types_and_directions_are_separate_edges(self, store: GrafeoGraphStore) -> None:
        rels = [
            Relationship(source=ALICE, target=BOB, type="KNOWS"),
            Relationship(source=ALICE, target=BOB, type="MANAGES"),
            Relationship(source=BOB, target=ALICE, type="KNOWS"),
        ]
        store.add_graph_documents([GraphDocument(nodes=[ALICE, BOB], relationships=rels, source=SOURCE_DOC)])
        store.add_graph_documents([GraphDocument(nodes=[ALICE, BOB], relationships=rels, source=SOURCE_DOC)])
        assert _count(store, "()-[]->()") == 3


# ── Source document identity ────────────────────────────────────────────────


class TestSourceDocumentId:
    def _source_ids(self, store: GrafeoGraphStore) -> list[str]:
        return sorted(row["id"] for row in store.query("MATCH (s:SourceDocument) RETURN s.node_id AS id"))

    def test_metadata_id_used(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        assert self._source_ids(store) == ["source_doc1"]

    def test_document_id_used_without_metadata_id(self, store: GrafeoGraphStore) -> None:
        doc = GraphDocument(nodes=[ALICE], relationships=[], source=Document(page_content="text", id="lc-id"))
        store.add_graph_documents([doc], include_source=True)
        assert self._source_ids(store) == ["source_lc-id"]

    def test_hash_of_full_text_without_ids(self, store: GrafeoGraphStore) -> None:
        """The fallback id is stable across processes (Python's hash() is not)."""
        text = "Alice met Bob."
        doc = GraphDocument(nodes=[ALICE], relationships=[], source=Document(page_content=text))
        store.add_graph_documents([doc], include_source=True)
        expected = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        assert self._source_ids(store) == [f"source_{expected}"]

    def test_documents_sharing_a_prefix_stay_apart(self, store: GrafeoGraphStore) -> None:
        prefix = "x" * 300
        for suffix, node in (("one", ALICE), ("two", BOB)):
            doc = GraphDocument(nodes=[node], relationships=[], source=Document(page_content=prefix + suffix))
            store.add_graph_documents([doc], include_source=True)
        assert len(self._source_ids(store)) == 2
        rows = store.query("MATCH (e)-[:MENTIONED_IN]->(s:SourceDocument) RETURN e.node_id AS e, s.node_id AS s")
        assert len({row["s"] for row in rows}) == 2


# ── Schema reporting ────────────────────────────────────────────────────────


class TestSchemaCache:
    def test_query_write_invalidates_schema(self, store: GrafeoGraphStore) -> None:
        assert store.get_schema == "Empty graph"
        store.query("CREATE (:Fresh {node_id: 'f'})")
        assert "Fresh" in store.get_structured_schema["labels"]

    def test_query_read_keeps_schema_cache(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        cached = store.get_structured_schema
        store.query("MATCH (n:Person) RETURN n.name")
        assert store._schema is cached

    def test_query_delete_invalidates_schema(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        assert "Company" in store.get_structured_schema["labels"]
        store.query("MATCH (n:Company) DETACH DELETE n")
        schema = store.get_structured_schema
        assert "Company" not in schema["labels"]
        assert "WORKS_AT" not in schema["edge_types"]

    def test_schema_is_sorted_and_stable(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        schema = store.get_structured_schema
        for key in ("labels", "edge_types", "property_keys"):
            assert schema[key] == sorted(schema[key])
        text = store.get_schema
        for _ in range(5):
            store.refresh_schema()
            assert store.get_schema == text


# ── Persistence ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", [".grafeo", ".db"], ids=["single-file", "wal-directory"])
class TestGraphStorePersistence:
    def test_reopen_and_reingest(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"kg{suffix}")
        with GrafeoGraphStore(db_path=db_path) as store:
            store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)

        with GrafeoGraphStore(db_path=db_path) as store:
            assert store.client.has_property_index("node_id")
            schema = store.get_structured_schema
            assert {"Person", "Company", "SourceDocument"} <= set(schema["labels"])
            store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
            assert _count(store, "(:Person)") == 2
            assert _count(store, "(:SourceDocument)") == 1
            assert _count(store, "()-[:WORKS_AT]->()") == 2
            assert _count(store, "()-[:MENTIONED_IN]->()") == 3
