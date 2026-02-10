from __future__ import annotations

import pytest

from grafeo_langchain import GrafeoGraphStore, GraphDocument, Node, Relationship

from .conftest import (
    ALICE,
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
        results = store.query("MATCH (n {node_id: 'alice'}) RETURN n")
        assert len(results) == 1
        node = store.client.get_node(results[0]["n"])
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

        results = store.query("MATCH (n {node_id: 'alice'}) RETURN n")
        node = store.client.get_node(results[0]["n"])
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
