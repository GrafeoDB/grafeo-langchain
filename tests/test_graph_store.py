from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import grafeo
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
        assert store.get_schema == (
            "Node labels and properties:\n"
            "Company {name: STRING, node_id: STRING}\n"
            "Person {age: INT64, name: STRING, node_id: STRING}\n"
            "Relationship types and properties:\n"
            "KNOWS\n"
            "WORKS_AT\n"
            "Relationships:\n"
            "(:Person)-[:KNOWS]->(:Person)\n"
            "(:Person)-[:WORKS_AT]->(:Company)"
        )

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
        """Relationships can reference nodes not in the nodes list: they get created."""
        eve = Node(id="eve", type="Person")
        rel = Relationship(source=eve, target=ALICE, type="FOLLOWS")
        doc = GraphDocument(nodes=[ALICE], relationships=[rel], source=SOURCE_DOC)
        store.add_graph_documents([doc])
        results = store.query("MATCH (e {node_id: 'eve'})-[:FOLLOWS]->(a {node_id: 'alice'}) RETURN e, a")
        assert len(results) == 1

    @pytest.mark.parametrize(("order", "expected"), [("", ["bob", "alice", "eve"]), (" DESC", ["alice", "bob", "eve"])])
    def test_order_by_puts_missing_values_last(self, store: GrafeoGraphStore, order: str, expected: list) -> None:
        # grafeo 0.6: GQL sorts nulls last in both directions (0.5 put them first with DESC)
        eve = Node(id="eve", type="Person", properties={"name": "Eve"})
        store.add_graph_documents([SAMPLE_GRAPH_DOC, GraphDocument(nodes=[eve], relationships=[], source=SOURCE_DOC)])
        rows = store.query(f"MATCH (p:Person) RETURN p.node_id AS id ORDER BY p.age{order}")
        assert [row["id"] for row in rows] == expected

    def test_number_with_a_point_in_results(self, store: GrafeoGraphStore) -> None:
        # grafeo 0.6.0: a GQL number with a point is an exact decimal, also once stored
        store.query("INSERT (:Product {node_id: 'p', price: 9.99})")
        row = store.query("MATCH (p:Product) RETURN p.price AS price, p.price * 3 AS total")[0]
        assert row == {"price": Decimal("9.99"), "total": Decimal("29.97")}

    def test_decimal_node_properties(self, store: GrafeoGraphStore) -> None:
        item = Node(id="item", type="Product", properties={"price": Decimal("9.99")})
        source = Document(page_content="An item.", metadata={"id": "s", "weight": Decimal("0.25")})
        store.add_graph_documents([GraphDocument(nodes=[item], relationships=[], source=source)], include_source=True)
        row = store.query("MATCH (p:Product), (s:SourceDocument) RETURN p.price AS price, s.weight AS weight")[0]
        assert row == {"price": Decimal("9.99"), "weight": Decimal("0.25")}
        assert isinstance(row["price"], Decimal) and isinstance(row["weight"], Decimal)


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


@pytest.mark.parametrize("suffix", [".grafeo", ".db"], ids=["grafeo-extension", "other-extension"])
class TestGraphStorePersistence:
    def test_reopen_and_reingest(self, tmp_path: Path, suffix: str) -> None:
        db_path = str(tmp_path / f"kg{suffix}")
        with GrafeoGraphStore(db_path=db_path) as store:
            store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        assert Path(db_path).is_file(), "grafeo 0.6 keeps every database in one file"

        with GrafeoGraphStore(db_path=db_path) as store:
            assert store.client.has_property_index("node_id")
            schema = store.get_structured_schema
            assert {"Person", "Company", "SourceDocument"} <= set(schema["labels"])
            store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
            assert _count(store, "(:Person)") == 2
            assert _count(store, "(:SourceDocument)") == 1
            assert _count(store, "()-[:WORKS_AT]->()") == 2
            assert _count(store, "()-[:MENTIONED_IN]->()") == 3


# ── Schema for LLM prompts ──────────────────────────────────────────────────


class TestSchemaDetail:
    """The schema text and the langchain-neo4j-shaped keys that QA chains read."""

    def test_structured_schema_neo4j_keys(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        schema = store.get_structured_schema
        assert schema["node_props"]["Person"] == [
            {"property": "age", "type": "INT64"},
            {"property": "name", "type": "STRING"},
            {"property": "node_id", "type": "STRING"},
        ]
        assert schema["rel_props"] == {"KNOWS": [], "WORKS_AT": []}
        assert schema["relationships"] == [
            {"start": "Person", "type": "KNOWS", "end": "Person"},
            {"start": "Person", "type": "WORKS_AT", "end": "Company"},
        ]

    def test_relationship_properties(self, store: GrafeoGraphStore) -> None:
        rel = Relationship(source=ALICE, target=BOB, type="KNOWS", properties={"since": 2020, "weight": 0.5})
        store.add_graph_documents([GraphDocument(nodes=[ALICE, BOB], relationships=[rel], source=SOURCE_DOC)])
        assert "KNOWS {since: INT64, weight: FLOAT64}" in store.get_schema.splitlines()

    def test_property_with_several_types(self, store: GrafeoGraphStore) -> None:
        nodes = [Node(id="a", type="Item", properties={"size": 1}), Node(id="b", type="Item", properties={"size": "L"})]
        store.add_graph_documents([GraphDocument(nodes=nodes, relationships=[], source=SOURCE_DOC)])
        assert "Item {node_id: STRING, size: INT64 | STRING}" in store.get_schema.splitlines()

    def test_value_types(self, store: GrafeoGraphStore) -> None:
        props = {"flag": True, "price": Decimal("1.50"), "ratio": 0.5}
        store.add_graph_documents(
            [GraphDocument(nodes=[Node(id="x", type="Thing", properties=props)], relationships=[], source=SOURCE_DOC)]
        )
        assert "Thing {flag: BOOL, node_id: STRING, price: DECIMAL, ratio: FLOAT64}" in store.get_schema.splitlines()

    def test_labels_without_relationships(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([GraphDocument(nodes=[ALICE], relationships=[], source=SOURCE_DOC)])
        assert store.get_schema == "Node labels and properties:\nPerson {age: INT64, name: STRING, node_id: STRING}"

    def test_source_documents_listed(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        lines = store.get_schema.splitlines()
        assert "SourceDocument {id: STRING, node_id: STRING, text: STRING}" in lines
        assert "()-[:MENTIONED_IN]->(:SourceDocument)" in lines, "one line for the store's own edges"
        assert not any("(:Company)-[:MENTIONED_IN]" in line or "(:Person)-[:MENTIONED_IN]" in line for line in lines)
        assert lines[-1] == "()-[:MENTIONED_IN]->(:SourceDocument)", "after the graph's own relationships"

    def test_patterns_merge_labels(self, store: GrafeoGraphStore) -> None:
        acme = Node(id="Acme", type="Organization")
        globex = Node(id="Globex", type="Company")
        lyon, berlin = Node(id="Lyon", type="City"), Node(id="Berlin", type="City")
        rels = [
            Relationship(source=acme, target=lyon, type="LOCATED_IN"),
            Relationship(source=globex, target=berlin, type="LOCATED_IN"),
            Relationship(source=ALICE, target=acme, type="WORKS_AT"),
            Relationship(source=BOB, target=globex, type="WORKS_AT"),
        ]
        store.add_graph_documents(
            [GraphDocument(nodes=[acme, globex, lyon, berlin, ALICE, BOB], relationships=rels, source=SOURCE_DOC)]
        )
        lines = store.get_schema.splitlines()
        relationships = lines[lines.index("Relationships:") + 1 :]
        assert relationships == [
            "(:Company|Organization)-[:LOCATED_IN]->(:City)",
            "(:Person)-[:WORKS_AT]->(:Company|Organization)",
        ]
        assert store.query("MATCH (c:Company|Organization)-[:LOCATED_IN]->(:City) RETURN count(c) AS n") == [
            {"n": 2}
        ], "the merged pattern is valid GQL, as an LLM may copy it"

    def test_structured_schema_keeps_every_pattern(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC], include_source=True)
        mentioned = [r for r in store.get_structured_schema["relationships"] if r["type"] == "MENTIONED_IN"]
        assert {r["start"] for r in mentioned} == {"Company", "Person"}

    def test_deleted_label_leaves_schema(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        store.query("MATCH (c:Company) DETACH DELETE c")
        schema = store.get_structured_schema
        assert "Company" not in schema["labels"]
        assert "Company" not in schema["node_props"]
        assert all(rel["type"] != "WORKS_AT" for rel in schema["relationships"])


# ── Read-only queries ────────────────────────────────────────────────────────


class TestReadOnlyQuery:
    def test_read_returns_rows(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        rows = store.query("MATCH (p:Person) RETURN p.node_id AS id ORDER BY id", read_only=True)
        assert rows == [{"id": "alice"}, {"id": "bob"}]

    def test_params(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        rows = store.query("MATCH (p:Person {node_id: $id}) RETURN p.age AS age", {"id": "bob"}, read_only=True)
        assert rows == [{"age": 25}]

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT (:Person {node_id: 'eve'})",
            "CREATE (:Person {node_id: 'eve'})",
            "MERGE (:Person {node_id: 'eve'})",
            "MATCH (p:Person {node_id: 'bob'}) SET p.age = 99",
            "MATCH (p:Person {node_id: 'bob'}) REMOVE p.age",
            "MATCH (p:Person {node_id: 'bob'}) DETACH DELETE p",
            "MATCH (:Person)-[r:KNOWS]->(:Person) DELETE r",
            "MATCH (p:Person {node_id: 'bob'}) SET p:Manager",
        ],
    )
    def test_write_refused_and_rolled_back(self, store: GrafeoGraphStore, query: str) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        before = store.query("MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN n, collect(r) AS rels ORDER BY n.node_id")
        with pytest.raises(ValueError, match="read-only query may not write"):
            store.query(query, read_only=True)
        after = store.query("MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN n, collect(r) AS rels ORDER BY n.node_id")
        assert after == before, "the graph is unchanged"

    def test_refused_write_keeps_schema_cache(self, store: GrafeoGraphStore) -> None:
        store.add_graph_documents([SAMPLE_GRAPH_DOC])
        cached = store.get_structured_schema
        with pytest.raises(ValueError):
            store.query("INSERT (:Robot {node_id: 'r'})", read_only=True)
        assert store.get_structured_schema is cached
        assert "Robot" not in store.get_structured_schema["labels"]

    def test_syntax_error_raises_grafeo_error(self, store: GrafeoGraphStore) -> None:
        with pytest.raises(grafeo.GrafeoError):
            store.query("MATCH (n RETURN n", read_only=True)
        assert store.query("MATCH (n) RETURN count(n) AS c", read_only=True) == [{"c": 0}], "the store still works"

    def test_store_usable_after_refusal(self, store: GrafeoGraphStore) -> None:
        with pytest.raises(ValueError):
            store.query("INSERT (:Robot {node_id: 'r'})", read_only=True)
        store.query("INSERT (:Robot {node_id: 'r'})")
        assert store.query("MATCH (r:Robot) RETURN r.node_id AS id", read_only=True) == [{"id": "r"}]

    def test_persistent_store_unchanged_after_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "ro.grafeo")
        with GrafeoGraphStore(db_path=path) as store:
            store.add_graph_documents([SAMPLE_GRAPH_DOC])
            with pytest.raises(ValueError):
                store.query("MATCH (n) DETACH DELETE n", read_only=True)
        with GrafeoGraphStore(db_path=path) as store:
            assert store.query("MATCH (n) RETURN count(n) AS c") == [{"c": 3}]
