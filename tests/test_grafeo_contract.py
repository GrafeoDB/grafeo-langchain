"""Engine contract: the Grafeo behavior grafeo-langchain relies on.

Each test pins one behavior of the ``grafeo`` Python package that the stores
depend on.  When a Grafeo release changes one of them, the matching test fails
and its message names the code in grafeo-langchain that has to follow.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import grafeo
import pytest

DIMS = 4
X = [1.0, 0.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0, 0.0]
Z = [0.0, 0.0, 1.0, 0.0]


def _version_tuple(version: str) -> tuple[int, ...]:
    """Return (major, minor, patch), ignoring any pre-release suffix such as ``rc1``."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    assert match, f"unexpected grafeo version string {version!r}"
    return tuple(int(part) for part in match.groups())


@pytest.fixture
def db() -> grafeo.GrafeoDB:
    database = grafeo.GrafeoDB()
    database.create_property_index("doc_id")
    return database


@pytest.fixture
def indexed_db(db: grafeo.GrafeoDB) -> grafeo.GrafeoDB:
    db.create_vector_index("Document", "embedding", dimensions=DIMS, metric="cosine")
    return db


def _gid(db: grafeo.GrafeoDB, doc_id: str) -> int:
    matches = db.find_nodes_by_property("doc_id", doc_id)
    assert len(matches) == 1, f"expected one node with doc_id={doc_id!r}, found {matches}"
    return matches[0]


# ── Package ───────────────────────────────────────────────────────────────────


class TestPackage:
    def test_minimum_version(self) -> None:
        assert _version_tuple(grafeo.__version__) >= (0, 5, 44), (
            "grafeo-langchain needs grafeo>=0.5.44 (upsert_nodes, result counters, write validation)"
        )

    def test_features_include_query_languages_and_vector_index(self) -> None:
        features = set(grafeo.features())
        missing = {"cypher", "gql", "vector-index"} - features
        assert not missing, f"the grafeo wheel lacks features the stores use: {sorted(missing)}"

    def test_shuffle_unordered_option(self) -> None:
        # tests/conftest.py uses it to catch code that relies on unspecified row order
        grafeo.GrafeoDB(shuffle_unordered=True).close()


# ── Nodes and properties ──────────────────────────────────────────────────────


class TestNodes:
    def test_find_nodes_by_property(self, db: grafeo.GrafeoDB) -> None:
        node = db.create_node(["Document"], {"doc_id": "a"})
        assert db.find_nodes_by_property("doc_id", "a") == [node.id]
        assert db.find_nodes_by_property("doc_id", "missing") == []

    def test_property_index_round_trip(self) -> None:
        database = grafeo.GrafeoDB()
        assert not database.has_property_index("node_id")
        database.create_property_index("node_id")
        assert database.has_property_index("node_id")

    def test_properties_returns_a_copy(self, db: grafeo.GrafeoDB) -> None:
        # the stores and the adapter pop keys from the returned dict
        node = db.create_node(["Document"], {"doc_id": "a", "text": "hello"})
        props = db.get_node(node.id).properties()
        props.pop("text")
        assert db.get_node(node.id).properties()["text"] == "hello", "Node.properties() must return a copy"

    def test_vector_property_reads_back_as_list(self, db: grafeo.GrafeoDB) -> None:
        node = db.create_node(["Document"], {"doc_id": "a", "embedding": X})
        assert db.get_node(node.id).properties()["embedding"] == X

    def test_get_node_of_deleted_node_is_none(self, db: grafeo.GrafeoDB) -> None:
        node = db.create_node(["Document"], {"doc_id": "a"})
        db.delete_node(node.id)
        assert db.get_node(node.id) is None, "the stores skip search hits whose node is gone"

    def test_get_nodes_by_label_with_limit(self, db: grafeo.GrafeoDB) -> None:
        db.create_node(["Document"], {"doc_id": "a", "embedding": X})
        db.create_node(["Document"], {"doc_id": "b", "embedding": Y})
        sample = db.get_nodes_by_label("Document", limit=1)
        assert len(sample) == 1
        node_id, props = sample[0]
        assert isinstance(node_id, int)
        assert props["embedding"] in (X, Y)
        assert db.get_nodes_by_label("Missing", limit=1) == []


# ── Upserts ───────────────────────────────────────────────────────────────────


class TestUpsertNodes:
    """GrafeoGraphVectorStore.add_texts writes every batch with one merging upsert_nodes call."""

    def _upsert(self, db: grafeo.GrafeoDB, rows: list[dict]) -> dict:
        return db.upsert_nodes(["Document"], rows, key="doc_id", replace=False)

    def test_creates_then_updates_by_key(self, indexed_db: grafeo.GrafeoDB) -> None:
        first = self._upsert(indexed_db, [{"doc_id": "a", "embedding": X}])
        assert (first["created"], first["updated"]) == (1, 0)
        gid = _gid(indexed_db, "a")
        second = self._upsert(indexed_db, [{"doc_id": "a", "embedding": Y}])
        assert (second["created"], second["updated"]) == (0, 1)
        assert _gid(indexed_db, "a") == gid, "an upsert must update the node in place"

    def test_merge_keeps_other_properties(self, indexed_db: grafeo.GrafeoDB) -> None:
        # add_texts removes the stale ones itself with remove_node_property
        self._upsert(indexed_db, [{"doc_id": "a", "embedding": X, "old": 1}])
        self._upsert(indexed_db, [{"doc_id": "a", "embedding": X, "new": 2}])
        gid = _gid(indexed_db, "a")
        assert indexed_db.get_node(gid).properties()["old"] == 1
        indexed_db.remove_node_property(gid, "old")
        props = indexed_db.get_node(gid).properties()
        assert "old" not in props
        assert props["new"] == 2

    def test_update_keeps_edges(self, indexed_db: grafeo.GrafeoDB) -> None:
        self._upsert(indexed_db, [{"doc_id": "a", "embedding": X}, {"doc_id": "b", "embedding": Y}])
        indexed_db.create_edge(_gid(indexed_db, "a"), _gid(indexed_db, "b"), "LINKS_TO")
        indexed_db.create_edge(_gid(indexed_db, "b"), _gid(indexed_db, "a"), "LINKS_TO")
        self._upsert(indexed_db, [{"doc_id": "a", "embedding": Z}])
        assert indexed_db.edge_count == 2

    def test_batch_is_all_or_nothing(self, indexed_db: grafeo.GrafeoDB) -> None:
        rows = [{"doc_id": "ok", "embedding": X}, {"doc_id": "bad", "embedding": [1.0, 0.0]}]
        with pytest.raises(grafeo.GrafeoError):
            self._upsert(indexed_db, rows)
        assert indexed_db.find_nodes_by_property("doc_id", "ok") == []


# ── Vector index ──────────────────────────────────────────────────────────────


class TestVectorIndex:
    """GrafeoGraphVectorStore creates the index once and relies on Grafeo keeping it in sync."""

    def test_empty_index_with_explicit_dimensions(self, indexed_db: grafeo.GrafeoDB) -> None:
        assert indexed_db.vector_search("Document", "embedding", X, 4) == []

    def test_index_follows_create_node(self, indexed_db: grafeo.GrafeoDB) -> None:
        node = indexed_db.create_node(["Document"], {"doc_id": "a", "embedding": X})
        hits = indexed_db.vector_search("Document", "embedding", X, 1)
        assert [hit[0] for hit in hits] == [node.id], "nodes created after create_vector_index must be searchable"

    def test_index_follows_upsert_nodes(self, indexed_db: grafeo.GrafeoDB) -> None:
        indexed_db.upsert_nodes(["Document"], [{"doc_id": "a", "embedding": X}], key="doc_id", replace=False)
        indexed_db.upsert_nodes(["Document"], [{"doc_id": "a", "embedding": Z}], key="doc_id", replace=False)
        node_id, distance = indexed_db.vector_search("Document", "embedding", Z, 1)[0]
        assert node_id == _gid(indexed_db, "a")
        assert distance == pytest.approx(0.0, abs=1e-5), "an upserted embedding must replace the indexed one"

    def test_merging_upsert_keeps_other_vectors_reachable(self, indexed_db: grafeo.GrafeoDB) -> None:
        # Updating a vector in place (SET n += row) must not disturb the other vectors.
        # add_texts merges instead of replacing: replace=True removes the vector first,
        # which can leave other vectors unreachable (see the next test).
        vectors = [[0.81, 0.17, 0.15, 0.87], [0.44, 0.50, 0.26, 0.23], [0.13, 0.66, 0.43, 0.88]]
        rows = [{"doc_id": f"d{i}", "embedding": vector} for i, vector in enumerate(vectors)]
        indexed_db.upsert_nodes(["Document"], rows, key="doc_id", replace=False)
        update = [{"doc_id": "d1", "embedding": [0.03, 0.96, 0.19, 0.48]}]
        indexed_db.upsert_nodes(["Document"], update, key="doc_id", replace=False)
        hits = {node_id for node_id, _ in indexed_db.vector_search("Document", "embedding", X, 10)}
        assert hits == {_gid(indexed_db, f"d{i}") for i in range(3)}

    def test_index_follows_detach_delete(self, indexed_db: grafeo.GrafeoDB) -> None:
        keep = indexed_db.create_node(["Document"], {"doc_id": "keep", "embedding": X})
        gone = indexed_db.create_node(["Document"], {"doc_id": "gone", "embedding": Y})
        indexed_db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": [gone.id]})
        assert [hit[0] for hit in indexed_db.vector_search("Document", "embedding", Y, 4)] == [keep.id]

    def test_rebuild_restores_every_vector_after_deletes(self, indexed_db: grafeo.GrafeoDB) -> None:
        # Removing a vector can leave other vectors unreachable in Grafeo's HNSW graph:
        # with this data, 0.5.44 finds 3 of the 4 remaining vectors before the rebuild.
        # GrafeoGraphVectorStore._ensure_index rebuilds the index after deletes and
        # relies on the rebuild finding them all.
        vectors = [[b / 255.0 for b in hashlib.sha256(f"document {i}".encode()).digest()[:4]] for i in range(5)]
        nodes = [indexed_db.create_node(["Document"], {"embedding": vector}).id for vector in vectors]
        indexed_db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": [nodes[1]]})
        indexed_db.rebuild_vector_index("Document", "embedding")
        hits = {node_id for node_id, _ in indexed_db.vector_search("Document", "embedding", X, 10)}
        assert hits == set(nodes) - {nodes[1]}

    def test_create_again_rebuilds_without_error(self, indexed_db: grafeo.GrafeoDB) -> None:
        # Every GrafeoGraphVectorStore.__init__ calls create_vector_index, also on a
        # reopened database whose index survived.
        node = indexed_db.create_node(["Document"], {"doc_id": "a", "embedding": X})
        indexed_db.create_vector_index("Document", "embedding", dimensions=DIMS, metric="cosine")
        assert [hit[0] for hit in indexed_db.vector_search("Document", "embedding", X, 4)] == [node.id]

    def test_create_with_other_dimensions_raises(self, db: grafeo.GrafeoDB) -> None:
        db.create_node(["Document"], {"doc_id": "a", "embedding": X})
        with pytest.raises(RuntimeError):
            db.create_vector_index("Document", "embedding", dimensions=DIMS * 2, metric="cosine")

    def test_index_rejects_wrong_dimensions_on_write(self, indexed_db: grafeo.GrafeoDB) -> None:
        with pytest.raises(grafeo.GrafeoError):
            indexed_db.create_node(["Document"], {"doc_id": "a", "embedding": [1.0, 0.0]})


class TestVectorSearch:
    def test_results_are_id_distance_pairs_closest_first(self, indexed_db: grafeo.GrafeoDB) -> None:
        near = indexed_db.create_node(["Document"], {"doc_id": "near", "embedding": [1.0, 0.1, 0.0, 0.0]})
        far = indexed_db.create_node(["Document"], {"doc_id": "far", "embedding": Y})
        hits = indexed_db.vector_search("Document", "embedding", X, 2)
        assert [hit[0] for hit in hits] == [near.id, far.id]
        assert all(isinstance(node_id, int) and isinstance(dist, float) for node_id, dist in hits)

    def test_cosine_distance_scale(self, indexed_db: grafeo.GrafeoDB) -> None:
        # the stores report score = 1 - distance
        indexed_db.create_node(["Document"], {"doc_id": "same", "embedding": X})
        indexed_db.create_node(["Document"], {"doc_id": "orthogonal", "embedding": Y})
        distances = sorted(dist for _, dist in indexed_db.vector_search("Document", "embedding", X, 2))
        assert distances[0] == pytest.approx(0.0, abs=1e-5)
        assert distances[1] == pytest.approx(1.0, abs=1e-5)

    def test_k_zero_returns_nothing(self, indexed_db: grafeo.GrafeoDB) -> None:
        indexed_db.create_node(["Document"], {"doc_id": "a", "embedding": X})
        assert indexed_db.vector_search("Document", "embedding", X, 0) == []


class TestFilters:
    """README: filters are exact-match; _utils.matches_filter mirrors these semantics."""

    @pytest.fixture(autouse=True)
    def _populate(self, indexed_db: grafeo.GrafeoDB) -> None:
        rows = [
            {"doc_id": "a", "embedding": X, "tag": "x", "year": 2024, "score": 0.5, "active": True},
            {"doc_id": "b", "embedding": Y, "tag": "y", "year": 2020, "score": 1.5, "active": False},
        ]
        indexed_db.upsert_nodes(["Document"], rows, key="doc_id", replace=True)

    def _hits(self, db: grafeo.GrafeoDB, filters: dict) -> set[int]:
        return {node_id for node_id, _ in db.vector_search("Document", "embedding", X, 10, filters=filters)}

    @pytest.mark.parametrize(
        "filters",
        [{"tag": "x"}, {"year": 2024}, {"score": 0.5}, {"active": True}, {"tag": "x", "year": 2024}],
        ids=["str", "int", "float", "bool", "and"],
    )
    def test_exact_match(self, indexed_db: grafeo.GrafeoDB, filters: dict) -> None:
        assert self._hits(indexed_db, filters) == {_gid(indexed_db, "a")}

    @pytest.mark.parametrize(
        "filters",
        [{"year": 2024.0}, {"active": 1}, {"tag": "x", "year": 2020}, {"missing": "x"}],
        ids=["int-vs-float", "bool-vs-int", "and-mismatch", "missing-key"],
    )
    def test_no_match(self, indexed_db: grafeo.GrafeoDB, filters: dict) -> None:
        assert self._hits(indexed_db, filters) == set()

    @pytest.mark.parametrize("value", [["x"], {"k": "x"}, None], ids=["list", "dict", "none"])
    def test_compound_or_null_value_matches_nothing(self, indexed_db: grafeo.GrafeoDB, value: object) -> None:
        assert self._hits(indexed_db, {"tag": value}) == set()

    def test_empty_filter_is_no_filter(self, indexed_db: grafeo.GrafeoDB) -> None:
        assert len(self._hits(indexed_db, {})) == 2

    def test_mmr_search_applies_filters(self, indexed_db: grafeo.GrafeoDB) -> None:
        hits = indexed_db.mmr_search("Document", "embedding", X, 2, fetch_k=10, lambda_mult=0.5, filters={"tag": "y"})
        assert [node_id for node_id, _ in hits] == [_gid(indexed_db, "b")]

    def test_mmr_search_returns_at_most_k(self, indexed_db: grafeo.GrafeoDB) -> None:
        hits = indexed_db.mmr_search("Document", "embedding", X, 1, fetch_k=10, lambda_mult=0.5)
        assert len(hits) == 1


# ── Edges, deletes and queries ────────────────────────────────────────────────


class TestEdges:
    def test_create_edge_with_and_without_properties(self, db: grafeo.GrafeoDB) -> None:
        a = db.create_node(["Document"], {"doc_id": "a"})
        b = db.create_node(["Document"], {"doc_id": "b"})
        plain = db.create_edge(a.id, b.id, "LINKS_TO", None)
        weighted = db.create_edge(a.id, b.id, "CITES", {"weight": 0.5})
        assert db.get_edge(plain.id).properties() == {}
        assert db.get_edge(weighted.id).properties() == {"weight": 0.5}

    def test_find_edge_by_endpoints_and_type(self, db: grafeo.GrafeoDB) -> None:
        # _utils.merge_edge runs this query to make link creation idempotent
        a = db.create_node(["Document"], {"doc_id": "a"})
        b = db.create_node(["Document"], {"doc_id": "b"})
        cites = db.create_edge(a.id, b.id, "CITES")
        db.create_edge(a.id, b.id, "LINKS_TO")
        db.create_edge(b.id, a.id, "CITES")
        query = "MATCH (s)-[r]->(t) WHERE id(s) = $src AND id(t) = $dst AND type(r) = $type RETURN id(r) AS rid"
        rows = list(db.execute(query, {"src": a.id, "dst": b.id, "type": "CITES"}))
        assert rows == [{"rid": cites.id}]

    def test_set_edge_property(self, db: grafeo.GrafeoDB) -> None:
        a = db.create_node(["Document"], {"doc_id": "a"})
        edge = db.create_edge(a.id, a.id, "SELF", {"weight": 1})
        db.set_edge_property(edge.id, "weight", 2)
        assert db.get_edge(edge.id).properties() == {"weight": 2}


class TestDelete:
    def test_delete_node_refuses_a_node_with_edges(self, db: grafeo.GrafeoDB) -> None:
        # Why GrafeoGraphVectorStore.delete uses DETACH DELETE
        a = db.create_node(["Document"], {"doc_id": "a"})
        b = db.create_node(["Document"], {"doc_id": "b"})
        db.create_edge(a.id, b.id, "LINKS_TO")
        with pytest.raises(grafeo.GrafeoError, match="DETACH DELETE"):
            db.delete_node(a.id)

    def test_detach_delete_by_id_list_reports_counters(self, db: grafeo.GrafeoDB) -> None:
        a = db.create_node(["Document"], {"doc_id": "a"})
        b = db.create_node(["Document"], {"doc_id": "b"})
        c = db.create_node(["Document"], {"doc_id": "c"})
        db.create_edge(a.id, b.id, "LINKS_TO")
        db.create_edge(c.id, a.id, "LINKS_TO")
        result = db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": [a.id]})
        assert result.counters["nodes_deleted"] == 1
        assert result.counters["edges_deleted"] == 2
        assert (db.node_count, db.edge_count) == (2, 0)

    def test_detach_delete_of_nothing(self, db: grafeo.GrafeoDB) -> None:
        result = db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": []})
        assert result.counters["nodes_deleted"] == 0


class TestQueries:
    def test_rows_are_dicts_named_by_alias(self, db: grafeo.GrafeoDB) -> None:
        db.create_node(["Document"], {"doc_id": "a"})
        result = db.execute("MATCH (n:Document) RETURN n.doc_id AS doc_id, count(n) AS n")
        assert result.columns == ["doc_id", "n"]
        assert list(result) == [{"doc_id": "a", "n": 1}]

    def test_unaliased_columns_are_named_after_the_expression(self, db: grafeo.GrafeoDB) -> None:
        # README examples read rows such as row["p.name"]
        db.create_node(["Person"], {"name": "Alix"})
        rows = list(db.execute("MATCH (p:Person) RETURN p.name, count(p)"))
        assert rows == [{"p.name": "Alix", "count(p)": 1}]

    def test_scalar(self, db: grafeo.GrafeoDB) -> None:
        assert db.execute("MATCH (n:Document) RETURN count(n) AS n").scalar() == 0

    def test_counters_report_writes(self, db: grafeo.GrafeoDB) -> None:
        # GrafeoGraphStore.query clears its schema cache when a query writes
        write = db.execute("CREATE (:Person {name: 'Alix'})")
        assert write.counters["nodes_created"] == 1
        read = db.execute("MATCH (p:Person) RETURN p.name AS name")
        assert not any(read.counters.values())
        expected = {"nodes_created", "nodes_deleted", "edges_created", "edges_deleted", "properties_set"}
        assert expected <= set(read.counters)

    def test_traversal_query_shortest_distance_closest_first(self) -> None:
        # GrafeoGraphVectorStore._expand: one row per neighbor, at its shortest distance,
        # sorted, also when the engine shuffles unordered rows; a cycle back to the
        # source is reported (the store skips it).
        db = grafeo.GrafeoDB(shuffle_unordered=True)
        ids = {name: db.create_node(["Document"], {"doc_id": name}).id for name in "sabcx"}
        for src, dst in [("s", "a"), ("a", "b"), ("b", "c"), ("s", "x"), ("s", "b"), ("c", "s")]:
            db.create_edge(ids[src], ids[dst], "LINKS_TO")
        query = (
            "MATCH p = (src:Document)-[*1..3]->(nb:Document) WHERE id(src) = $src "
            "RETURN id(nb) AS nid, min(length(p)) AS hops ORDER BY hops, nid"
        )
        expected = [(ids["a"], 1), (ids["b"], 1), (ids["x"], 1), (ids["c"], 2), (ids["s"], 3)]
        for _ in range(5):
            rows = [(row["nid"], row["hops"]) for row in db.execute(query, {"src": ids["s"]})]
            assert rows == sorted(expected, key=lambda row: (row[1], row[0]))
        assert list(db.execute(query, {"src": ids["x"]})) == []


class TestSchema:
    def test_shape(self, db: grafeo.GrafeoDB) -> None:
        a = db.create_node(["Person"], {"node_id": "a"})
        db.create_edge(a.id, a.id, "KNOWS")
        schema = db.schema()
        assert {entry["name"]: entry["count"] for entry in schema["labels"]} == {"Person": 1}
        assert {entry["name"]: entry["count"] for entry in schema["edge_types"]} == {"KNOWS": 1}
        assert "node_id" in schema["property_keys"]

    def test_deleted_labels_stay_with_count_zero(self, db: grafeo.GrafeoDB) -> None:
        # GrafeoGraphStore.refresh_schema leaves out entries with a count of 0
        db.execute("CREATE (:Gone {node_id: 'g'})")
        db.execute("MATCH (n:Gone) DETACH DELETE n")
        counts = {entry["name"]: entry.get("count") for entry in db.schema()["labels"]}
        assert counts.get("Gone", 0) == 0


# ── Persistence ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", [".grafeo", ".db"], ids=["single-file", "wal-directory"])
class TestPersistence:
    def test_data_survives_reopen(self, tmp_path: Path, suffix: str) -> None:
        path = str(tmp_path / f"contract{suffix}")
        db = grafeo.GrafeoDB(path)
        a = db.create_node(["Document"], {"doc_id": "a", "embedding": X, "n": 1})
        b = db.create_node(["Document"], {"doc_id": "b", "embedding": Y})
        db.create_edge(a.id, b.id, "LINKS_TO", {"w": 0.5})
        db.close()

        db = grafeo.GrafeoDB(path)
        try:
            assert db.node_count == 2
            assert db.edge_count == 1
            props = db.get_nodes_by_label("Document", limit=1)[0][1]
            assert props["embedding"] in (X, Y)
            # indexes may or may not survive a reopen (WAL directories keep them only
            # from 0.6.0): the stores recreate them, which must work in both cases
            if not db.has_property_index("doc_id"):
                db.create_property_index("doc_id")
            db.create_vector_index("Document", "embedding", dimensions=DIMS, metric="cosine")
            node_id, _ = db.vector_search("Document", "embedding", X, 1)[0]
            assert db.get_node(node_id).properties()["doc_id"] == "a"
            assert db.find_nodes_by_property("doc_id", "b") != []
        finally:
            db.close()
