"""GrafeoDB-backed graph store for LangChain.

Implements the same interface as ``langchain_community.graphs.GraphStore``
without requiring ``langchain-community`` as a dependency.
"""

from __future__ import annotations

import hashlib
from typing import Any

import grafeo
from langchain_core.documents import Document

from grafeo_langchain._utils import METADATA_TYPES, merge_edge
from grafeo_langchain.graph_document import GraphDocument, Node


class GrafeoGraphStore:
    """Knowledge graph store backed by GrafeoDB.

    Stores LLM-extracted triples (nodes + relationships) as native Grafeo
    graph elements.  Supports GQL and Cypher queries.

    Args:
        db_path: Path to a persistent database file, ``None`` for in-memory.  The
            database is one file at this path (``.grafeo`` is the usual extension),
            with its write-ahead log in ``<db_path>.wal/`` while it is open.  A
            database written by grafeo 0.5.x is migrated on open (see the README).
    """

    def __init__(self, *, db_path: str | None = None) -> None:
        self._db = grafeo.GrafeoDB(db_path) if db_path else grafeo.GrafeoDB()
        if not self._db.has_property_index("node_id"):
            self._db.create_property_index("node_id")
        self._schema: dict[str, Any] | None = None

    @property
    def client(self) -> grafeo.GrafeoDB:
        """Return the underlying GrafeoDB instance."""
        return self._db

    # ── GraphStore interface ──────────────────────────────────────────────────

    @property
    def get_schema(self) -> str:
        """Return the graph schema as text for an LLM prompt.

        Lists each node label and relationship type with its properties and their
        types, then the relationship patterns, such as ``(:Person)-[:WORKS_AT]->(:Company)``.
        Patterns that differ only in a label are merged (``(:Company|Organization)``), and the
        ``MENTIONED_IN`` edges to source documents take one line, ``()-[:MENTIONED_IN]->(:SourceDocument)``.
        """
        schema = self.get_structured_schema
        if not schema.get("labels") and not schema.get("edge_types"):
            return "Empty graph"
        parts = ["Node labels and properties:"]
        parts += [_describe(label, schema["node_props"].get(label, [])) for label in schema["labels"]]
        if schema["edge_types"]:
            parts.append("Relationship types and properties:")
            parts += [
                _describe(edge_type, schema["rel_props"].get(edge_type, [])) for edge_type in schema["edge_types"]
            ]
            parts.append("Relationships:")
            parts += _patterns(schema["relationships"])
        return "\n".join(parts)

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        """Return the structured graph schema.

        Keys: ``labels``, ``edge_types`` and ``property_keys`` (sorted names), and, in
        the shape ``langchain-neo4j`` uses, ``node_props`` and ``rel_props`` (name to a
        list of ``{"property", "type"}``) and ``relationships`` (``{"start", "type", "end"}``).
        """
        if self._schema is None:
            self.refresh_schema()
        return self._schema or {}

    def query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute a GQL/Cypher query and return results as dicts.

        A query that writes to the graph clears the cached schema.  Writes made
        through ``client`` do not: call ``refresh_schema()`` after those.

        Args:
            query: The query text.
            params: Query parameters, referenced as ``$name`` in the query.
            read_only: Run the query in a transaction that is always rolled back, and
                raise ``ValueError`` if it wrote anything: the graph stays unchanged.
                Use it for queries from an LLM or a user.
        """
        if read_only:
            return self._query_read_only(query, params)
        result = self._db.execute(query, params)
        if any(result.counters.values()):
            self._schema = None
        return list(result)

    def _query_read_only(self, query: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        tx = self._db.begin_transaction()
        try:
            result = tx.execute(query, params)
            writes = {name: count for name, count in result.counters.items() if count}
            if writes:
                msg = f"a read-only query may not write to the graph, and this one would have ({writes}): {query}"
                raise ValueError(msg)
            return list(result)
        finally:
            tx.rollback()

    def refresh_schema(self) -> None:
        """Refresh the cached schema from the database.

        Labels and relationship types without any node or edge left are left out,
        and every list is sorted, so the schema text is stable for LLM prompts.
        """
        raw = self._db.schema()
        labels = raw.get("labels", [])
        edge_types = raw.get("edge_types", [])
        self._schema = {
            "labels": sorted(entry["name"] for entry in labels),
            "edge_types": sorted(entry["name"] for entry in edge_types),
            "property_keys": sorted(raw.get("property_keys", [])),
            "node_props": {entry["name"]: _properties(entry) for entry in labels},
            "rel_props": {entry["name"]: _properties(entry) for entry in edge_types},
            "relationships": sorted(
                (
                    {"start": endpoint["source"], "type": entry["name"], "end": endpoint["target"]}
                    for entry in edge_types
                    for endpoint in entry.get("endpoints", [])
                ),
                key=lambda rel: (rel["type"], rel["start"], rel["end"]),
            ),
        }

    def add_graph_documents(
        self,
        graph_documents: list[GraphDocument],
        *,
        include_source: bool = False,
    ) -> None:
        """Ingest LLM-extracted graph documents.

        Ingestion is idempotent: nodes are matched by id and relationships by
        source, target and type, so adding the same documents again updates
        their properties instead of duplicating them.

        Args:
            graph_documents: Documents containing nodes and relationships.
            include_source: If ``True``, store each source document as a
                ``SourceDocument`` node and link extracted entities to it.
        """
        for graph_doc in graph_documents:
            source_gid: int | None = None
            if include_source and graph_doc.source:
                source_gid = self._upsert_source_node(graph_doc.source)

            node_id_map: dict[str, int] = {}
            for node in graph_doc.nodes:
                gid = self._upsert_node(node)
                node_id_map[node.id] = gid
                if source_gid is not None:
                    merge_edge(self._db, gid, source_gid, "MENTIONED_IN")

            for rel in graph_doc.relationships:
                src_gid = node_id_map.get(rel.source.id)
                tgt_gid = node_id_map.get(rel.target.id)
                if src_gid is None:
                    src_gid = self._upsert_node(rel.source)
                    node_id_map[rel.source.id] = src_gid
                if tgt_gid is None:
                    tgt_gid = self._upsert_node(rel.target)
                    node_id_map[rel.target.id] = tgt_gid

                edge_type = rel.type.replace(" ", "_").replace("-", "_").upper()
                merge_edge(self._db, src_gid, tgt_gid, edge_type, rel.properties)

        self._schema = None  # invalidate cache

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _upsert_node(self, node: Node) -> int:
        matches = self._db.find_nodes_by_property("node_id", node.id)
        if matches:
            for key, value in node.properties.items():
                self._db.set_node_property(matches[0], key, value)
            return matches[0]

        label = node.type.replace(" ", "_").replace("-", "_")
        if not label.isidentifier():
            label = "Node"

        props = {"node_id": node.id, **node.properties}
        return self._db.create_node([label], props).id

    def _upsert_source_node(self, doc: Document) -> int:
        node_id = f"source_{_source_id(doc)}"
        matches = self._db.find_nodes_by_property("node_id", node_id)
        if matches:
            return matches[0]

        props: dict[str, Any] = {
            "node_id": node_id,
            "text": doc.page_content[:5000],
            **{k: v for k, v in doc.metadata.items() if isinstance(v, METADATA_TYPES)},
        }
        return self._db.create_node(["SourceDocument"], props).id

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoGraphStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _properties(entry: dict[str, Any]) -> list[dict[str, str]]:
    """Sorted ``{"property", "type"}`` dicts of a schema entry; a property with values of several types joins them."""
    props = entry.get("properties", [])
    return [
        {"property": prop["name"], "type": " | ".join(prop.get("types", []))}
        for prop in sorted(props, key=lambda prop: prop["name"])
    ]


def _patterns(relationships: list[dict[str, str]]) -> list[str]:
    """Relationship patterns for the schema text, one line per type and pair of label sets.

    Patterns that differ only in a start or an end label are merged into a label union, and
    the store's own ``MENTIONED_IN`` edges to source documents become one line.
    """
    starts: dict[tuple[str, str], set[str]] = {}  # (type, end) -> start labels
    mentioned = False
    for rel in relationships:
        if rel["type"] == "MENTIONED_IN" and rel["end"] == "SourceDocument":
            mentioned = True
        else:
            starts.setdefault((rel["type"], rel["end"]), set()).add(rel["start"])
    ends: dict[tuple[tuple[str, ...], str], set[str]] = {}  # (start labels, type) -> end labels
    for (rel_type, end), start_labels in starts.items():
        ends.setdefault((tuple(sorted(start_labels)), rel_type), set()).add(end)
    lines = sorted(
        (rel_type, f"(:{'|'.join(start_labels)})-[:{rel_type}]->(:{'|'.join(sorted(end_labels))})")
        for (start_labels, rel_type), end_labels in ends.items()
    )
    patterns = [line for _, line in lines]
    if mentioned:
        patterns.append("()-[:MENTIONED_IN]->(:SourceDocument)")
    return patterns


def _describe(name: str, props: list[dict[str, str]]) -> str:
    if not props:
        return name
    fields = ", ".join(f"{prop['property']}: {prop['type']}" for prop in props)
    return f"{name} {{{fields}}}"


def _source_id(doc: Document) -> str:
    """Return a stable id for a source document.

    Uses the ``id`` metadata key, then ``Document.id``, then an MD5 hash of the
    full text (as ``langchain-neo4j`` does), so the same document maps to the
    same ``SourceDocument`` node in every process.
    """
    meta_id = doc.metadata.get("id")
    if meta_id is not None:
        return str(meta_id)
    doc_id = getattr(doc, "id", None)
    if doc_id is not None:
        return str(doc_id)
    return hashlib.md5(doc.page_content.encode("utf-8"), usedforsecurity=False).hexdigest()
