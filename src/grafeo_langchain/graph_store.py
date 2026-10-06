"""GrafeoDB-backed graph store for LangChain.

Implements the same interface as ``langchain_community.graphs.GraphStore``
without requiring ``langchain-community`` as a dependency.
"""

from __future__ import annotations

import hashlib
from typing import Any

import grafeo
from langchain_core.documents import Document

from grafeo_langchain._utils import merge_edge
from grafeo_langchain.graph_document import GraphDocument, Node


class GrafeoGraphStore:
    """Knowledge graph store backed by GrafeoDB.

    Stores LLM-extracted triples (nodes + relationships) as native Grafeo
    graph elements.  Supports GQL and Cypher queries.

    Args:
        db_path: Path to a persistent database.  ``None`` for in-memory.  A path
            ending in ``.grafeo`` is a single-file database; any other path is a
            WAL-directory database.
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
        """Return the graph schema as a human-readable string."""
        schema = self.get_structured_schema
        parts: list[str] = []
        if schema.get("labels"):
            parts.append(f"Node labels: {', '.join(schema['labels'])}")
        if schema.get("edge_types"):
            parts.append(f"Relationship types: {', '.join(schema['edge_types'])}")
        if schema.get("property_keys"):
            parts.append(f"Properties: {', '.join(schema['property_keys'])}")
        return "\n".join(parts) if parts else "Empty graph"

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        """Return the structured graph schema."""
        if self._schema is None:
            self.refresh_schema()
        return self._schema or {}

    def query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a GQL/Cypher query and return results as dicts.

        A query that writes to the graph clears the cached schema.  Writes made
        through ``client`` do not: call ``refresh_schema()`` after those.
        """
        result = self._db.execute(query, params)
        if any(result.counters.values()):
            self._schema = None
        return list(result)

    def refresh_schema(self) -> None:
        """Refresh the cached schema from the database.

        Labels and relationship types without any node or edge left are left out,
        and every list is sorted, so the schema text is stable for LLM prompts.
        """
        raw = self._db.schema()
        self._schema = {
            "labels": _names_in_use(raw.get("labels", [])),
            "edge_types": _names_in_use(raw.get("edge_types", [])),
            "property_keys": sorted(raw.get("property_keys", [])),
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
            **{k: v for k, v in doc.metadata.items() if isinstance(v, str | int | float | bool)},
        }
        return self._db.create_node(["SourceDocument"], props).id

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoGraphStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _names_in_use(entries: list[dict[str, Any]]) -> list[str]:
    # Grafeo keeps a label or edge type in its schema, with a count of 0, after
    # its last node or edge is deleted.
    return sorted(entry["name"] for entry in entries if entry.get("count", 1) > 0)


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
