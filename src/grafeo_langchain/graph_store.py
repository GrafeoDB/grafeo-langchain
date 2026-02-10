"""GrafeoDB-backed graph store for LangChain.

Implements the same interface as ``langchain_community.graphs.GraphStore``
without requiring ``langchain-community`` as a dependency.
"""

from __future__ import annotations

from typing import Any

import grafeo
from langchain_core.documents import Document

from grafeo_langchain.graph_document import GraphDocument, Node


class GrafeoGraphStore:
    """Knowledge graph store backed by GrafeoDB.

    Stores LLM-extracted triples (nodes + relationships) as native Grafeo
    graph elements.  Supports GQL and Cypher queries.

    Args:
        db_path: Path to a persistent database file.  ``None`` for in-memory.
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
        """Execute a GQL/Cypher query and return results as dicts."""
        result = self._db.execute(query, params)
        return list(result)

    def refresh_schema(self) -> None:
        """Refresh the cached schema from the database."""
        raw = self._db.schema()
        self._schema = {
            "labels": [entry["name"] for entry in raw.get("labels", [])],
            "edge_types": [entry["name"] for entry in raw.get("edge_types", [])],
            "property_keys": raw.get("property_keys", []),
        }

    def add_graph_documents(
        self,
        graph_documents: list[GraphDocument],
        *,
        include_source: bool = False,
    ) -> None:
        """Ingest LLM-extracted graph documents.

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
                    self._db.create_edge(gid, source_gid, "MENTIONED_IN")

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
                self._db.create_edge(src_gid, tgt_gid, edge_type, rel.properties or None)

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
        doc_id = doc.metadata.get("id", str(hash(doc.page_content[:200])))
        node_id = f"source_{doc_id}"
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
