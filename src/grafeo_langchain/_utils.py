"""Helpers shared by the stores."""

from __future__ import annotations

from typing import Any

import grafeo

_FIND_EDGES = "MATCH (s)-[r]->(t) WHERE id(s) = $src AND id(t) = $dst AND type(r) = $type RETURN id(r) AS rid"


def merge_edge(
    db: grafeo.GrafeoDB,
    src: int,
    dst: int,
    edge_type: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Create an ``edge_type`` edge from ``src`` to ``dst`` unless one already exists.

    Works like Cypher ``MERGE``: every existing edge of that type between the two
    nodes gets ``properties`` set on it, so ingesting the same link twice keeps one edge.
    """
    rows = db.execute(_FIND_EDGES, {"src": src, "dst": dst, "type": edge_type})
    edge_ids = [row["rid"] for row in rows]
    if not edge_ids:
        db.create_edge(src, dst, edge_type, properties or None)
        return
    for edge_id in edge_ids:
        for key, value in (properties or {}).items():
            db.set_edge_property(edge_id, key, value)


def matches_filter(properties: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Check an exact-match metadata filter the way Grafeo's vector search filters do.

    A value matches only when its type matches too: ``1`` matches neither ``True`` nor ``1.0``.
    """
    if not filter:
        return True
    for key, expected in filter.items():
        if key not in properties:
            return False
        actual = properties[key]
        if type(actual) is not type(expected) or actual != expected:
            return False
    return True
