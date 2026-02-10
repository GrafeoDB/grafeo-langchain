"""Lightweight graph document types compatible with LangChain's GraphDocument API.

These mirror the types in ``langchain_community.graphs.graph_document`` so that
users don't need ``langchain-community`` as a runtime dependency.  Any object with
the same attributes (duck typing) is accepted by :class:`GrafeoGraphStore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class Node:
    """A node in a knowledge graph."""

    id: str
    type: str = "Node"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A directed relationship between two nodes."""

    source: Node
    target: Node
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDocument:
    """A document with extracted graph structure (nodes + relationships)."""

    nodes: list[Node]
    relationships: list[Relationship]
    source: Document
