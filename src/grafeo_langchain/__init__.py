from __future__ import annotations

from grafeo_langchain.graph_document import GraphDocument, Node, Relationship
from grafeo_langchain.graph_store import GrafeoGraphStore
from grafeo_langchain.graph_vector_store import GrafeoGraphVectorStore

__version__ = "0.2.0"

__all__ = [
    "GrafeoGraphStore",
    "GrafeoGraphVectorStore",
    "GraphDocument",
    "Node",
    "Relationship",
    "__version__",
]

try:
    from grafeo_langchain.adapter import GrafeoAdapter

    __all__ += ["GrafeoAdapter"]
except ImportError:
    pass
