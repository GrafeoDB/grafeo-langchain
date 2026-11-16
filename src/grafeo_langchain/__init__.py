from __future__ import annotations

from grafeo_langchain.graph_document import GraphDocument, Node, Relationship
from grafeo_langchain.graph_store import GrafeoGraphStore
from grafeo_langchain.graph_vector_store import GrafeoGraphVectorStore
from grafeo_langchain.qa_chain import GrafeoQAChain

__version__ = "0.3.0"

__all__ = [
    "GrafeoGraphStore",
    "GrafeoGraphVectorStore",
    "GrafeoQAChain",
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
