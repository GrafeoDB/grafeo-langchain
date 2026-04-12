"""GrafeoAdapter for langchain-graph-retriever integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

from langchain_core.documents import Document
from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.langchain import LangchainAdapter

from grafeo_langchain.graph_vector_store import GrafeoGraphVectorStore


class GrafeoAdapter(LangchainAdapter[GrafeoGraphVectorStore]):
    """Adapter for ``GrafeoGraphVectorStore`` with ``langchain-graph-retriever``.

    Enables BFS, Eager, and other traversal strategies from
    ``langchain-graph-retriever`` on top of a ``GrafeoGraphVectorStore``.

    Usage::

        from grafeo_langchain import GrafeoGraphVectorStore
        from grafeo_langchain.adapter import GrafeoAdapter
        from langchain_graph_retriever import GraphRetriever

        store = GrafeoGraphVectorStore(embedding=my_embeddings)
        adapter = GrafeoAdapter(vector_store=store)
        retriever = GraphRetriever(store=adapter, edges=[("topic", "topic")])
        docs = retriever.invoke("my query")

    Args:
        vector_store: A ``GrafeoGraphVectorStore`` instance.
    """

    @override
    def _search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        store = self.vector_store
        store._ensure_index()
        results = store._db.vector_search(
            "Document",
            "embedding",
            embedding,
            k,
            filters=filter,
        )
        return self._nodes_to_docs(results, store)

    @override
    def _get(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        store = self.vector_store
        docs: list[Document] = []
        for doc_id in ids:
            for node_gid in store._db.find_nodes_by_property("doc_id", doc_id):
                node = store._db.get_node(node_gid)
                if node is None:
                    continue
                props = node.properties()
                text = props.pop("text", "")
                embedding = props.pop("embedding", None)
                props.pop("doc_id", None)

                if filter and not self._matches_filter(props, filter):
                    continue

                meta: dict[str, Any] = {
                    METADATA_EMBEDDING_KEY: embedding,
                    "id": doc_id,
                }
                for key, val in props.items():
                    if isinstance(val, str | int | float | bool):
                        meta[key] = val

                docs.append(Document(id=doc_id, page_content=text, metadata=meta))
        return docs

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _nodes_to_docs(
        results: list[tuple[int, float]],
        store: GrafeoGraphVectorStore,
    ) -> list[Document]:
        docs: list[Document] = []
        for node_id, distance in results:
            node = store._db.get_node(node_id)
            if node is None:
                continue
            props = node.properties()
            text = props.pop("text", "")
            embedding = props.pop("embedding", None)
            doc_id = props.pop("doc_id", str(node_id))

            meta: dict[str, Any] = {
                METADATA_EMBEDDING_KEY: embedding,
                "id": doc_id,
                "score": 1.0 - distance,
            }
            for key, val in props.items():
                if isinstance(val, str | int | float | bool):
                    meta[key] = val

            docs.append(Document(id=doc_id, page_content=text, metadata=meta))
        return docs

    @staticmethod
    def _matches_filter(props: dict[str, Any], filter: dict[str, Any]) -> bool:
        return all(props.get(k) == v for k, v in filter.items())
