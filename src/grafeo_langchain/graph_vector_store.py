"""GrafeoDB-backed vector store with graph traversal for LangChain."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import grafeo
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

_LINK_KEY = "__graph_links__"


class GrafeoGraphVectorStore(VectorStore):
    """Combined vector + graph store backed by GrafeoDB.

    Documents are stored as graph nodes with embedding vectors.  Explicit
    links between documents enable graph-enhanced retrieval: after vector
    search finds seed documents, graph traversal discovers structurally
    connected documents that may not be semantically similar.

    Args:
        embedding: LangChain ``Embeddings`` instance for encoding text.
        db_path: Path to a persistent database file.  ``None`` for in-memory.
        embedding_dimensions: Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        db_path: str | None = None,
        embedding_dimensions: int = 1536,
    ) -> None:
        self._embedding = embedding
        self._db = grafeo.GrafeoDB(db_path) if db_path else grafeo.GrafeoDB()
        self._dims = embedding_dimensions

        if not self._db.has_property_index("doc_id"):
            self._db.create_property_index("doc_id")

        self._index_dirty = False
        self._node_count = 0

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts with embeddings and optional graph links.

        Graph links are specified via a ``__graph_links__`` key in metadata::

            metadatas=[{
                "__graph_links__": [
                    {"target_id": "other_doc", "type": "RELATES_TO"},
                ],
            }]
        """
        texts_list = list(texts)
        metadatas = metadatas or [{} for _ in texts_list]
        ids = ids or [str(self._node_count + i) for i in range(len(texts_list))]

        vectors = self._embedding.embed_documents(texts_list)

        created_ids: list[str] = []
        for text, meta, doc_id, vec in zip(texts_list, metadatas, ids, vectors, strict=True):
            meta = dict(meta)  # avoid mutating caller's dict
            links = meta.pop(_LINK_KEY, [])

            props: dict[str, Any] = {"doc_id": doc_id, "text": text, "embedding": vec}
            for k, v in meta.items():
                if isinstance(v, str | int | float | bool):
                    props[k] = v

            node = self._db.create_node(["Document"], props)
            self._node_count += 1

            for link in links:
                target_id = link.get("target_id", "")
                edge_type = link.get("type", "LINKS_TO")
                for target_gid in self._db.find_nodes_by_property("doc_id", target_id):
                    self._db.create_edge(
                        node.id,
                        target_gid,
                        edge_type,
                        link.get("properties") or None,
                    )

            created_ids.append(doc_id)

        self._index_dirty = True
        return created_ids

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Find the *k* most similar documents by vector distance."""
        self._ensure_index()
        results = self._db.vector_search("Document", "embedding", embedding, k)
        return self._results_to_documents(results)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Embed *query* and find similar documents."""
        return self.similarity_search_by_vector(self._embedding.embed_query(query), k=k, **kwargs)

    # ── Graph-enhanced retrieval ──────────────────────────────────────────────

    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
    ) -> list[Document]:
        """Vector search followed by multi-hop graph traversal.

        Finds seed documents by vector similarity, then traverses graph
        links up to *depth* hops to discover connected documents.
        """
        seeds = self.similarity_search(query, k=k)
        if not seeds or depth < 1:
            return seeds

        # _results_to_documents always sets "id" in metadata, so doc_id is never empty
        # for vector-search seeds.  Seeds without an id are skipped during graph expansion.
        seen_ids: set[str] = {doc.metadata.get("id", "") for doc in seeds}
        result: list[Document] = list(seeds)

        for doc in seeds:
            doc_id = doc.metadata.get("id", "")
            if not doc_id:
                continue
            self._traverse_neighbors(doc_id, depth, seen_ids, result)

        return result[: k * 2]

    def mmr_traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """MMR-diversified graph traversal using Grafeo's native MMR search."""
        self._ensure_index()
        query_vec = self._embedding.embed_query(query)

        mmr_results = self._db.mmr_search(
            "Document", "embedding", query_vec, k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        seeds = self._results_to_documents(mmr_results)
        if not seeds or depth < 1:
            return seeds

        seen_ids: set[str] = {doc.metadata.get("id", "") for doc in seeds}
        result: list[Document] = list(seeds)

        for doc in seeds:
            doc_id = doc.metadata.get("id", "")
            if not doc_id:
                continue
            self._traverse_neighbors(doc_id, depth, seen_ids, result)

        return result[: k * 2]

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        db_path: str | None = None,
        **kwargs: Any,
    ) -> GrafeoGraphVectorStore:
        store = cls(embedding=embedding, db_path=db_path, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        *,
        db_path: str | None = None,
        **kwargs: Any,
    ) -> GrafeoGraphVectorStore:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, db_path=db_path, **kwargs)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Rebuild the vector index if documents have been added since the last build.

        Grafeo's HNSW index is currently a static snapshot — nodes added
        after ``create_vector_index()`` are not automatically indexed.
        This method rebuilds the index when dirty.

        TODO: Remove once Grafeo supports incremental vector index inserts.
        """
        if not self._index_dirty or self._node_count == 0:
            return
        with contextlib.suppress(RuntimeError):
            self._db.create_vector_index("Document", "embedding", dimensions=self._dims, metric="cosine")
        self._index_dirty = False

    def _traverse_neighbors(
        self,
        doc_id: str,
        depth: int,
        seen_ids: set[str],
        result: list[Document],
    ) -> None:
        # Variable-length paths require literal depth in GQL
        neighbors = self._db.execute(
            f"MATCH (src {{doc_id: $did}})-[*1..{depth}]->(nb:Document) RETURN nb",
            {"did": doc_id},
        )
        for nb_node in neighbors.nodes():
            props = nb_node.properties()
            nid = props.get("doc_id", str(nb_node.id))
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            text = props.pop("text", "")
            props.pop("embedding", None)
            props.pop("doc_id", None)
            result.append(Document(page_content=text, metadata={"id": nid, "source": "graph_traversal", **props}))

    def _results_to_documents(self, results: list[tuple[int, float]]) -> list[Document]:
        docs: list[Document] = []
        for node_id, distance in results:
            node = self._db.get_node(node_id)
            if node is None:
                continue
            props = node.properties()
            text = props.pop("text", "")
            props.pop("embedding", None)
            doc_id = props.pop("doc_id", str(node_id))
            docs.append(Document(page_content=text, metadata={"id": doc_id, "score": 1.0 - distance, **props}))
        return docs

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoGraphVectorStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
