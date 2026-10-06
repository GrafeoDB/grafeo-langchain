"""GrafeoDB-backed vector store with graph traversal for LangChain."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

import grafeo
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from grafeo_langchain._utils import matches_filter, merge_edge

_LINK_KEY = "__graph_links__"
_LABEL = "Document"
_EMBEDDING = "embedding"
# Properties the store writes itself: metadata cannot overwrite them.
_RESERVED_KEYS = frozenset({"doc_id", "text", _EMBEDDING})

# One row per reachable document at its shortest distance, closest first.  The
# depth is part of the query text: variable-length bounds cannot be parameters.
_TRAVERSAL_QUERY = (
    "MATCH p = (src:Document)-[*1..{depth}]->(nb:Document) WHERE id(src) = $src "
    "RETURN id(nb) AS nid, min(length(p)) AS hops ORDER BY hops, nid"
)


class GrafeoGraphVectorStore(VectorStore):
    """Combined vector + graph store backed by GrafeoDB.

    Documents are stored as graph nodes with embedding vectors.  Explicit
    links between documents enable graph-enhanced retrieval: after vector
    search finds seed documents, graph traversal discovers structurally
    connected documents that may not be semantically similar.

    Args:
        embedding: LangChain ``Embeddings`` instance for encoding text.
        db_path: Path to a persistent database.  ``None`` for in-memory.  A path
            ending in ``.grafeo`` is a single-file database; any other path is a
            WAL-directory database.
        embedding_dimensions: Dimensionality of the embedding vectors.  Auto-detected
            from the model if not provided.  When given, validated against the model.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        db_path: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self._embedding = embedding

        # Auto-detect dimensions by probing the embedding model
        probe = self._embedding.embed_query("dimension probe")
        detected = len(probe)
        if embedding_dimensions is not None and embedding_dimensions != detected:
            msg = (
                f"embedding_dimensions={embedding_dimensions} does not match "
                f"the embedding model's actual output ({detected} dimensions)"
            )
            raise ValueError(msg)
        self._dims = detected

        self._db = grafeo.GrafeoDB(db_path) if db_path else grafeo.GrafeoDB()
        try:
            self._open()
        except BaseException:
            self._db.close()  # release the database file before raising
            raise

    def _open(self) -> None:
        if not self._db.has_property_index("doc_id"):
            self._db.create_property_index("doc_id")

        self._check_stored_dimensions()
        # Built once, from the documents already stored (a reopened database), or
        # empty.  Grafeo keeps it in sync with later writes; see _ensure_index.
        self._db.create_vector_index(_LABEL, _EMBEDDING, dimensions=self._dims, metric="cosine")
        self._index_stale = False

        # Counter for generated ids.  Ids in use are skipped (see _new_id).
        self._next_auto_id: int = self._db.execute("MATCH (n:Document) RETURN count(n) AS n").scalar() or 0

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

        A text whose id is already stored replaces that document: its text,
        embedding and metadata are overwritten, its links are kept, and new
        links are added once.  Texts without an id (``ids`` not given, or a
        ``None`` entry) get a generated id that no stored document uses.
        """
        texts_list = list(texts)
        metadatas_list = list(metadatas) if metadatas is not None else [{} for _ in texts_list]
        given_ids: list[str | None] = list(ids) if ids is not None else [None] * len(texts_list)
        if not len(texts_list) == len(metadatas_list) == len(given_ids):
            msg = (
                f"got {len(texts_list)} texts, {len(metadatas_list)} metadatas and {len(given_ids)} ids: "
                "the lengths must match"
            )
            raise ValueError(msg)
        if not texts_list:
            return []

        vectors = self._embedding.embed_documents(texts_list)
        for vector in vectors:
            self._check_dimensions(vector)
        doc_ids = self._assign_ids(given_ids)

        rows: dict[str, dict[str, Any]] = {}  # by doc_id: a repeated id keeps its last row
        pending_links: list[tuple[str, list[dict[str, Any]]]] = []
        for text, meta, doc_id, vector in zip(texts_list, metadatas_list, doc_ids, vectors, strict=True):
            meta = dict(meta)  # avoid mutating caller's dict
            links = meta.pop(_LINK_KEY, [])
            row: dict[str, Any] = {
                k: v for k, v in meta.items() if k not in _RESERVED_KEYS and isinstance(v, str | int | float | bool)
            }
            row.update({"doc_id": doc_id, "text": text, _EMBEDDING: vector})
            rows[doc_id] = row
            if links:
                pending_links.append((doc_id, links))

        stale = self._stale_properties(rows)
        # One statement for the batch: every row is written or none is.  A stored
        # doc_id is updated in place, so its node keeps its edges.  Merge, not
        # replace: replace=True removes and re-adds the embedding, and removing a
        # vector damages Grafeo's HNSW graph (see _ensure_index).
        self._db.upsert_nodes([_LABEL], list(rows.values()), key="doc_id", replace=False)
        for gid, keys in stale:
            for key in keys:
                self._db.remove_node_property(gid, key)

        # Links are created once every node of the batch exists, so they may
        # point forward within the batch.  Links to unknown ids are skipped.
        for doc_id, links in pending_links:
            for source_gid in self._db.find_nodes_by_property("doc_id", doc_id):
                for link in links:
                    target_id = link.get("target_id", "")
                    edge_type = link.get("type", "LINKS_TO")
                    for target_gid in self._db.find_nodes_by_property("doc_id", target_id):
                        merge_edge(self._db, source_gid, target_gid, edge_type, link.get("properties"))

        return doc_ids

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Find the *k* most similar documents by vector distance."""
        return [doc for _, doc in self._vector_hits(embedding, k, filter)]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Embed *query* and find similar documents."""
        return self.similarity_search_by_vector(self._embedding.embed_query(query), k=k, filter=filter, **kwargs)

    # ── Graph-enhanced retrieval ──────────────────────────────────────────────

    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Vector search followed by multi-hop graph traversal.

        Finds seed documents by vector similarity, then traverses graph
        links up to *depth* hops to discover connected documents, closest
        first.  *filter* applies to the traversed documents too.  Returns at
        most ``2 * k`` documents.
        """
        seeds = self._vector_hits(self._embedding.embed_query(query), k, filter)
        return self._expand(seeds, depth=depth, filter=filter, limit=k * 2)

    def mmr_traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """MMR-diversified graph traversal using Grafeo's native MMR search."""
        query_vec = self._embedding.embed_query(query)
        self._check_dimensions(query_vec)
        self._ensure_index()

        mmr_results = self._db.mmr_search(
            _LABEL,
            _EMBEDDING,
            query_vec,
            k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filters=filter,
        )
        seeds = self._to_hits(mmr_results)
        return self._expand(seeds, depth=depth, filter=filter, limit=k * 2)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        db_path: str | None = None,
        **kwargs: Any,
    ) -> GrafeoGraphVectorStore:
        store = cls(embedding=embedding, db_path=db_path, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
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
        if "ids" not in kwargs and any(doc.id for doc in documents):
            kwargs["ids"] = [doc.id for doc in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, db_path=db_path, **kwargs)

    # ── Delete ──────────────────────────────────────────────────────────────────

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete documents by their doc_id, with their graph links.

        Args:
            ids: List of document IDs to delete.

        Returns:
            True if any documents were deleted.
        """
        if not ids:
            return False
        node_ids = [gid for doc_id in ids for gid in self._db.find_nodes_by_property("doc_id", doc_id)]
        if not node_ids:
            return False
        # DETACH: Grafeo refuses to delete a node that still has edges.
        result = self._db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": node_ids})
        deleted = result.counters["nodes_deleted"] > 0
        if deleted:
            self._index_stale = True
        return deleted

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Rebuild the vector index before a search if documents were deleted.

        Grafeo (0.5.44, unchanged on 0.6.0) removes a deleted vector from its
        HNSW graph without reconnecting that vector's neighbors, so other
        documents can drop out of every search.  Writes need no rebuild.
        """
        if self._index_stale:
            self._db.rebuild_vector_index(_LABEL, _EMBEDDING)
            self._index_stale = False

    def _stale_properties(self, rows: dict[str, dict[str, Any]]) -> list[tuple[int, list[str]]]:
        """Return the properties of stored documents that the new rows no longer have."""
        stale: list[tuple[int, list[str]]] = []
        for doc_id, row in rows.items():
            for gid in self._db.find_nodes_by_property("doc_id", doc_id):
                node = self._db.get_node(gid)
                if node is None:
                    continue
                keys = [key for key in node.properties() if key not in row]
                if keys:
                    stale.append((gid, keys))
        return stale

    def _check_dimensions(self, vector: list[float]) -> None:
        # Grafeo panics (instead of raising) on a search vector of the wrong size.
        if len(vector) != self._dims:
            msg = f"got a {len(vector)}-dimensional embedding, but this store holds {self._dims}-dimensional embeddings"
            raise ValueError(msg)

    def _check_stored_dimensions(self) -> None:
        sample = self._db.get_nodes_by_label(_LABEL, limit=1)
        if not sample:
            return
        stored = sample[0][1].get(_EMBEDDING)
        if isinstance(stored, list) and len(stored) != self._dims:
            msg = (
                f"the database holds {len(stored)}-dimensional embeddings, "
                f"but the embedding model produces {self._dims} dimensions"
            )
            raise ValueError(msg)

    def _assign_ids(self, given_ids: list[str | None]) -> list[str]:
        taken = {doc_id for doc_id in given_ids if doc_id is not None}
        doc_ids: list[str] = []
        for doc_id in given_ids:
            if doc_id is None:
                doc_id = self._new_id(taken)
                taken.add(doc_id)
            doc_ids.append(doc_id)
        return doc_ids

    def _new_id(self, taken: set[str]) -> str:
        # Sequential ids, skipping ids in use: after deletes and a reopen the
        # counter can point at a stored document, which add_texts would replace.
        while True:
            candidate = str(self._next_auto_id)
            self._next_auto_id += 1
            if candidate not in taken and not self._db.find_nodes_by_property("doc_id", candidate):
                return candidate

    def _vector_hits(
        self,
        embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[tuple[int, Document]]:
        self._check_dimensions(embedding)
        self._ensure_index()
        return self._to_hits(self._db.vector_search(_LABEL, _EMBEDDING, embedding, k, filters=filter))

    def _to_hits(self, results: list[tuple[int, float]]) -> list[tuple[int, Document]]:
        """Turn ``(node id, distance)`` search results into ``(node id, Document)`` pairs."""
        hits: list[tuple[int, Document]] = []
        for node_id, distance in results:
            node = self._db.get_node(node_id)
            if node is None:
                continue
            hits.append((node_id, self._to_document(node.properties(), node_id, source="vector", score=1.0 - distance)))
        return hits

    def _expand(
        self,
        seeds: list[tuple[int, Document]],
        *,
        depth: int,
        filter: dict[str, Any] | None,
        limit: int,
    ) -> list[Document]:
        """Append the documents linked from *seeds*, closest first, up to *limit* in total."""
        result = [doc for _, doc in seeds]
        depth = operator.index(depth)  # goes into the query text, so it must be an integer
        if not seeds or depth < 1:
            return result

        query = _TRAVERSAL_QUERY.format(depth=depth)
        candidates: list[tuple[int, int]] = []
        for seed_id, _ in seeds:
            rows = self._db.execute(query, {"src": seed_id})
            candidates.extend((row["hops"], row["nid"]) for row in rows)
        candidates.sort(key=lambda candidate: candidate[0])  # stable: equal distances keep seed order

        seen = {seed_id for seed_id, _ in seeds}
        for _, node_id in candidates:
            if len(result) >= limit:
                break
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self._db.get_node(node_id)
            if node is None:
                continue
            props = node.properties()
            if matches_filter(props, filter):
                result.append(self._to_document(props, node_id, source="graph_traversal", score=None))
        return result

    @staticmethod
    def _to_document(props: dict[str, Any], node_id: int, *, source: str, score: float | None) -> Document:
        props = dict(props)
        text = props.pop("text", "")
        props.pop(_EMBEDDING, None)
        doc_id = props.pop("doc_id", str(node_id))
        return Document(
            id=doc_id,
            page_content=text,
            metadata={"id": doc_id, "source": source, "score": score, **props},
        )

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoGraphVectorStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
