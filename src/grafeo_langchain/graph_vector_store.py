"""GrafeoDB-backed vector store with graph traversal for LangChain."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import grafeo
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from grafeo_langchain._utils import METADATA_TYPES, matches_filter, merge_edge, relevance

_LINK_KEY = "__graph_links__"
_LABEL = "Document"
_EMBEDDING = "embedding"
# Result metadata: 0 for a search hit, the number of hops for a linked document.  The
# name follows langchain-graph-retriever, which sets the same key on its results.
_DEPTH = "_depth"
# Properties the store writes itself, and result keys it sets: metadata cannot overwrite them.
_RESERVED_KEYS = frozenset({"doc_id", "text", _EMBEDDING, _DEPTH})

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
        db_path: Path to a persistent database file, ``None`` for in-memory.  The
            database is one file at this path (``.grafeo`` is the usual extension),
            with its write-ahead log in ``<db_path>.wal/`` while it is open.  A
            database written by grafeo 0.5.x is migrated on open (see the README).
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
        # empty.  Grafeo keeps it in sync with later writes and deletes.
        self._db.create_vector_index(_LABEL, _EMBEDDING, dimensions=self._dims, metric="cosine")

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
                k: v for k, v in meta.items() if k not in _RESERVED_KEYS and isinstance(v, METADATA_TYPES)
            }
            row.update({"doc_id": doc_id, "text": text, _EMBEDDING: vector})
            rows[doc_id] = row
            if links:
                pending_links.append((doc_id, links))

        # One statement for the batch: every row is written or none is.  A stored
        # doc_id is updated in place, so its node keeps its edges; replace=True
        # drops the metadata keys the new row no longer has.
        self._db.upsert_nodes([_LABEL], list(rows.values()), key="doc_id", replace=True)

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
        return [doc for _, doc, _ in self._vector_hits(embedding, k, filter)]

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

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Find the *k* most similar documents, each with its cosine distance (0 is identical)."""
        return [(doc, distance) for _, doc, distance in self._vector_hits(embedding, k, filter)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Embed *query* and find similar documents, each with its cosine distance (0 is identical).

        ``similarity_search_with_relevance_scores`` returns a score from 0 to 1 instead
        (``1 - distance / 2``), the value search results also carry in ``metadata["score"]``.
        """
        return self.similarity_search_with_score_by_vector(self._embedding.embed_query(query), k, filter=filter)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # LangChain's cosine function (1 - distance) goes below 0 for distances past 1.
        return relevance

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Pick *k* of the *fetch_k* closest documents, balancing relevance and diversity.

        Uses Grafeo's native MMR search.  *lambda_mult* runs from 0 (most diverse)
        to 1 (most relevant).
        """
        return [doc for _, doc, _ in self._mmr_hits(embedding, k, fetch_k, lambda_mult, filter)]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Embed *query* and pick *k* relevant, diverse documents (see ``max_marginal_relevance_search_by_vector``)."""
        return self.max_marginal_relevance_search_by_vector(
            self._embedding.embed_query(query), k, fetch_k, lambda_mult, filter=filter
        )

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Return the stored documents with these ids, in the order given.  Unknown ids are skipped."""
        docs: list[Document] = []
        for doc_id in ids:
            for node_id in self._db.find_nodes_by_property("doc_id", doc_id):
                node = self._db.get_node(node_id)
                if node is not None:
                    docs.append(self._to_document(node.properties(), node_id))
        return docs

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
        seeds = self._mmr_hits(self._embedding.embed_query(query), k, fetch_k, lambda_mult, filter)
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
        return result.counters["nodes_deleted"] > 0

    # ── Internals ─────────────────────────────────────────────────────────────

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
    ) -> list[tuple[int, Document, float]]:
        self._check_dimensions(embedding)
        return self._to_hits(self._db.vector_search(_LABEL, _EMBEDDING, embedding, k, filters=filter))

    def _mmr_hits(
        self,
        embedding: list[float],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: dict[str, Any] | None,
    ) -> list[tuple[int, Document, float]]:
        self._check_dimensions(embedding)
        results = self._db.mmr_search(
            _LABEL, _EMBEDDING, embedding, k, fetch_k=fetch_k, lambda_mult=lambda_mult, filters=filter
        )
        return self._to_hits(results)

    def _to_hits(self, results: list[tuple[int, float]]) -> list[tuple[int, Document, float]]:
        """Turn ``(node id, distance)`` search results into ``(node id, Document, distance)`` triples."""
        hits: list[tuple[int, Document, float]] = []
        for node_id, distance in results:
            node = self._db.get_node(node_id)
            if node is None:
                continue
            doc = self._to_document(node.properties(), node_id, depth=0, score=relevance(distance))
            hits.append((node_id, doc, distance))
        return hits

    def _expand(
        self,
        seeds: list[tuple[int, Document, float]],
        *,
        depth: int,
        filter: dict[str, Any] | None,
        limit: int,
    ) -> list[Document]:
        """Append the documents linked from *seeds*, closest first, up to *limit* in total."""
        result = [doc for _, doc, _ in seeds]
        depth = operator.index(depth)  # goes into the query text, so it must be an integer
        if not seeds or depth < 1:
            return result

        query = _TRAVERSAL_QUERY.format(depth=depth)
        candidates: list[tuple[int, int]] = []
        for seed_id, _, _ in seeds:
            rows = self._db.execute(query, {"src": seed_id})
            candidates.extend((row["hops"], row["nid"]) for row in rows)
        candidates.sort(key=lambda candidate: candidate[0])  # stable: equal distances keep seed order

        seen = {seed_id for seed_id, _, _ in seeds}
        for hops, node_id in candidates:
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
                result.append(self._to_document(props, node_id, depth=hops, score=None))
        return result

    @staticmethod
    def _to_document(
        props: dict[str, Any],
        node_id: int,
        *,
        depth: int | None = None,
        score: float | None = None,
    ) -> Document:
        """Build a Document from node properties; *depth* marks a retrieved one (0: search hit)."""
        props = dict(props)
        text = props.pop("text", "")
        props.pop(_EMBEDDING, None)
        doc_id = props.pop("doc_id", str(node_id))
        metadata: dict[str, Any] = {"id": doc_id}
        if depth is not None:
            # A stored "source" or "score" (a loader sets "source") wins over these;
            # _depth is reserved, so it always holds the store's value.
            metadata.update({"source": "vector" if depth == 0 else "graph_traversal", "score": score})
        metadata.update(props)
        if depth is not None:
            metadata[_DEPTH] = depth
        return Document(id=doc_id, page_content=text, metadata=metadata)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoGraphVectorStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
