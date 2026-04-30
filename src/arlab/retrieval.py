import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import psycopg

from arlab.fusion import RRF_K, rrf_fuse
from arlab.ingest import EMBED_MODEL, EMBED_REVISION, embed_texts

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
RETRIEVERS = ("fts", "dense", "hybrid", "hybrid_rerank")
# How deep each list goes before fusion or reranking. Deeper lists give fusion more overlap to work
# with but cost rerank time linearly.
CANDIDATES = 50
EF_SEARCH = 40

ScorePairs = Callable[[str, Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float


def fts_search(conn: psycopg.Connection, query: str, k: int) -> list[Hit]:
    # plainto_tsquery ANDs every term, and a ticket sentence almost never matches every term in one
    # chunk, so the terms are ORed and ts_rank_cd orders by how many match and how close together.
    # ts_rank_cd has no IDF or length saturation, so this is BM25-style, not BM25.
    # The ORed text is cast, not re-parsed: to_tsquery would stem the stems again
    # ('releas' -> 'relea'), and kind names like HorizontalPodAutoscaler would match nothing.
    rows = conn.execute(
        """
        SELECT id, ts_rank_cd(tsv, q) AS score
        FROM chunks, cast(replace(plainto_tsquery('english', %s)::text, '&', '|') AS tsquery) q
        WHERE tsv @@ q
        ORDER BY score DESC, id
        LIMIT %s
        """,
        (query, k),
    ).fetchall()
    return [Hit(chunk_id, score) for chunk_id, score in rows]


def dense_search(
    conn: psycopg.Connection, query_vec: np.ndarray, k: int, ef_search: int = EF_SEARCH
) -> list[Hit]:
    with conn.transaction():
        # ef_search must be >= k or HNSW returns fewer than k rows.
        conn.execute(f"SET LOCAL hnsw.ef_search = {max(ef_search, k):d}")
        rows = conn.execute(
            "SELECT id, 1 - (embedding <=> %s) AS score FROM chunks"
            " ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        ).fetchall()
    return [Hit(chunk_id, score) for chunk_id, score in rows]


def hybrid_search(conn: psycopg.Connection, query: str, query_vec: np.ndarray, k: int) -> list[Hit]:
    lexical = [h.chunk_id for h in fts_search(conn, query, CANDIDATES)]
    dense = [h.chunk_id for h in dense_search(conn, query_vec, CANDIDATES)]
    return [Hit(chunk_id, score) for chunk_id, score in rrf_fuse([lexical, dense])[:k]]


@functools.cache
def _rerank_model() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION)


def cross_encoder_scores(query: str, texts: Sequence[str]) -> Sequence[float]:
    return _rerank_model().predict([(query, t) for t in texts]).tolist()


def rerank(
    conn: psycopg.Connection,
    query: str,
    candidates: list[Hit],
    k: int,
    score_pairs: ScorePairs = cross_encoder_scores,
) -> list[Hit]:
    ids = [h.chunk_id for h in candidates]
    rows: dict[str, str] = dict(
        conn.execute(
            "SELECT id, heading || E'\\n' || text FROM chunks WHERE id = ANY(%s)", (ids,)
        ).fetchall()
    )
    missing = [i for i in ids if i not in rows]
    if missing:
        raise LookupError(f"candidates not in the index (re-ingested mid-run?): {missing}")
    scores = score_pairs(query, [rows[i] for i in ids])
    ranked = sorted(zip(ids, scores, strict=True), key=lambda p: (-p[1], p[0]))
    return [Hit(chunk_id, float(score)) for chunk_id, score in ranked[:k]]


def settings() -> dict[str, object]:
    """Retriever settings stored with each eval report, so two reports can be compared fairly."""
    return {
        "candidates": CANDIDATES,
        "ef_search": EF_SEARCH,
        "rrf_k": RRF_K,
        "embed_model": f"{EMBED_MODEL}@{EMBED_REVISION}",
        "rerank_model": f"{RERANK_MODEL}@{RERANK_REVISION}",
    }


def exact_dense_search(conn: psycopg.Connection, query_vec: np.ndarray, k: int) -> list[str]:
    """Brute-force nearest neighbors, the ground truth for HNSW recall."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        rows = conn.execute(
            "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (query_vec, k)
        ).fetchall()
    return [r[0] for r in rows]


def search(conn: psycopg.Connection, retriever: str, query: str, k: int) -> list[Hit]:
    if retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}, expected one of {RETRIEVERS}")
    if retriever == "fts":
        return fts_search(conn, query, k)
    query_vec = embed_texts([query])[0]
    if retriever == "dense":
        return dense_search(conn, query_vec, k)
    if retriever == "hybrid":
        return hybrid_search(conn, query, query_vec, k)
    return rerank(conn, query, hybrid_search(conn, query, query_vec, CANDIDATES), k)
