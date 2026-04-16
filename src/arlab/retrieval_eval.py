import time
from dataclasses import dataclass, field

import numpy as np
import psycopg

from arlab.ingest import embed_texts
from arlab.labels import LabeledQuery
from arlab.metrics import mrr, ndcg_at_k, recall_at_k
from arlab.retrieval import dense_search, exact_dense_search, search


@dataclass(frozen=True)
class RetrieverReport:
    retriever: str
    n_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    p50_ms: float
    p95_ms: float
    # Share of top-10 results nobody judged; None when no query has a judged list. High values
    # mean the metrics understate this retriever, because unjudged results count as misses.
    unjudged_at_10: float | None = None
    # Kept so two retrievers can be compared per query (paired test, bootstrap), not only by mean.
    per_query: list[dict[str, object]] = field(default_factory=list)


def evaluate(
    conn: psycopg.Connection, retriever: str, queries: list[LabeledQuery]
) -> RetrieverReport:
    # One untimed query first, so model loading doesn't land in the latency numbers.
    search(conn, retriever, queries[0].text, k=10)
    rows = []
    latencies = []
    per_query: list[dict[str, object]] = []
    unjudged = []
    for q in queries:
        # Timed from query text to ranked ids, so query embedding is included for dense retrievers.
        started = time.perf_counter()
        ranked = [h.chunk_id for h in search(conn, retriever, q.text, k=10)]
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = set(q.grades)
        scores = (
            recall_at_k(ranked, relevant, 5),
            recall_at_k(ranked, relevant, 10),
            mrr(ranked, relevant),
            ndcg_at_k(ranked, q.grades, 10),
        )
        rows.append(scores)
        per_query.append({"query_id": q.id, "ranked": ranked, "scores": list(scores)})
        if q.judged is not None:
            unjudged.append(sum(cid not in q.judged for cid in ranked) / max(len(ranked), 1))
    r5, r10, mean_rr, ndcg10 = (float(m) for m in np.mean(rows, axis=0))
    p50, p95 = (float(p) for p in np.percentile(latencies, [50, 95]))
    return RetrieverReport(
        retriever,
        len(queries),
        r5,
        r10,
        mean_rr,
        ndcg10,
        p50,
        p95,
        float(np.mean(unjudged)) if unjudged else None,
        per_query,
    )


def hnsw_recall(
    conn: psycopg.Connection, texts: list[str], k: int, ef_values: list[int]
) -> dict[int, float]:
    """Mean overlap of HNSW top-k with exact top-k, per ef_search value."""
    vecs = embed_texts(texts)
    exact = [set(exact_dense_search(conn, v, k)) for v in vecs]
    return {
        ef: float(
            np.mean(
                [
                    len(truth & {h.chunk_id for h in dense_search(conn, v, k, ef)}) / k
                    for v, truth in zip(vecs, exact, strict=True)
                ]
            )
        )
        for ef in ef_values
    }


def markdown_table(reports: list[RetrieverReport]) -> str:
    lines = [
        "| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms | Unjudged@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        unjudged = "n/a" if r.unjudged_at_10 is None else f"{r.unjudged_at_10:.2f}"
        lines.append(
            f"| {r.retriever} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} | {r.mrr:.3f}"
            f" | {r.ndcg_at_10:.3f} | {r.p50_ms:.1f} | {r.p95_ms:.1f} | {unjudged} |"
        )
    return "\n".join(lines)
