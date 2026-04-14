import time
from dataclasses import dataclass

import numpy as np
import psycopg

from arlab.labels import LabeledQuery
from arlab.metrics import mrr, ndcg_at_k, recall_at_k
from arlab.retrieval import search


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


def evaluate(
    conn: psycopg.Connection, retriever: str, queries: list[LabeledQuery]
) -> RetrieverReport:
    # One untimed query first, so model loading doesn't land in the latency numbers.
    search(conn, retriever, queries[0].text, k=10)
    rows = []
    latencies = []
    for q in queries:
        started = time.perf_counter()
        ranked = [h.chunk_id for h in search(conn, retriever, q.text, k=10)]
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = set(q.grades)
        rows.append(
            (
                recall_at_k(ranked, relevant, 5),
                recall_at_k(ranked, relevant, 10),
                mrr(ranked, relevant),
                ndcg_at_k(ranked, q.grades, 10),
            )
        )
    r5, r10, mean_rr, ndcg10 = (float(m) for m in np.mean(rows, axis=0))
    p50, p95 = (float(p) for p in np.percentile(latencies, [50, 95]))
    return RetrieverReport(retriever, len(queries), r5, r10, mean_rr, ndcg10, p50, p95)


def markdown_table(reports: list[RetrieverReport]) -> str:
    lines = [
        "| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.retriever} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} | {r.mrr:.3f}"
            f" | {r.ndcg_at_10:.3f} | {r.p50_ms:.1f} | {r.p95_ms:.1f} |"
        )
    return "\n".join(lines)
