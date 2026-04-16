import numpy as np
import pytest

from arlab import retrieval
from arlab.chunking import Chunk
from arlab.db import apply_schema
from arlab.ingest import load_chunks
from arlab.retrieval import Hit, dense_search, fts_search, rerank


def unit(i: int) -> np.ndarray:
    v = np.zeros(384, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def loaded(conn):
    apply_schema(conn)
    chunks = [
        Chunk(
            "pods#crash-0",
            "pods",
            "Debug Pods > CrashLoopBackOff",
            "Back-off restarting failed container.",
        ),
        Chunk("svc#dns-0", "svc", "Debug Services > DNS", "Check that the Service name resolves."),
        Chunk("pv#bind-0", "pv", "Volumes > Binding", "A PersistentVolumeClaim binds to a volume."),
        Chunk("hpa#scale-0", "hpa", "Autoscaling", "A HorizontalPodAutoscaler scales a release."),
    ]
    load_chunks(conn, chunks, np.stack([unit(0), unit(1), unit(2), unit(3)]))
    return conn


def test_fts_matches_any_term_not_all(loaded):
    hits = fts_search(loaded, "why is my container restarting with no logs", k=5)
    assert [h.chunk_id for h in hits] == ["pods#crash-0"]


def test_fts_finds_exact_error_token(loaded):
    assert fts_search(loaded, "CrashLoopBackOff", k=5)[0].chunk_id == "pods#crash-0"


def test_dense_orders_by_cosine_similarity(loaded):
    query = unit(1) + 0.5 * unit(2)
    hits = dense_search(loaded, query / np.linalg.norm(query), k=3)
    assert [h.chunk_id for h in hits][:2] == ["svc#dns-0", "pv#bind-0"]


def test_rerank_reorders_by_score_and_cuts_to_k(loaded):
    candidates = [Hit("pods#crash-0", 0.9), Hit("svc#dns-0", 0.5), Hit("pv#bind-0", 0.1)]

    def score_pairs(query, texts):
        return [0.0 if "Back-off" in t else 1.0 if "Service" in t else 0.5 for t in texts]

    hits = rerank(loaded, "q", candidates, k=2, score_pairs=score_pairs)
    assert [h.chunk_id for h in hits] == ["svc#dns-0", "pv#bind-0"]


def test_hybrid_passes_both_rankings_to_fusion(loaded, monkeypatch):
    seen = []

    def fake_fuse(rankings, k=60):
        seen.extend(rankings)
        return [(rankings[0][0], 1.0)]

    monkeypatch.setattr(retrieval, "rrf_fuse", fake_fuse)
    hits = retrieval.hybrid_search(loaded, "CrashLoopBackOff", unit(2), k=1)
    assert seen[0][0] == "pods#crash-0"
    assert seen[1][0] == "pv#bind-0"
    assert hits == [Hit("pods#crash-0", 1.0)]


def test_unknown_retriever_is_rejected(loaded):
    with pytest.raises(ValueError):
        retrieval.search(loaded, "bm25", "q", k=5)


@pytest.mark.parametrize("query", ["HorizontalPodAutoscaler", "release", "apiVersion release"])
def test_fts_does_not_stem_query_terms_twice(loaded, query):
    # Re-parsing plainto_tsquery output stems again: 'releas' -> 'relea' matches nothing.
    assert "hpa#scale-0" in [h.chunk_id for h in fts_search(loaded, query, k=5)]


def test_rerank_names_candidates_missing_from_the_index(loaded):
    with pytest.raises(LookupError, match="gone#x-0"):
        rerank(loaded, "q", [Hit("gone#x-0", 1.0)], k=1, score_pairs=lambda q, t: [0.0] * len(t))
