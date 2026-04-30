# RRF uses ranks, not scores, because ts_rank_cd and cosine similarity are on different scales and
# a weighted score sum would need per-query normalization. k=60 is the value from Cormack et al. (2009).
RRF_K = 60


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one list of (id, score), best first."""
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(dict.fromkeys(ranking), 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    # Ties break by id so two identical eval runs give the same order. Exact ties are chunks at the
    # same rank in different lists, so any other tie-break would be just as arbitrary.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
