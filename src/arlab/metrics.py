import math


def _dedupe(ranked: list[str]) -> list[str]:
    # A retriever that returns the same chunk twice must not get credit twice.
    return list(dict.fromkeys(ranked))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("query has no relevant labels")
    hits = sum(1 for doc in _dedupe(ranked)[:k] if doc in relevant)
    return hits / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        raise ValueError("query has no relevant labels")
    for rank, doc in enumerate(_dedupe(ranked), 1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def _dcg(grades: list[int]) -> float:
    # Exponential gain rewards a grade-2 chunk at the top much more than a grade-1 chunk.
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, 1))


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    ideal = _dcg(sorted((g for g in grades.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        raise ValueError("query has no relevant labels")
    actual = _dcg([grades.get(doc, 0) for doc in _dedupe(ranked)[:k]])
    return actual / ideal
