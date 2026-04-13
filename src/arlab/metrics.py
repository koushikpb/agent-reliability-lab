tests/test_metrics.py defines the expected behavior.


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    raise NotImplementedError


def mrr(ranked: list[str], relevant: set[str]) -> float:
    raise NotImplementedError


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    raise NotImplementedError
