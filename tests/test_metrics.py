import math

import pytest

from arlab.metrics import mrr, ndcg_at_k, recall_at_k

# xfail only while the stub raises NotImplementedError; once implemented, a wrong answer fails.
pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="metrics not written yet")

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_counts_relevant_in_top_k_over_all_relevant():
    assert recall_at_k(RANKED, {"b", "e", "z"}, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, {"b", "e", "z"}, k=5) == pytest.approx(2 / 3)


def test_recall_with_no_relevant_is_an_error_not_zero():
    # A query with no labels would silently drag the mean down; refuse it instead.
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), k=5)


def test_mrr_uses_first_relevant_rank_only():
    assert mrr(RANKED, {"c", "d"}) == pytest.approx(1 / 3)
    assert mrr(RANKED, {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=10) == pytest.approx(1.0)


def test_ndcg_uses_graded_gain_and_log2_discount():
    # gain = 2^grade - 1, discount = log2(rank + 1)
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=10) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_is_over_all_labels_not_just_retrieved():
    # "b" is relevant but never retrieved; it still counts in the ideal ranking.
    idcg = (2**2 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    assert ndcg_at_k(["a", "x"], {"a": 2, "b": 2}, k=2) == pytest.approx(3 / idcg)
