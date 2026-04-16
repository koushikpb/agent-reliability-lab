import pytest

from arlab.fusion import rrf_fuse

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="RRF not written yet")


def test_rrf_scores_are_sum_of_reciprocal_ranks():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_output_is_sorted_best_first():
    ids = [doc for doc, _ in rrf_fuse([["a", "b"], ["b", "c"]])]
    assert ids[0] == "b"


def test_rrf_ties_break_deterministically_by_id():
    # Nondeterministic tie order would make two identical eval runs disagree.
    assert [d for d, _ in rrf_fuse([["b"], ["a"]])] == ["a", "b"]


def test_empty_ranking_contributes_nothing():
    # Lexical search often returns nothing for a paraphrased query; fusion must still work.
    assert rrf_fuse([[], ["a"]]) == rrf_fuse([["a"]])


def test_all_empty_rankings_fuse_to_empty():
    assert rrf_fuse([[], []]) == []
