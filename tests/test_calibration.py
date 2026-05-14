import pytest

from arlab.calibration import agreement, cohen_kappa, markdown


def test_perfect_agreement_is_kappa_one():
    assert cohen_kappa([(True, True), (False, False)]) == 1.0


def test_chance_level_agreement_is_kappa_zero():
    # Each rater says True half the time, independently: 50% agreement is all chance.
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert cohen_kappa(pairs) == 0.0


def test_kappa_matches_a_hand_computed_case():
    # 20 TT, 5 TF, 10 FT, 15 FF: po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4
    pairs = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert cohen_kappa(pairs) == pytest.approx(0.4)


def test_high_agreement_can_still_be_low_kappa():
    # 90% agreement, but the judge says True to everything: no better than chance.
    pairs = [(True, True)] * 9 + [(False, True)]
    assert cohen_kappa(pairs) == 0.0


def test_kappa_undefined_when_both_raters_use_one_label():
    assert cohen_kappa([(True, True), (True, True)]) is None


def test_agreement_counts_judge_errors():
    pairs = [(True, True), (True, False), (False, True), (False, True)]
    a = agreement("grounded", pairs)
    assert (a.judge_false_positives, a.judge_false_negatives) == (2, 1)
    assert "| grounded | 4 | 0.25 |" in markdown([a])
