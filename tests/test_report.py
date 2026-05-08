import pytest

from arlab.report import ResultRow, markdown, summarize


def row(run: str, scenario: str, success: bool, cost: float = 0.1) -> ResultRow:
    return ResultRow(
        run, scenario, "none", success, None if success else "wrong_status", 1000, cost
    )


def test_summary_per_run_category_and_flaky():
    rows = [
        row("r1", "a", True),
        row("r1", "b", False),
        row("r2", "a", True),
        row("r2", "b", True),
    ]
    summary = summarize(rows, {"a": "easy_lookup", "b": "multi_hop"})
    assert summary.repeat_pass_rates == [0.5, 1.0]
    assert summary.category_pass == {"easy_lookup": (2, 2), "multi_hop": (1, 2)}
    assert summary.flaky == ["b"]
    assert summary.failure_classes == {"wrong_status": 1}
    assert summary.cost_per_repeat == pytest.approx(0.2)
    assert summary.cost_per_success == pytest.approx(0.4 / 3)
    assert "| multi_hop | 1 | 2 | 0.50 |" in markdown(summary)


def test_no_successes_has_no_cost_per_success():
    assert summarize([row("r1", "a", False)], {}).cost_per_success is None


def test_empty_batch_is_an_error():
    with pytest.raises(ValueError):
        summarize([], {})


def test_repeats_pair_attempts_across_partial_runs():
    # Two runs of scenario a, then two runs that only cover b: still two repeats of each.
    rows = [row("r1", "a", True), row("r2", "a", False), row("r3", "b", True), row("r4", "b", True)]
    summary = summarize(rows, {})
    assert summary.repeat_pass_rates == [1.0, 0.5]
    assert summary.cost_per_repeat == pytest.approx(0.2)
