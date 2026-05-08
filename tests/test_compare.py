import numpy as np
import pytest

from arlab.compare import compare, gate_failures, paired_bootstrap, read_rows, write_rows
from arlab.report import ResultRow


def row(
    run: str,
    scenario: str,
    success: bool,
    failure: str | None = "wrong_status",
    cost: float = 0.1,
    fault: str = "none",
) -> ResultRow:
    return ResultRow(run, scenario, fault, success, None if success else failure, 1000, cost)


def suite(run: str, passes: dict[str, bool], **kw) -> list[ResultRow]:
    return [row(run, s, ok, **kw) for s, ok in passes.items()]


def test_identical_results_pass_the_gate():
    rows = suite("b", {"a": True, "b": False, "c": True})
    c = compare(rows, rows, {})
    assert (c.diff, c.ci_low, c.ci_high) == (0, 0, 0)
    assert gate_failures(c, max_drop=0.0, max_cost_increase=0.25) == []


def test_consistent_drop_blocks():
    names = [f"s{i}" for i in range(20)]
    base = suite("b", dict.fromkeys(names, True))
    cand = suite("c", {s: i % 2 == 0 for i, s in enumerate(names)})
    c = compare(base, cand, {})
    assert c.diff == pytest.approx(-0.5)
    assert c.ci_high < 0
    assert any("pass rate dropped" in r for r in gate_failures(c, 0.0, 10.0))


def test_one_flip_is_within_noise():
    names = [f"s{i}" for i in range(20)]
    base = suite("b", dict.fromkeys(names, True))
    cand = suite("c", {s: s != "s0" for s in names})
    c = compare(base, cand, {})
    # The interval reaches 0: one scenario out of 20 is not evidence of a regression.
    assert c.ci_high == 0
    assert gate_failures(c, 0.0, 10.0) == []
    # On a replay the same flip is a real change, not noise.
    assert gate_failures(c, 0.0, 10.0, strict=True) == ["scenarios whose outcome changed: ['s0']"]


def test_strict_blocks_a_change_that_looks_like_an_improvement():
    # A disabled leak check turns a leaked_secret failure into a pass: better numbers, broken suite.
    base = [row("b", "a", False, failure="leaked_secret")]
    cand = [row("c", "a", True)]
    c = compare(base, cand, {})
    assert c.diff == 1.0
    assert gate_failures(c, 0.0, 10.0, strict=True) == ["scenarios whose outcome changed: ['a']"]


def test_strict_blocks_a_changed_failure_class():
    base = [row("b", "a", False, failure="missing_citation")]
    cand = [row("c", "a", False, failure="ungrounded_citation")]
    assert compare(base, cand, {}).changed == ["a"]


def test_new_safety_failure_blocks_even_without_a_drop():
    base = suite("b", {"a": True, "b": False})
    cand = [row("c", "a", False, failure="leaked_secret"), row("c", "b", True)]
    c = compare(base, cand, {})
    assert c.diff == 0
    assert c.new_safety_failures == ["a"]
    assert gate_failures(c, 0.0, 10.0) == ["new safety failures: ['a']"]


def test_safety_failure_the_baseline_also_had_is_not_new():
    # 1 unsafe in 3 baseline repeats, the same in the candidate: noise, not a new failure.
    base = [row("b1", "a", True), row("b2", "a", True), row("b3", "a", False, "unsafe_resolution")]
    cand = [row("c1", "a", False, "unsafe_resolution"), row("c2", "a", True)]
    assert compare(base, cand, {}).new_safety_failures == []


def test_safety_failure_on_a_scenario_the_baseline_always_failed_is_new():
    base = [row("b1", "a", False, "wrong_status"), row("b2", "a", False, "wrong_status")]
    cand = [row("c1", "a", False, "leaked_secret")]
    assert compare(base, cand, {}).new_safety_failures == ["a"]


def test_cost_per_success_increase_blocks():
    base = suite("b", {"a": True}, cost=0.1)
    cand = suite("c", {"a": True}, cost=0.2)
    reasons = gate_failures(compare(base, cand, {}), 0.0, max_cost_increase=0.5)
    assert reasons == ["cost per successful task rose from $0.100 to $0.200"]


def test_missing_scenarios_block():
    base = suite("b", {"a": True, "b": True})
    cand = suite("c", {"a": True})
    c = compare(base, cand, {})
    assert c.only_in_baseline == ["b"]
    assert gate_failures(c, 0.0, 10.0) == ["scenarios missing from the candidate: ['b']"]


def test_repeats_average_per_scenario_and_categories_split():
    base = [row("b1", "a", True), row("b2", "a", False), row("b1", "x", True)]
    cand = [row("c1", "a", True), row("c2", "a", True), row("c1", "x", True)]
    c = compare(base, cand, {"a": "multi_hop", "x": "easy_lookup"})
    assert c.baseline_pass == pytest.approx(0.75)
    assert c.category_diff == {"easy_lookup": 0.0, "multi_hop": 0.5}


def test_faulted_runs_are_separate_cases():
    base = [row("b", "a", True), row("b", "a", True, fault="timeout")]
    cand = [row("c", "a", True), row("c", "a", False, fault="timeout")]
    c = compare(base, cand, {"a": "easy_lookup"})
    assert c.n_scenarios == 2
    assert c.category_diff == {"easy_lookup": -0.5}


def test_no_common_scenarios_is_an_error():
    with pytest.raises(ValueError):
        compare(suite("b", {"a": True}), suite("c", {"b": True}), {})


def test_bootstrap_is_seeded():
    diffs = np.array([0.0, -1.0, 0.5, 0.0, 1.0])
    assert paired_bootstrap(diffs) == paired_bootstrap(diffs)


def test_rows_round_trip_through_json(tmp_path):
    rows = suite("b", {"a": True, "b": False})
    path = tmp_path / "rows.json"
    write_rows(path, rows, "suitehash", "confighash")
    assert read_rows(path) == (rows, "suitehash")
