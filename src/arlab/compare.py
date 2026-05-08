import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arlab.checks import SAFETY_CLASSES
from arlab.report import ResultRow


@dataclass(frozen=True)
class Comparison:
    n_scenarios: int
    baseline_pass: float
    candidate_pass: float
    # Mean of per-scenario differences (candidate - baseline) and its 95% paired-bootstrap interval.
    diff: float
    ci_low: float
    ci_high: float
    category_diff: dict[str, float]
    # Scenarios where the candidate has a safety class the baseline never had there. Keyed on the
    # class, not on pass/fail: a scenario the baseline always failed can still newly leak a secret,
    # and a baseline that was unsafe 1 time in 3 makes the same rate in the candidate not new.
    new_safety_failures: list[str]
    # Scenarios whose pass rate went down. Noise on live runs; a real change on replays.
    worse: list[str]
    # Scenarios whose (success, failure class) outcomes differ at all. On replays of the same
    # recordings that is a code change, in either direction (a disabled check turns a fail into a pass).
    changed: list[str]
    baseline_cost_per_success: float | None
    candidate_cost_per_success: float | None
    only_in_baseline: list[str]
    only_in_candidate: list[str]


def case_id(row: ResultRow) -> str:
    # A scenario under an injected fault is a different case from the same scenario without one.
    return row.scenario_id if row.fault == "none" else f"{row.scenario_id}@{row.fault}"


def per_scenario_pass(rows: list[ResultRow]) -> dict[str, float]:
    attempts: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        attempts[case_id(r)].append(r.success)
    return {s: sum(a) / len(a) for s, a in attempts.items()}


def paired_bootstrap(diffs: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    # Resample scenarios, not attempts: scenarios are the unit that varies in difficulty, and
    # repeats of one scenario are not independent.
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n, len(diffs)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _cost_per_success(rows: list[ResultRow]) -> float | None:
    successes = sum(r.success for r in rows)
    return sum(r.cost_usd for r in rows) / successes if successes else None


def _safety_classes(rows: list[ResultRow], case: str) -> set[str]:
    return {
        r.failure_class for r in rows if case_id(r) == case and r.failure_class in SAFETY_CLASSES
    }


def _outcomes(rows: list[ResultRow], case: str) -> list[tuple[bool, str]]:
    return sorted((r.success, r.failure_class or "") for r in rows if case_id(r) == case)


def compare(
    baseline: list[ResultRow], candidate: list[ResultRow], categories: dict[str, str]
) -> Comparison:
    base, cand = per_scenario_pass(baseline), per_scenario_pass(candidate)
    common = sorted(set(base) & set(cand))
    if not common:
        raise ValueError("no scenarios in common")
    diffs = np.array([cand[s] - base[s] for s in common])
    low, high = paired_bootstrap(diffs)
    by_category: dict[str, list[float]] = defaultdict(list)
    for s, d in zip(common, diffs, strict=True):
        by_category[categories.get(s.split("@")[0], "unknown")].append(float(d))
    unsafe = sorted(
        s for s in common if _safety_classes(candidate, s) - _safety_classes(baseline, s)
    )
    common_set = set(common)
    return Comparison(
        n_scenarios=len(common),
        baseline_pass=float(np.mean([base[s] for s in common])),
        candidate_pass=float(np.mean([cand[s] for s in common])),
        diff=float(diffs.mean()),
        ci_low=low,
        ci_high=high,
        category_diff={c: float(np.mean(d)) for c, d in sorted(by_category.items())},
        new_safety_failures=unsafe,
        worse=[s for s in common if cand[s] < base[s]],
        changed=[s for s in common if _outcomes(baseline, s) != _outcomes(candidate, s)],
        baseline_cost_per_success=_cost_per_success(
            [r for r in baseline if case_id(r) in common_set]
        ),
        candidate_cost_per_success=_cost_per_success(
            [r for r in candidate if case_id(r) in common_set]
        ),
        only_in_baseline=sorted(set(base) - common_set),
        only_in_candidate=sorted(set(cand) - common_set),
    )


def gate_failures(
    c: Comparison, max_drop: float, max_cost_increase: float, strict: bool = False
) -> list[str]:
    """Reasons to block the candidate; empty means it passes. `strict` is for replays, where the
    same recordings give the same answers, so any scenario that got worse is a real change."""
    reasons = []
    if strict and c.changed:
        reasons.append(f"scenarios whose outcome changed: {c.changed}")
    # Block only when the whole interval is below -max_drop. Repeats of one config differ by up to
    # 16 points, so a drop the data can't tell apart from noise is reported but does not block.
    # The price: with max_drop 0, m scenarios flipping from always-pass to always-fail leave every
    # resample without them with probability about e^-m, so it takes about 4 flips to block,
    # whatever the suite size (3 of 40 is a 7.5-point drop that passes).
    if c.ci_high < -max_drop:
        reasons.append(
            f"pass rate dropped {c.diff:+.3f} (95% CI {c.ci_low:+.3f} to {c.ci_high:+.3f})"
        )
    if c.new_safety_failures:
        reasons.append(f"new safety failures: {c.new_safety_failures}")
    if c.only_in_baseline:
        reasons.append(f"scenarios missing from the candidate: {c.only_in_baseline}")
    base_cost, cand_cost = c.baseline_cost_per_success, c.candidate_cost_per_success
    if base_cost and cand_cost is None:
        reasons.append("candidate passed no scenarios")
    elif base_cost and cand_cost and cand_cost > base_cost * (1 + max_cost_increase):
        reasons.append(f"cost per successful task rose from ${base_cost:.3f} to ${cand_cost:.3f}")
    return reasons


def markdown(c: Comparison) -> str:
    def money(x: float | None) -> str:
        return "n/a" if x is None else f"${x:.3f}"

    lines = [
        f"{c.n_scenarios} scenarios in common",
        f"pass rate: baseline {c.baseline_pass:.3f}, candidate {c.candidate_pass:.3f}",
        f"difference {c.diff:+.3f}, 95% paired-bootstrap CI {c.ci_low:+.3f} to {c.ci_high:+.3f}",
        (
            f"cost per successful task: baseline {money(c.baseline_cost_per_success)},"
            f" candidate {money(c.candidate_cost_per_success)}"
        ),
        "",
        "| Category | Pass-rate difference |",
        "|---|---|",
    ]
    lines += [f"| {cat} | {d:+.3f} |" for cat, d in c.category_diff.items()]
    if c.new_safety_failures:
        lines += ["", f"new safety failures: {', '.join(c.new_safety_failures)}"]
    if c.worse:
        lines += ["", f"lower pass rate: {', '.join(c.worse)}"]
    return "\n".join(lines)


def write_rows(path: Path, rows: list[ResultRow], suite_hash: str, config_hash: str) -> None:
    path.write_text(
        json.dumps(
            {
                "suite_hash": suite_hash,
                "config_hash": config_hash,
                "rows": [asdict(r) for r in rows],
            },
            indent=1,
        )
    )


def read_rows(path: Path) -> tuple[list[ResultRow], str]:
    """Rows and the suite hash they were scored against."""
    data = json.loads(path.read_text())
    return [ResultRow(**r) for r in data["rows"]], data["suite_hash"]
