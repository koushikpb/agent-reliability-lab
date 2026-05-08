from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import psycopg


@dataclass(frozen=True)
class ResultRow:
    run_id: str
    scenario_id: str
    fault: str
    success: bool
    failure_class: str | None
    latency_ms: int
    cost_usd: float


@dataclass(frozen=True)
class BatchSummary:
    n_runs: int
    n_scenarios: int
    # Repeat k is the k-th attempt at each scenario, so a batch can mix full and partial runs. The
    # spread across repeats is the noise floor for comparing configs.
    repeat_pass_rates: list[float]
    category_pass: dict[str, tuple[int, int]]  # category -> (passes, attempts) over all runs
    failure_classes: Counter[str]
    # Scenarios that passed in some repeats and failed in others.
    flaky: list[str]
    # A partial run and its top-up are one repeat, so cost is per repeat, not per stored run.
    cost_per_repeat: float
    cost_per_success: float | None
    p50_latency_ms: float
    p95_latency_ms: float


def load_batch(conn: psycopg.Connection, batch_id: str) -> tuple[list[ResultRow], set[str]]:
    """Rows from the batch's finished runs, and the suite hashes those runs were made with."""
    rows = conn.execute(
        "SELECT s.run_id, s.scenario_id, s.fault, s.success, s.failure_class, s.latency_ms,"
        # numeric comes back as Decimal, which doesn't mix with the float math in compare.
        " s.cost_usd::float8 FROM scenario_results s JOIN runs r ON r.id = s.run_id"
        " WHERE r.batch_id = %s AND r.status IN ('finished', 'interrupted') ORDER BY r.started_at, s.scenario_id",
        (batch_id,),
    ).fetchall()
    hashes = conn.execute(
        "SELECT DISTINCT suite_hash FROM runs WHERE batch_id = %s AND status IN ('finished', 'interrupted')",
        (batch_id,),
    ).fetchall()
    return [ResultRow(*r) for r in rows], {h[0] for h in hashes}


def summarize(rows: list[ResultRow], categories: dict[str, str]) -> BatchSummary:
    if not rows:
        raise ValueError("no results in this batch")
    runs: set[str] = set()
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    by_scenario: dict[str, list[bool]] = defaultdict(list)
    category_pass: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        runs.add(r.run_id)
        by_repeat[len(by_scenario[r.scenario_id])].append(r.success)
        by_scenario[r.scenario_id].append(r.success)
        tally = category_pass[categories.get(r.scenario_id, "unknown")]
        tally[0] += r.success
        tally[1] += 1
    successes = sum(r.success for r in rows)
    total_cost = sum(r.cost_usd for r in rows)
    p50, p95 = (float(p) for p in np.percentile([r.latency_ms for r in rows], [50, 95]))
    return BatchSummary(
        n_runs=len(runs),
        n_scenarios=len(by_scenario),
        repeat_pass_rates=[sum(rs) / len(rs) for _, rs in sorted(by_repeat.items())],
        category_pass={c: (p, n) for c, (p, n) in sorted(category_pass.items())},
        failure_classes=Counter(r.failure_class for r in rows if r.failure_class),
        flaky=sorted(s for s, results in by_scenario.items() if 0 < sum(results) < len(results)),
        cost_per_repeat=total_cost / len(by_repeat),
        # Cost per successful task, not per task: a cheap config that fails more isn't cheaper.
        cost_per_success=total_cost / successes if successes else None,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
    )


def markdown(summary: BatchSummary) -> str:
    rates = summary.repeat_pass_rates
    per_success = "n/a" if summary.cost_per_success is None else f"${summary.cost_per_success:.3f}"
    lines = [
        f"{summary.n_runs} runs, {summary.n_scenarios} scenarios, {len(rates)} repeats",
        f"pass rate: mean {np.mean(rates):.3f}, range {min(rates):.3f}-{max(rates):.3f}",
        f"per repeat: {', '.join(f'{r:.2f}' for r in rates)}",
        f"cost: ${summary.cost_per_repeat:.2f} per repeat, {per_success} per successful task",
        f"latency: p50 {summary.p50_latency_ms / 1000:.1f}s, p95 {summary.p95_latency_ms / 1000:.1f}s",
        "",
        "| Category | Passed | Attempts | Pass rate |",
        "|---|---|---|---|",
    ]
    for category, (passed, attempts) in summary.category_pass.items():
        lines.append(f"| {category} | {passed} | {attempts} | {passed / attempts:.2f} |")
    lines += ["", "| Failure class | Count |", "|---|---|"]
    lines += [f"| {cls} | {n} |" for cls, n in summary.failure_classes.most_common()]
    if summary.flaky:
        lines += ["", f"flaky across repeats: {', '.join(summary.flaky)}"]
    return "\n".join(lines)
