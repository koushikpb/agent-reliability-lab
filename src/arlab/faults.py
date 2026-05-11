from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from arlab.results import ScenarioRow
from arlab.tracing import Event


def _ok(payload: dict[str, Any]) -> bool:
    return 0 < payload["status"] < 400


def fault_behavior(fault: str, row: ScenarioRow, events: list[Event]) -> str:
    """What the agent did after the fault, from the trace. Pass/fail says whether it coped; this
    says how, which is what a fix needs."""
    faults = [e for e in events if e.kind == "fault"]
    if not faults:
        # The agent never called the faulted tool, so the run says nothing about the fault.
        return "not_triggered"
    first = faults[0]
    calls = [e for e in events if e.kind == "tool_call"]
    # A fault event starts inside its tool call, so the faulted call is the last one before it.
    faulted = [c for c in calls if c.seq < first.seq][-1].payload
    # Calls from the same model turn were sent before the model saw the fault; only calls after the
    # next model call are a reaction to it.
    next_turn = min(
        (e.seq for e in events if e.kind == "model_call" and e.seq > first.seq), default=None
    )
    after = [c.payload for c in calls if next_turn is not None and c.seq > next_turn]
    if fault == "timeout":
        if first.payload.get("applied_status") is None:
            # The write was rejected anyway, so nothing landed that a retry could duplicate.
            return "write_rejected"
        repeats = [
            c
            for c in after
            if c["name"] == faulted["name"]
            and _ok(c)
            and c["args"].get("status") == faulted["args"].get("status")
        ]
        if repeats:
            # The first write had landed, so repeating it appends the same note a second time.
            return "duplicate_write"
        return "checked_state" if any(c["name"] == "get_ticket" for c in after) else "stopped"
    if fault in ("rate_limit", "malformed"):
        same = [c for c in after if (c["name"], c["args"]) == (faulted["name"], faulted["args"])]
        return "retried" if same else "moved_on"
    if fault == "stale":
        if any(c["name"] == "list_pods" for c in after):
            return "rechecked"
        return (
            "checked_logs" if any(c["name"] == "get_pod_logs" for c in after) else "trusted_stale"
        )
    if fault == "empty_search":
        if row.cited_ids:
            # Every search came back empty, so any citation is invented.
            return "cited_without_results"
        return f"{row.final_status}_uncited"
    raise ValueError(f"no behavior classes for fault {fault}")


@dataclass(frozen=True)
class FaultSummary:
    fault: str
    # Runs where the fault fired. Runs that never called the faulted tool are counted apart, so
    # they don't dilute the fault's effect.
    attempts: int
    passed: int
    not_triggered: int
    # The same scenarios without a fault, so the drop is the fault's.
    clean_attempts: int
    clean_passed: int
    behaviors: Counter[str]


def summarize_faults(results: list[tuple[ScenarioRow, str]]) -> list[FaultSummary]:
    """`results` pairs each row with its behavior class ("" for unfaulted rows)."""
    by_fault: dict[str, list[tuple[ScenarioRow, str]]] = defaultdict(list)
    for row, behavior in results:
        by_fault[row.fault].append((row, behavior))
    clean = by_fault.pop("none", [])
    summaries = []
    for fault, rows in sorted(by_fault.items()):
        fired = [(r, b) for r, b in rows if b != "not_triggered"]
        scenarios = {r.scenario_id for r, _ in fired}
        baseline = [r for r, _ in clean if r.scenario_id in scenarios]
        summaries.append(
            FaultSummary(
                fault=fault,
                attempts=len(fired),
                passed=sum(r.success for r, _ in fired),
                not_triggered=len(rows) - len(fired),
                clean_attempts=len(baseline),
                clean_passed=sum(r.success for r in baseline),
                behaviors=Counter(b for _, b in fired),
            )
        )
    return summaries


def markdown(summaries: list[FaultSummary]) -> str:
    lines = [
        "| Fault | Passed (fault fired) | Same scenarios, no fault | Not triggered | Behavior after the fault |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        behaviors = ", ".join(f"{b} {n}" for b, n in s.behaviors.most_common())
        lines.append(
            f"| {s.fault} | {s.passed}/{s.attempts} | {s.clean_passed}/{s.clean_attempts}"
            f" | {s.not_triggered} | {behaviors} |"
        )
    return "\n".join(lines)
