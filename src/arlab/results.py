import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from arlab.agent import AgentResult, ToolCall, citations
from arlab.tracing import Event

STATUS_TOOLS = ("update_ticket", "escalate")


@dataclass(frozen=True)
class ScenarioRow:
    scenario_id: str
    fault: str
    success: bool
    final_status: str
    tool_errors: int
    invalid_calls: int
    redundant_calls: int
    cited_ids: list[str]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    failure_class: str | None


def redundant_calls(calls: list[ToolCall]) -> int:
    """Calls repeating an earlier call with identical arguments. Retries after a server error are
    not redundant: the agent had no result to reuse."""
    seen: Counter[str] = Counter()
    count = 0
    for c in calls:
        key = json.dumps([c.name, c.args], sort_keys=True)
        if seen[key]:
            count += 1
        if not c.is_tool_error:
            seen[key] += 1
    return count


def _row(
    scenario_id: str,
    fault: str,
    expected: dict[str, Any],
    final_status: str,
    calls: list[ToolCall],
    cited: list[str],
    latency_ms: int,
    tokens: tuple[int, int],
    cost: float,
) -> ScenarioRow:
    # Final status only, until milestone 5 adds the full deterministic checks and failure classes.
    return ScenarioRow(
        scenario_id=scenario_id,
        fault=fault,
        success=final_status == expected["final_status"],
        final_status=final_status,
        tool_errors=sum(c.is_tool_error for c in calls),
        invalid_calls=sum(c.is_invalid for c in calls),
        redundant_calls=redundant_calls(calls),
        cited_ids=cited,
        latency_ms=latency_ms,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        # Rounded to the numeric(10, 6) column so a stored row and a rebuilt row compare equal.
        cost_usd=round(cost, 6),
        failure_class=None,
    )


def row_from_result(
    scenario_id: str,
    fault: str,
    expected: dict[str, Any],
    final_status: str,
    result: AgentResult,
    latency_ms: int,
) -> ScenarioRow:
    return _row(
        scenario_id,
        fault,
        expected,
        final_status,
        result.tool_calls,
        result.cited_ids,
        latency_ms,
        (result.input_tokens, result.output_tokens),
        result.cost_usd,
    )


def row_from_events(
    scenario_id: str, fault: str, expected: dict[str, Any], events: list[Event]
) -> ScenarioRow:
    """Rebuilds a result row from the trace alone. Must agree with row_from_result; a test checks
    that, which proves the events hold everything needed to explain a run."""
    calls = [
        ToolCall(e.payload["name"], e.payload["args"], e.payload["result"], e.payload["status"])
        for e in events
        if e.kind == "tool_call"
    ]
    model_calls = [e.payload for e in events if e.kind == "model_call"]
    final_status = "open"
    for c in calls:
        if c.name in STATUS_TOOLS and not c.is_error:
            final_status = json.loads(c.result)["status"]
    # The final answer is the text of the terminal model call only (see AgentResult.final_text).
    last = model_calls[-1] if model_calls else None
    final_text = ""
    if last and last["stop_reason"] != "tool_use":
        final_text = "\n".join(b["text"] for b in last["content"] if b["type"] == "text")
    notes = [c.args.get("note", "") for c in calls if c.name == "update_ticket"]
    [scenario] = [e for e in events if e.kind == "scenario"]
    return _row(
        scenario_id,
        fault,
        expected,
        final_status,
        calls,
        citations([*notes, final_text]),
        scenario.duration_ms,
        (sum(m["input_tokens"] for m in model_calls), sum(m["output_tokens"] for m in model_calls)),
        sum(m["cost_usd"] for m in model_calls),
    )
