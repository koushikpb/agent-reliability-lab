import json
from collections import Counter
from dataclasses import dataclass

from arlab.agent import AgentResult, ToolCall, answer_texts, citations
from arlab.checks import Expected, check_run
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
    expected: Expected,
    final_status: str,
    stop: str,
    calls: list[ToolCall],
    final_text: str,
    latency_ms: int,
    tokens: tuple[int, int],
    cost: float,
) -> ScenarioRow:
    checks = check_run(expected, final_status, calls, stop, final_text)
    return ScenarioRow(
        scenario_id=scenario_id,
        fault=fault,
        success=checks.passed,
        final_status=final_status,
        tool_errors=sum(c.is_tool_error for c in calls),
        invalid_calls=sum(c.is_invalid for c in calls),
        redundant_calls=redundant_calls(calls),
        cited_ids=citations(answer_texts(calls, final_text)),
        latency_ms=latency_ms,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        # Rounded to the numeric(10, 6) column so a stored row and a rebuilt row compare equal.
        cost_usd=round(cost, 6),
        failure_class=checks.failure_class,
    )


def row_from_result(
    scenario_id: str,
    fault: str,
    expected: Expected,
    final_status: str,
    result: AgentResult,
    latency_ms: int,
) -> ScenarioRow:
    return _row(
        scenario_id,
        fault,
        expected,
        final_status,
        result.stop,
        result.tool_calls,
        result.final_text,
        latency_ms,
        (result.input_tokens, result.output_tokens),
        result.cost_usd,
    )


def calls_and_answer(events: list[Event]) -> tuple[list[ToolCall], str]:
    """The tool calls and the final answer text, from the trace alone."""
    calls = [
        ToolCall(e.payload["name"], e.payload["args"], e.payload["result"], e.payload["status"])
        for e in events
        if e.kind == "tool_call"
    ]
    model_calls = [e.payload for e in events if e.kind == "model_call"]
    # The final answer is the text of the terminal model call only (see AgentResult.final_text).
    last = model_calls[-1] if model_calls else None
    final_text = ""
    if last and last["stop_reason"] not in ("tool_use", "pause_turn"):
        final_text = "\n".join(b["text"] for b in last["content"] if b["type"] == "text")
    return calls, final_text


def row_from_events(
    scenario_id: str, fault: str, expected: Expected, events: list[Event]
) -> ScenarioRow:
    """Rebuilds a result row from the trace alone. Must agree with row_from_result; a test checks
    that, which proves the events hold everything needed to explain a run."""
    calls, final_text = calls_and_answer(events)
    model_calls = [e.payload for e in events if e.kind == "model_call"]
    final_status = "open"
    for e in events:
        if e.kind == "tool_call" and e.payload["name"] in STATUS_TOOLS:
            if 0 < e.payload["status"] < 400:  # the same test as ToolCall.is_error
                final_status = json.loads(e.payload["result"])["status"]
        # A write applied before an injected timeout changed the ticket though the agent saw an error.
        elif e.kind == "fault" and e.payload.get("applied_status"):
            final_status = e.payload["applied_status"]
    [scenario] = [e for e in events if e.kind == "scenario"]
    return _row(
        scenario_id,
        fault,
        expected,
        final_status,
        scenario.payload["stop"],
        calls,
        final_text,
        scenario.duration_ms,
        (sum(m["input_tokens"] for m in model_calls), sum(m["output_tokens"] for m in model_calls)),
        sum(m["cost_usd"] for m in model_calls),
    )
