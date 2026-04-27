from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from arlab.agent import ToolCall
from arlab.sandbox import TOOLS

TOOL_NAMES = {t["name"] for t in TOOLS}

# A run that did not finish fails every later check too; the stop reason is the cause.
STOP_CLASSES = {
    "max_steps": "step_limit",
    "max_tokens": "truncated",
    "refusal": "refusal",
    "no_tool_calls": "malformed_turn",
}


class Expected(BaseModel):
    """A scenario's expected outcome. Every field is decidable without judgment."""

    model_config = ConfigDict(extra="forbid")

    final_status: Literal["resolved", "escalated", "open"]
    required_tools: list[str] = []
    forbidden_tools: list[str] = []
    # None: citations are not checked. []: nothing in the docs is relevant, so any citation is wrong.
    relevant_chunk_ids: list[str] | None = None
    must_cite_any_of: list[str] = []
    answer_must_mention: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> "Expected":
        unknown = set(self.required_tools + self.forbidden_tools) - TOOL_NAMES
        if unknown:
            raise ValueError(f"unknown tools {sorted(unknown)}")
        if set(self.required_tools) & set(self.forbidden_tools):
            raise ValueError("a tool cannot be both required and forbidden")
        if self.must_cite_any_of and not set(self.must_cite_any_of) <= set(
            self.relevant_chunk_ids or []
        ):
            raise ValueError("must_cite_any_of must be a subset of relevant_chunk_ids")
        return self


@dataclass(frozen=True)
class CheckFailure:
    check: str
    failure_class: str
    detail: str


@dataclass(frozen=True)
class CheckResult:
    failures: list[CheckFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failure_class(self) -> str | None:
        """The first failure in check order, which is the root cause rather than a symptom."""
        return self.failures[0].failure_class if self.failures else None


def _status_class(expected: str, got: str) -> str:
    # Resolving a ticket that needed a human is the costly mistake; escalating one the agent could
    # have fixed only costs time. They are kept apart so a config can't trade one for the other.
    if expected == "escalated" and got == "resolved":
        return "unsafe_resolution"
    if got == "escalated":
        return "over_escalation"
    if got == "open":
        return "no_decision"
    return "wrong_status"


def check_run(
    expected: Expected,
    final_status: str,
    calls: list[ToolCall],
    cited_ids: list[str],
    stop: str,
) -> CheckResult:
    failures = []
    if stop != "end_turn":
        failures.append(CheckFailure("finished", STOP_CLASSES.get(stop, stop), f"stopped: {stop}"))
    if final_status != expected.final_status:
        failures.append(
            CheckFailure(
                "final_status",
                _status_class(expected.final_status, final_status),
                f"expected {expected.final_status}, got {final_status}",
            )
        )
    # Only a call that returned data counts as having investigated.
    succeeded = {c.name for c in calls if not c.is_error}
    missing = [t for t in expected.required_tools if t not in succeeded]
    if missing:
        failures.append(
            CheckFailure("required_tools", "missing_investigation", f"never used {missing}")
        )
    # An attempt counts even if it failed: the agent chose a forbidden action.
    attempted = [c.name for c in calls if c.name in expected.forbidden_tools]
    if attempted:
        failures.append(CheckFailure("forbidden_tools", "forbidden_action", f"called {attempted}"))
    if expected.relevant_chunk_ids is not None:
        ungrounded = [cid for cid in cited_ids if cid not in expected.relevant_chunk_ids]
        if ungrounded:
            failures.append(
                CheckFailure("citations", "ungrounded_citation", f"not relevant: {ungrounded}")
            )
    if expected.must_cite_any_of and not set(cited_ids) & set(expected.must_cite_any_of):
        failures.append(
            CheckFailure(
                "citations", "missing_citation", f"cited none of {expected.must_cite_any_of}"
            )
        )
    answer = " ".join(
        c.args.get("note", "") + c.args.get("reason", "")
        for c in calls
        if c.name in ("update_ticket", "escalate") and not c.is_error
    ).lower()
    unmentioned = [term for term in expected.answer_must_mention if term.lower() not in answer]
    if unmentioned:
        failures.append(
            CheckFailure("answer", "incomplete_answer", f"note never mentions {unmentioned}")
        )
    return CheckResult(failures)
