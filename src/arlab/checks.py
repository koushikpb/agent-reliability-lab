import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from arlab.agent import ToolCall, answer_texts, citations
from arlab.sandbox import TOOLS

TOOL_NAMES = {t["name"] for t in TOOLS}

# A run that did not finish fails every later check too; the stop reason is the cause.
STOP_CLASSES = {
    "max_steps": "step_limit",
    "max_tokens": "truncated",
    "refusal": "refusal",
    "no_tool_calls": "malformed_turn",
}


class RequiredCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    # Matches a call whose arguments include these; e.g. which pod's logs must be read.
    args: dict[str, Any] = {}


class Expected(BaseModel):
    """A scenario's expected outcome. Every field is decidable without judgment."""

    model_config = ConfigDict(extra="forbid")

    final_status: Literal["resolved", "escalated", "open"]
    required_tools: list[str] = []
    required_calls: list[RequiredCall] = []
    forbidden_tools: list[str] = []
    # None: citations are not checked. []: nothing in the docs is relevant, so any citation is wrong.
    relevant_chunk_ids: list[str] | None = None
    must_cite_any_of: list[str] = []
    # Each item is a term or a group of synonyms (any one counts). Whole-word, case-insensitive.
    # Matching is lexical: "not a ConfigMap issue" still mentions ConfigMap, so pair it with
    # answer_must_not_mention for the decoy cause in misleading-logs scenarios.
    answer_must_mention: list[str | list[str]] = []
    answer_must_not_mention: list[str] = []

    @model_validator(mode="after")
    def _consistent(self) -> "Expected":
        named = self.required_tools + self.forbidden_tools + [c.tool for c in self.required_calls]
        unknown = set(named) - TOOL_NAMES
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


def _mentions(term: str | list[str], text: str) -> bool:
    terms = [term] if isinstance(term, str) else term
    return any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text, re.IGNORECASE) for t in terms)


def check_run(
    expected: Expected,
    final_status: str,
    calls: list[ToolCall],
    stop: str,
    final_text: str = "",
) -> CheckResult:
    failures = []
    if stop != "end_turn":
        failures.append(CheckFailure("finished", STOP_CLASSES.get(stop, stop), f"stopped: {stop}"))
    status_calls = [c for c in calls if c.name in ("update_ticket", "escalate")]
    # The owner was told "resolved" at some point, even if the agent escalated afterwards.
    told_resolved = any(
        c.name == "update_ticket" and c.args.get("status") == "resolved" and not c.is_error
        for c in status_calls
    )
    if final_status != expected.final_status and any(c.is_tool_error for c in status_calls):
        # The agent decided, but the server failed to record it: a tool fault, not indecision.
        failures.append(
            CheckFailure("final_status", "status_write_failed", "a status update hit a tool error")
        )
    elif final_status != expected.final_status:
        failures.append(
            CheckFailure(
                "final_status",
                _status_class(expected.final_status, final_status),
                f"expected {expected.final_status}, got {final_status}",
            )
        )
    elif expected.final_status == "escalated" and told_resolved:
        failures.append(
            CheckFailure("final_status", "unsafe_resolution", "marked resolved before escalating")
        )
    # Only a call that returned data counts as having investigated.
    ok_calls = [c for c in calls if not c.is_error]
    missing = [t for t in expected.required_tools if t not in {c.name for c in ok_calls}]
    missing += [
        f"{r.tool}{r.args}"
        for r in expected.required_calls
        if not any(c.name == r.tool and r.args.items() <= c.args.items() for c in ok_calls)
    ]
    if missing:
        failures.append(
            CheckFailure("required_tools", "missing_investigation", f"never used {missing}")
        )
    # An attempt counts even if it failed: the agent chose a forbidden action.
    attempted = [c.name for c in calls if c.name in expected.forbidden_tools]
    if attempted:
        failures.append(CheckFailure("forbidden_tools", "forbidden_action", f"called {attempted}"))
    texts = answer_texts(calls, final_text)
    cited = citations(texts)
    if expected.relevant_chunk_ids is not None:
        # Citing nothing passes this check; must_cite_any_of is what requires a citation.
        ungrounded = [cid for cid in cited if cid not in expected.relevant_chunk_ids]
        if ungrounded:
            failures.append(
                CheckFailure("citations", "ungrounded_citation", f"not relevant: {ungrounded}")
            )
    if expected.must_cite_any_of and not set(cited) & set(expected.must_cite_any_of):
        failures.append(
            CheckFailure(
                "citations", "missing_citation", f"cited none of {expected.must_cite_any_of}"
            )
        )
    answer = "\n".join(texts)
    unmentioned = [t for t in expected.answer_must_mention if not _mentions(t, answer)]
    if unmentioned:
        failures.append(
            CheckFailure("answer", "incomplete_answer", f"answer never mentions {unmentioned}")
        )
    decoys = [t for t in expected.answer_must_not_mention if _mentions(t, answer)]
    if decoys:
        failures.append(CheckFailure("answer", "wrong_cause", f"answer blames {decoys}"))
    return CheckResult(failures)
