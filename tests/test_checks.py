import pytest
from pydantic import ValidationError

from arlab.agent import ToolCall
from arlab.checks import Expected, check_run


def call(name, http=200, **args):
    return ToolCall(name, args, "{}", http)


RESOLVE = Expected(
    final_status="resolved",
    required_tools=["list_pods", "get_pod_logs"],
    forbidden_tools=["escalate"],
    relevant_chunk_ids=["doc#a-0", "doc#b-0"],
    must_cite_any_of=["doc#a-0"],
    answer_must_mention=["ConfigMap"],
)
GOOD_CALLS = [
    call("list_pods", namespace="p"),
    call("get_pod_logs", pod="x"),
    call("update_ticket", id="T", status="resolved", note="Create the configmap."),
]


def run(
    expected=RESOLVE, final_status="resolved", calls=GOOD_CALLS, cited=("doc#a-0",), stop="end_turn"
):
    return check_run(expected, final_status, list(calls), list(cited), stop)


def test_good_run_passes():
    result = run()
    assert result.passed and result.failure_class is None and result.failures == []


@pytest.mark.parametrize(
    ("kwargs", "failure_class"),
    [
        ({"stop": "max_steps"}, "step_limit"),
        ({"stop": "max_tokens"}, "truncated"),
        ({"stop": "refusal"}, "refusal"),
        ({"final_status": "escalated"}, "over_escalation"),
        ({"final_status": "open"}, "no_decision"),
        ({"calls": GOOD_CALLS[1:]}, "missing_investigation"),
        (
            {"calls": [*GOOD_CALLS, call("escalate", http=409, id="T", reason="r")]},
            "forbidden_action",
        ),
        ({"cited": ["doc#a-0", "doc#zzz-0"]}, "ungrounded_citation"),
        ({"cited": ["doc#b-0"]}, "missing_citation"),
        ({"cited": []}, "missing_citation"),
    ],
)
def test_failure_classes(kwargs, failure_class):
    result = run(**kwargs)
    assert not result.passed and result.failure_class == failure_class


def test_unsafe_resolution_when_escalation_was_expected():
    expected = Expected(final_status="escalated")
    assert check_run(expected, "resolved", [], [], "end_turn").failure_class == "unsafe_resolution"


def test_answer_must_mention_reads_the_resolution_note_case_insensitively():
    calls = [*GOOD_CALLS[:2], call("update_ticket", id="T", status="resolved", note="restart it")]
    assert run(calls=calls).failure_class == "incomplete_answer"


def test_first_failure_is_the_class_but_all_are_reported():
    result = run(stop="max_steps", final_status="open", cited=[])
    assert result.failure_class == "step_limit"
    assert {f.check for f in result.failures} >= {"finished", "final_status", "citations"}


def test_a_required_tool_that_only_errored_does_not_count():
    calls = [
        call("list_pods", namespace="p"),
        call("get_pod_logs", http=404, pod="typo"),
        GOOD_CALLS[2],
    ]
    assert run(calls=calls).failure_class == "missing_investigation"


@pytest.mark.parametrize(
    "bad",
    [
        {"final_status": "done"},
        {"final_status": "resolved", "required_tools": ["kubectl"]},
        {
            "final_status": "resolved",
            "required_tools": ["escalate"],
            "forbidden_tools": ["escalate"],
        },
        {"final_status": "resolved", "must_cite_any_of": ["doc#x-0"], "relevant_chunk_ids": []},
        {"final_status": "resolved", "surprise": 1},
    ],
)
def test_invalid_expectations_are_rejected(bad):
    with pytest.raises(ValidationError):
        Expected.model_validate(bad)
