from dataclasses import replace
from pathlib import Path

import pytest

from arlab.config import load_config
from arlab.faults import fault_behavior, markdown, summarize_faults
from arlab.runner import load_suite, run_scenario
from tests.test_agent import ScriptedProvider, fake_search, reply, resolve_script, tool_use

CONFIG = load_config(Path("configs/baseline.yaml"))


@pytest.fixture
def scenario():
    return {s.id: s for s in load_suite(Path("scenarios/smoke"))}["crashloop-missing-configmap"]


def behavior(scenario, script, fault):
    outcome = run_scenario(scenario, ScriptedProvider(script), fake_search, CONFIG, fault)
    return fault_behavior(fault, outcome.row, outcome.events), outcome


def with_retry(script, index):
    """The script with the tool call at `index` sent a second time right after it."""
    retry = reply(*[b for b in script[index].content if b["type"] == "tool_use"])
    return [*script[: index + 1], retry, *script[index + 1 :]]


def test_retried_write_after_timeout_is_a_duplicate(scenario):
    script = with_retry(resolve_script(scenario.ticket_id, "payments"), 2)
    assert behavior(scenario, script, "timeout")[0] == "duplicate_write"


def test_write_not_retried_after_timeout_stops(scenario):
    found, outcome = behavior(scenario, resolve_script(scenario.ticket_id, "payments"), "timeout")
    assert found == "stopped"
    assert outcome.final_status == "resolved"


def test_rate_limited_read_retried(scenario):
    script = resolve_script(scenario.ticket_id, "payments")
    script.insert(1, reply(tool_use(9, "list_pods", {"namespace": "payments"})))
    assert behavior(scenario, script, "rate_limit")[0] == "retried"


def test_rate_limited_read_not_retried(scenario):
    script = resolve_script(scenario.ticket_id, "payments")
    assert behavior(scenario, script, "rate_limit")[0] == "moved_on"


def test_fault_on_a_tool_never_called_is_not_triggered(scenario):
    script = resolve_script(scenario.ticket_id, "payments")
    assert behavior(scenario, script, "malformed")[0] == "not_triggered"


def test_citing_after_empty_search_is_invented(scenario):
    found, outcome = behavior(
        scenario, resolve_script(scenario.ticket_id, "payments"), "empty_search"
    )
    assert found == "cited_without_results"
    assert outcome.row.cited_ids


def test_summary_sets_each_fault_beside_the_same_scenarios_clean(scenario):
    _, clean = behavior(scenario, resolve_script(scenario.ticket_id, "payments"), "timeout")
    clean_row = replace(clean.row, fault="none")
    found, faulted = behavior(scenario, resolve_script(scenario.ticket_id, "payments"), "stale")
    [summary] = summarize_faults([(clean_row, ""), (faulted.row, found)])
    assert (summary.fault, summary.clean_attempts, summary.behaviors) == ("stale", 1, {found: 1})
    assert "| stale |" in markdown([summary])


def test_calls_in_the_same_turn_as_the_fault_are_not_a_reaction(scenario):
    # list_pods (stale) and get_pod_logs go out in one turn: the logs read is not a re-check.
    script = resolve_script(scenario.ticket_id, "payments")
    script[0] = reply(
        tool_use(1, "list_pods", {"namespace": "payments"}),
        tool_use(2, "get_pod_logs", {"pod": "anything"}),
    )
    assert behavior(scenario, script, "stale")[0] == "trusted_stale"


def test_a_different_status_after_a_timeout_is_not_a_duplicate(scenario):
    script = resolve_script(scenario.ticket_id, "payments")
    progress = reply(
        tool_use(
            7, "update_ticket", {"id": scenario.ticket_id, "status": "in_progress", "note": "n"}
        )
    )
    script.insert(2, progress)
    assert behavior(scenario, script, "timeout")[0] == "stopped"


def test_not_triggered_runs_stay_out_of_the_pass_rate(scenario):
    found, outcome = behavior(scenario, resolve_script(scenario.ticket_id, "payments"), "malformed")
    [summary] = summarize_faults([(outcome.row, found)])
    assert (summary.attempts, summary.not_triggered) == (0, 1)
