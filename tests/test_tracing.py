import json
from pathlib import Path

from arlab.config import load_config
from arlab.db import apply_schema
from arlab.results import row_from_events
from arlab.runner import load_suite, run_scenario
from arlab.store import finish_run, load_events, load_row, record_scenario, start_run
from tests.test_agent import ScriptedProvider, fake_search, reply, resolve_script, tool_use

CONFIG = load_config(Path("configs/baseline.yaml"))


def messy_script():
    """Exercises every column: an invalid call, a redundant call, a citation, a final answer."""
    return [
        reply(tool_use(1, "get_pod_logs", {"pod": "nope"})),
        reply(tool_use(2, "list_pods", {"namespace": "payments"})),
        reply(tool_use(3, "list_pods", {"namespace": "payments"})),
        reply(
            tool_use(4, "search_docs", {"query": "configmap"}),
            tool_use(5, "get_ticket", {"id": "T-1001"}),
        ),
        reply(
            tool_use(
                6,
                "update_ticket",
                {"id": "T-1001", "status": "resolved", "note": "[cite: doc#a-0] create it"},
            )
        ),
        reply({"type": "text", "text": "Done, see [cite: doc#b-1]."}, stop="end_turn"),
    ]


def test_scenario_result_rebuilds_from_events_alone(conn):
    apply_schema(conn)
    scenario = {s.id: s for s in load_suite(Path("scenarios/smoke"))}["crashloop-missing-configmap"]
    outcome = run_scenario(scenario, ScriptedProvider(messy_script()), fake_search, CONFIG)
    run_id = start_run(
        conn,
        batch_id="b",
        suite="smoke",
        suite_hash="s",
        config=CONFIG,
        index_version="i",
        mode="replay",
    )
    record_scenario(conn, run_id, outcome.row, outcome.events)
    finish_run(conn, run_id, "finished")

    stored = load_row(conn, run_id, scenario.id, "none")
    rebuilt = row_from_events(
        scenario.id, "none", scenario.expected, load_events(conn, run_id, scenario.id, "none")
    )
    assert rebuilt == stored
    assert (stored.invalid_calls, stored.redundant_calls, stored.tool_errors) == (1, 1, 0)
    assert stored.cited_ids == ["doc#a-0", "doc#b-1"]
    assert stored.final_status == "resolved" and stored.success


def test_events_form_a_tree_in_start_order():
    scenario = load_suite(Path("scenarios/smoke"))[0]
    outcome = run_scenario(
        scenario, ScriptedProvider(resolve_script("T-1001", "payments")), fake_search, CONFIG
    )
    events = outcome.events
    assert [e.seq for e in events] == list(range(len(events)))
    root = events[0]
    assert root.kind == "scenario" and root.parent_span_id is None
    by_span = {e.span_id: e for e in events}
    [retrieval] = [e for e in events if e.kind == "retrieval"]
    assert by_span[retrieval.parent_span_id].payload["name"] == "search_docs"
    assert all(e.parent_span_id == root.span_id for e in events if e.kind == "model_call")
    assert json.loads(json.dumps([e.payload for e in events]))  # payloads are plain JSON


def test_failed_run_is_marked_failed(conn):
    apply_schema(conn)
    run_id = start_run(
        conn, batch_id="b", suite="s", suite_hash="h", config=CONFIG, index_version="i", mode="live"
    )
    finish_run(conn, run_id, "failed")
    assert conn.execute("SELECT status FROM runs WHERE id = %s", (run_id,)).fetchone() == (
        "failed",
    )
