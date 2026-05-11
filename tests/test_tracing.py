import json
from pathlib import Path

import pytest

from arlab.config import load_config
from arlab.db import apply_schema
from arlab.llm import RecordingStore, ReplayMiss
from arlab.results import row_from_events
from arlab.runner import ScenarioCrashed, load_suite, replay_search, run_scenario
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


def test_crashed_scenario_keeps_its_events_with_the_error():
    class Exploding(ScriptedProvider):
        def complete(self, system, messages, tools):
            if len(self.requests) == 1:
                raise ConnectionError("api unreachable")
            return super().complete(system, messages, tools)

    scenario = load_suite(Path("scenarios/smoke"))[0]
    with pytest.raises(ScenarioCrashed) as crash:
        run_scenario(scenario, Exploding(resolve_script("T-1001", "payments")), fake_search, CONFIG)
    events = crash.value.events
    assert isinstance(crash.value.__cause__, ConnectionError)
    assert [e.kind for e in events if e.kind == "model_call"] == ["model_call", "model_call"]
    [failed_call] = [e for e in events if e.kind == "model_call" and "error" in e.payload]
    assert "api unreachable" in failed_call.payload["error"]
    assert any(e.kind == "scenario" and "error" in e.payload for e in events)


def test_search_replay_miss_crashes_the_scenario_instead_of_reaching_the_agent(tmp_path):
    scenario = load_suite(Path("scenarios/smoke"))[0]
    store = RecordingStore(tmp_path)
    with pytest.raises(ScenarioCrashed) as crash:
        run_scenario(
            scenario,
            ScriptedProvider(resolve_script("T-1001", "payments")),
            replay_search(store, "idx", "dense", 8),
            CONFIG,
        )
    assert isinstance(crash.value.__cause__, ReplayMiss)


def test_escalation_and_step_limit_rebuild_from_events(conn):
    apply_schema(conn)
    scenario = {s.id: s for s in load_suite(Path("scenarios/smoke"))}["pvc-lost-database"]
    script = [
        reply(tool_use(1, "escalate", {"id": "T-1003", "reason": "PVC lost, data loss risk"})),
        reply(tool_use(2, "get_ticket", {"id": "T-1003"})),
        reply(tool_use(3, "get_ticket", {"id": "T-1003"})),
    ]
    config = CONFIG.model_copy(update={"agent": CONFIG.agent.model_copy(update={"max_steps": 3})})
    outcome = run_scenario(scenario, ScriptedProvider(script), fake_search, config)
    run_id = start_run(
        conn,
        batch_id="b",
        suite="s",
        suite_hash="h",
        config=config,
        index_version="i",
        mode="replay",
    )
    record_scenario(conn, run_id, outcome.row, outcome.events)
    stored = load_row(conn, run_id, scenario.id, "none")
    rebuilt = row_from_events(
        scenario.id, "none", scenario.expected, load_events(conn, run_id, scenario.id, "none")
    )
    assert rebuilt == stored
    assert (stored.final_status, stored.failure_class) == ("escalated", "step_limit")
    assert stored.redundant_calls == 1
    # Latency in the row is the scenario span; check it against wall time, not against itself.
    assert 0 <= stored.latency_ms < 10_000


def test_rescore_writes_a_new_run_and_leaves_the_source(conn):
    from arlab.checks import Expected
    from arlab.store import rescore_run

    apply_schema(conn)
    scenario = {s.id: s for s in load_suite(Path("scenarios/smoke"))}["crashloop-missing-configmap"]
    outcome = run_scenario(scenario, ScriptedProvider(messy_script()), fake_search, CONFIG)
    source = start_run(
        conn,
        batch_id="b",
        suite="smoke",
        suite_hash="s",
        config=CONFIG,
        index_version="i",
        mode="replay",
    )
    record_scenario(conn, source, outcome.row, outcome.events)
    finish_run(conn, source, "finished")
    stricter = Expected(final_status="resolved", answer_must_mention=["ConfigMap"])

    new, missing = rescore_run(
        conn,
        source,
        {scenario.id: stricter, "never-ran": stricter},
        batch_id="b-rescored",
        suite=Path("scenarios/smoke"),
        config=CONFIG,
    )

    assert load_row(conn, source, scenario.id, "none").success
    assert load_row(conn, new, scenario.id, "none").failure_class == "incomplete_answer"
    assert missing == ["never-ran"]
    assert conn.execute("SELECT source_run_id FROM runs WHERE id = %s", (new,)).fetchone() == (
        source,
    )


def test_timed_out_write_rebuilds_from_events(conn):
    # The agent sees an error, but the ticket was resolved: the stored row reads that from the
    # server, and the rebuilt row must read it from the fault event.
    apply_schema(conn)
    scenario = {s.id: s for s in load_suite(Path("scenarios/smoke"))}["crashloop-missing-configmap"]
    script = resolve_script(scenario.ticket_id, "payments")
    outcome = run_scenario(scenario, ScriptedProvider(script), fake_search, CONFIG, "timeout")
    assert outcome.final_status == "resolved"
    assert outcome.row.tool_errors == 1
    assert [e.payload["fault"] for e in outcome.events if e.kind == "fault"] == ["timeout"]
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
    stored = load_row(conn, run_id, scenario.id, "timeout")
    events = load_events(conn, run_id, scenario.id, "timeout")
    assert row_from_events(scenario.id, "timeout", scenario.expected, events) == stored


def test_batch_loads_interrupted_runs_but_not_failed_ones(conn):
    from arlab.compare import compare, gate_failures
    from arlab.report import load_batch

    apply_schema(conn)
    scenario = {s.id: s for s in load_suite(Path("scenarios/smoke"))}["crashloop-missing-configmap"]
    for status in ("interrupted", "failed"):
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
        finish_run(conn, run_id, status)

    rows, hashes = load_batch(conn, "b")
    assert len(rows) == 1 and hashes == {"s"}
    # Postgres numeric would come back as Decimal and break the cost arithmetic in the gate.
    assert isinstance(rows[0].cost_usd, float)
    gate_failures(compare(rows, rows, {}), max_drop=0.0, max_cost_increase=0.25)
