from dataclasses import replace
from pathlib import Path

import pytest

from arlab.agent import run_agent
from arlab.config import load_config, system_prompt
from arlab.llm import ModelResponse, RecordingProvider, RecordingStore, ReplayMiss, ReplayProvider
from arlab.runner import load_suite, recording_search, replay_search, run_scenario
from arlab.sandbox import ToolClient, create_tool_server

CONFIG = load_config(Path("configs/baseline.yaml"))
MODEL = "claude-opus-5-5"


def tool_use(i, name, args):
    return {"type": "tool_use", "id": f"tu_{i}", "name": name, "input": args}


def reply(*blocks, stop="tool_use"):
    return ModelResponse(list(blocks), stop, MODEL, input_tokens=1000, output_tokens=100)


class ScriptedProvider:
    """Plays back fixed responses and keeps the requests it received."""

    model = MODEL

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, system, messages, tools):
        self.requests.append([dict(m) for m in messages])
        return self.responses.pop(0)


def fake_search(query):
    return [
        {"chunk_id": "concepts/configuration/configmap#configmaps-0", "heading": "H", "text": "t"}
    ]


def resolve_script(ticket_id, namespace):
    return [
        reply(
            {"type": "thinking", "thinking": "", "signature": "sig"},
            tool_use(1, "get_ticket", {"id": ticket_id}),
            tool_use(2, "list_pods", {"namespace": namespace}),
        ),
        reply(tool_use(3, "search_docs", {"query": "missing configmap crashloop"})),
        reply(
            tool_use(
                4,
                "update_ticket",
                {
                    "id": ticket_id,
                    "status": "resolved",
                    "note": "Create the ConfigMap. [cite: concepts/configuration/configmap#configmaps-0]",
                },
            )
        ),
        reply({"type": "text", "text": "Resolved."}, stop="end_turn"),
    ]


@pytest.fixture
def suite():
    return {s.id: s for s in load_suite(Path("scenarios/smoke"))}


def test_resolves_smoke_ticket_end_to_end(suite):
    scenario = suite["crashloop-missing-configmap"]
    provider = ScriptedProvider(resolve_script("T-1001", "payments"))
    outcome = run_scenario(scenario, provider, fake_search, CONFIG)

    assert outcome.final_status == scenario.expected.final_status == "resolved"
    assert outcome.result.stop == "end_turn"
    assert outcome.result.steps == 4
    assert [c.name for c in outcome.result.tool_calls] == [
        "get_ticket",
        "list_pods",
        "search_docs",
        "update_ticket",
    ]
    assert outcome.result.cited_ids == ["concepts/configuration/configmap#configmaps-0"]
    assert outcome.result.cost_usd == pytest.approx(4 * (1000 * 4 + 100 * 20) / 1e6)


def test_thinking_blocks_and_parallel_results_go_back_unchanged(suite):
    provider = ScriptedProvider(resolve_script("T-1001", "payments"))
    run_scenario(suite["crashloop-missing-configmap"], provider, fake_search, CONFIG)
    second_request = provider.requests[1]
    assert second_request[1]["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig",
    }
    results = second_request[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]


def test_tool_errors_are_reported_to_the_model(suite):
    provider = ScriptedProvider(
        [
            reply(tool_use(1, "get_pod_logs", {"pod": "no-such-pod"})),
            reply({"type": "text", "text": "Giving up."}, stop="end_turn"),
        ]
    )
    outcome = run_scenario(suite["image-pull-typo"], provider, fake_search, CONFIG)
    assert outcome.result.tool_calls[0].is_error
    assert provider.requests[1][2]["content"][0]["is_error"] is True
    assert outcome.final_status == "open"


def test_stops_at_max_steps():
    server = create_tool_server({"tickets": {"T": {"status": "open", "notes": []}}}, fake_search)
    provider = ScriptedProvider([reply(tool_use(i, "get_ticket", {"id": "T"})) for i in range(5)])
    result = run_agent(provider, ToolClient.in_process(server), "sys", "T", max_steps=3)
    assert (result.stop, result.steps) == ("max_steps", 3)


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_non_tool_stop_reasons_end_the_run(stop):
    server = create_tool_server({"tickets": {}}, fake_search)
    result = run_agent(
        ScriptedProvider([reply(stop=stop)]), ToolClient.in_process(server), "s", "T", 5
    )
    assert result.stop == stop


def test_record_then_replay_gives_the_same_run(suite, tmp_path):
    scenario = suite["crashloop-missing-configmap"]
    store = RecordingStore(tmp_path)
    recorded = run_scenario(
        scenario,
        RecordingProvider(ScriptedProvider(resolve_script("T-1001", "payments")), store),
        recording_search(fake_search, store, "idx", "dense", 8),
        CONFIG,
    )
    replayed = run_scenario(
        scenario, ReplayProvider(MODEL, store), replay_search(store, "idx", "dense", 8), CONFIG
    )
    # Timing differs between any two runs; everything the agent did must not.
    assert replayed.result == recorded.result
    assert replace(replayed.row, latency_ms=0) == replace(recorded.row, latency_ms=0)
    assert [(e.kind, e.payload) for e in replayed.events] == [
        (e.kind, e.payload) for e in recorded.events
    ]


def test_replay_misses_when_the_prompt_changes(suite, tmp_path):
    scenario = suite["crashloop-missing-configmap"]
    store = RecordingStore(tmp_path)
    provider = RecordingProvider(ScriptedProvider(resolve_script("T-1001", "payments")), store)
    run_scenario(scenario, provider, fake_search, CONFIG)
    server = create_tool_server(scenario.cluster, fake_search)
    with pytest.raises(ReplayMiss):
        run_agent(
            ReplayProvider(MODEL, store),
            ToolClient.in_process(server),
            system_prompt(CONFIG) + " changed",
            "T-1001",
            12,
        )


def test_all_smoke_scenarios_load(suite):
    assert set(suite) == {"crashloop-missing-configmap", "image-pull-typo", "pvc-lost-database"}
    for s in suite.values():
        assert s.cluster["tickets"][s.ticket_id]["status"] == "open"


def test_final_text_comes_only_from_the_terminal_response():
    server = create_tool_server({"tickets": {"T": {"status": "open", "notes": []}}}, fake_search)
    narration = {"type": "text", "text": "Checking [cite: early#x-0] first."}
    provider = ScriptedProvider(
        [reply(narration, tool_use(1, "get_ticket", {"id": "T"})) for _ in range(2)]
    )
    result = run_agent(provider, ToolClient.in_process(server), "s", "T", max_steps=2)
    assert result.stop == "max_steps"
    assert result.final_text == "" and result.cited_ids == []


def test_pause_turn_resumes_instead_of_stopping():
    server = create_tool_server({"tickets": {}}, fake_search)
    provider = ScriptedProvider(
        [reply(stop="pause_turn"), reply({"type": "text", "text": "ok"}, stop="end_turn")]
    )
    result = run_agent(provider, ToolClient.in_process(server), "s", "T", max_steps=5)
    assert (result.stop, result.steps, result.final_text) == ("end_turn", 2, "ok")
