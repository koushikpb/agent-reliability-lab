import json
from pathlib import Path
from types import SimpleNamespace

from arlab.judge import (
    JudgedRow,
    Reference,
    Verdict,
    call_judge,
    judge_prompt,
    judgment_markdown,
    scenario_reference,
)
from arlab.runner import load_suite, run_scenario
from tests.test_agent import ScriptedProvider, fake_search, reply, tool_use
from tests.test_tracing import CONFIG


def test_reference_reads_header_and_decoy(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "# Misleading logs: the liveness probe hits the wrong port.\n# Fix the port.\nid: s\n"
        "expected: {}\n# decoy: Kafka (judged in M10, not by a word check)\n"
    )
    assert scenario_reference(path) == Reference(
        "the liveness probe hits the wrong port. Fix the port.", "Kafka"
    )


def test_reference_drops_check_setup_notes(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "# The disk is full. Resolved.\n# Every pod's logs show it, so get_pod_logs is required as a"
        " tool rather than on one pod.\nid: s\n"
    )
    assert scenario_reference(path).cause == "The disk is full. Resolved."


def test_every_core_scenario_has_a_reference():
    for path in Path("scenarios/core").glob("*.yaml"):
        assert scenario_reference(path).cause, path


def test_prompt_shows_answer_and_cited_chunk_text():
    scenario = {s.id: s for s in load_suite(Path("scenarios/smoke"))}["crashloop-missing-configmap"]
    script = [
        reply(tool_use(1, "search_docs", {"query": "configmap"})),
        reply(
            tool_use(
                2,
                "update_ticket",
                {
                    "id": scenario.ticket_id,
                    "status": "resolved",
                    "note": "Create it [cite: concepts/configuration/configmap#configmaps-0]"
                    " [cite: made/up#x-0]",
                },
            )
        ),
        reply({"type": "text", "text": "Done."}, stop="end_turn"),
    ]
    outcome = run_scenario(scenario, ScriptedProvider(script), fake_search, CONFIG)
    prompt = judge_prompt({"title": "T", "body": "B"}, Reference("cause", None), outcome.events)
    assert "Create it" in prompt and "Done." in prompt
    assert "[concepts/configuration/configmap#configmaps-0]\nt" in prompt
    assert "[made/up#x-0]\n(never retrieved in this run)" in prompt
    assert "<decoy>\nnone\n</decoy>" in prompt


def test_call_judge_parses_structured_output():
    verdict = {"reason": "r", "diagnosis_correct": True, "blames_decoy": None, "grounded": False}

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["output_config"]["format"]["type"] == "json_schema"
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=json.dumps(verdict))],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0),
                to_dict=lambda: {"raw": True},
            )

    result, raw, cost = call_judge(SimpleNamespace(messages=FakeMessages()), "p")
    assert result == Verdict(**verdict)
    assert raw == {"raw": True}
    assert cost == 3.0


def test_judgment_report_lists_disagreements():
    rows = [
        JudgedRow("a", True, None, Verdict("", False, None, True)),
        JudgedRow("b", True, None, Verdict("", True, True, None)),
        JudgedRow("c", False, "ungrounded_citation", Verdict("", True, False, True)),
    ]
    text = judgment_markdown(rows)
    assert "| diagnosis_correct | 1/2 | 1/1 | 2/3 |" in text
    # blames_decoy is null for scenarios without a decoy, so they drop out of the denominator.
    assert "| blames_decoy | 1/1 | 0/1 | 1/2 |" in text
    assert "ungrounded_citation results the judge calls grounded: 1/1" in text
    assert "passed with a wrong diagnosis: a" in text
    assert "passed while blaming the decoy: b" in text
