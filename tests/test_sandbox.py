import json

import pytest

from arlab.sandbox import TOOLS, ToolClient, create_tool_server

FIXTURE = {
    "tickets": {"T-1": {"id": "T-1", "title": "pods restarting", "status": "open", "notes": []}},
    "pods": {"shop": [{"name": "web-1", "status": "CrashLoopBackOff", "restarts": 7}]},
    "logs": {"web-1": "Error: configmap shop-config not found"},
}


def fake_search(query):
    return [{"chunk_id": "doc#a-0", "heading": "H", "text": f"result for {query}"}]


@pytest.fixture
def client():
    return ToolClient.in_process(create_tool_server(FIXTURE, fake_search))


def test_tools_read_fixture_state(client):
    pods, err = client.call("list_pods", {"namespace": "shop"})
    assert not err and json.loads(pods)[0]["status"] == "CrashLoopBackOff"
    logs, _ = client.call("get_pod_logs", {"pod": "web-1"})
    assert "configmap" in json.loads(logs)["logs"]
    hits, _ = client.call("search_docs", {"query": "configmap"})
    assert json.loads(hits)[0]["chunk_id"] == "doc#a-0"


def test_update_ticket_mutates_a_copy_of_the_fixture(client):
    client.call("update_ticket", {"id": "T-1", "status": "resolved", "note": "fixed"})
    ticket, _ = client.call("get_ticket", {"id": "T-1"})
    assert json.loads(ticket)["status"] == "resolved"
    assert FIXTURE["tickets"]["T-1"]["status"] == "open"


def test_errors_are_returned_to_the_model_not_raised(client):
    text, err = client.call("get_pod_logs", {"pod": "nope"})
    assert err and "not found" in text
    text, err = client.call("update_ticket", {"id": "T-1", "status": "done", "note": "x"})
    assert err and "422" in text
    text, err = client.call("delete_cluster", {})
    assert err and "unknown tool" in text


def test_escalate_sets_status(client):
    client.call("escalate", {"id": "T-1", "reason": "needs a human"})
    ticket, _ = client.call("get_ticket", {"id": "T-1"})
    assert json.loads(ticket)["status"] == "escalated"


def test_every_tool_schema_is_strict():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
