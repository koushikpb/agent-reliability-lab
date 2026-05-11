import json

import httpx2
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
    pods = client.call("list_pods", {"namespace": "shop"})
    assert pods.status == 200 and json.loads(pods.text)[0]["status"] == "CrashLoopBackOff"
    assert "configmap" in json.loads(client.call("get_pod_logs", {"pod": "web-1"}).text)["logs"]
    hits = client.call("search_docs", {"query": "configmap"})
    assert json.loads(hits.text)[0]["chunk_id"] == "doc#a-0"


def test_update_ticket_mutates_a_copy_of_the_fixture(client):
    client.call("update_ticket", {"id": "T-1", "status": "resolved", "note": "fixed"})
    assert json.loads(client.call("get_ticket", {"id": "T-1"}).text)["status"] == "resolved"
    assert FIXTURE["tickets"]["T-1"]["status"] == "open"


def test_errors_are_returned_to_the_model_not_raised(client):
    missing = client.call("get_pod_logs", {"pod": "nope"})
    assert missing.status == 404 and "not found" in missing.text
    bad = client.call("update_ticket", {"id": "T-1", "status": "done", "note": "x"})
    assert bad.status == 422
    unknown = client.call("delete_cluster", {})
    assert unknown.status == 404 and "unknown tool" in unknown.text


def test_escalate_sets_status(client):
    client.call("escalate", {"id": "T-1", "reason": "needs a human"})
    assert json.loads(client.call("get_ticket", {"id": "T-1"}).text)["status"] == "escalated"


def test_escalated_ticket_cannot_be_resolved_afterwards(client):
    client.call("escalate", {"id": "T-1", "reason": "data loss risk"})
    flip = client.call("update_ticket", {"id": "T-1", "status": "resolved", "note": "nvm"})
    assert flip.status == 409
    assert json.loads(client.call("get_ticket", {"id": "T-1"}).text)["status"] == "escalated"


def test_server_exceptions_propagate_instead_of_posing_as_tool_errors():
    # A crash inside the server (database down, replay miss) is not the agent's tool error.
    def broken_search(query):
        raise RuntimeError("index unavailable")

    crashing = ToolClient.in_process(create_tool_server(FIXTURE, broken_search))
    with pytest.raises(RuntimeError, match="index unavailable"):
        crashing.call("search_docs", {"query": "x"})


def transport_client(handler):
    return ToolClient(httpx2.Client(transport=httpx2.MockTransport(handler), base_url="http://t"))


def test_plain_text_error_body_reaches_the_model():
    client = transport_client(lambda request: httpx2.Response(502, text="Bad Gateway"))
    response = client.call("list_pods", {"namespace": "x"})
    assert response.status == 502 and "Bad Gateway" in response.text


def test_timeout_and_malformed_body_are_tool_errors_with_status_zero():
    def timeout(request):
        raise httpx2.ReadTimeout("slow", request=request)

    assert transport_client(timeout).call("list_pods", {"namespace": "x"}).status == 0
    garbled = transport_client(lambda request: httpx2.Response(200, text='{"pods": ['))
    response = garbled.call("list_pods", {"namespace": "x"})
    assert response.status == 0 and "malformed" in response.text


def test_every_tool_schema_is_strict():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def faulty(fault):
    events = []
    server = create_tool_server(FIXTURE, fake_search, fault, events.append)
    return ToolClient.in_process(server), server, events


def test_timeout_applies_the_write_then_loses_the_response():
    client, server, events = faulty("timeout")
    args = {"id": "T-1", "status": "resolved", "note": "done"}
    assert client.call("update_ticket", args).status == 0
    assert events == [{"fault": "timeout", "tool": "update_ticket", "applied_status": "resolved"}]
    # A blind retry writes the note a second time: update_ticket is not idempotent.
    assert client.call("update_ticket", args).status == 200
    assert server.state.cluster["tickets"]["T-1"]["notes"] == ["done", "done"]


def test_rate_limit_tells_the_model_when_to_retry():
    client, _, events = faulty("rate_limit")
    first = client.call("list_pods", {"namespace": "shop"})
    assert (first.status, first.text) == (429, "error 429: rate limited (retry after 2s)")
    assert client.call("list_pods", {"namespace": "shop"}).status == 200
    assert len(events) == 1


def test_malformed_body_fires_once():
    client, _, _ = faulty("malformed")
    assert client.call("get_pod_logs", {"pod": "web-1"}).text.startswith("error: malformed")
    assert "configmap" in client.call("get_pod_logs", {"pod": "web-1"}).text


def test_stale_snapshot_hides_the_incident_once():
    client, _, _ = faulty("stale")
    stale = json.loads(client.call("list_pods", {"namespace": "shop"}).text)
    assert stale == [{"name": "web-1", "status": "Running", "restarts": 0, "ready": True}]
    fresh = json.loads(client.call("list_pods", {"namespace": "shop"}).text)
    assert fresh[0]["status"] == "CrashLoopBackOff"


def test_empty_search_is_empty_on_every_call():
    client, _, events = faulty("empty_search")
    assert client.call("search_docs", {"query": "a"}).text == "[]"
    assert client.call("search_docs", {"query": "b"}).text == "[]"
    assert len(events) == 2


def test_fault_leaves_other_tools_alone():
    client, _, events = faulty("malformed")
    assert client.call("list_pods", {"namespace": "shop"}).status == 200
    assert events == []


def test_unknown_fault_is_rejected():
    with pytest.raises(ValueError):
        create_tool_server(FIXTURE, fake_search, "flood")


def test_stale_fault_waits_for_a_real_namespace():
    client, _, events = faulty("stale")
    assert client.call("list_pods", {"namespace": "nope"}).status == 404
    assert (
        json.loads(client.call("list_pods", {"namespace": "shop"}).text)[0]["status"] == "Running"
    )
    assert len(events) == 1
