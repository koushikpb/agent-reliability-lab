from pathlib import Path

from arlab.config import config_hash, load_config, tool_schemas
from arlab.sandbox import TOOLS, ToolClient, create_tool_server


def test_unset_seeded_bug_keeps_existing_hashes():
    # Recorded before the field existed; a changed hash would orphan every recording.
    assert config_hash(load_config(Path("configs/baseline.yaml"))) == "a1c14122b0ea4e04"


def test_seeded_bug_changes_the_hash():
    base = load_config(Path("configs/haiku-v1.yaml"))
    seeded = load_config(Path("configs/seeded/wrong-chunks.yaml"))
    assert config_hash(base) != config_hash(seeded)


def test_schema_drift_renames_the_argument_the_server_reads():
    drifted = load_config(Path("configs/seeded/schema-drift.yaml"))
    [logs] = [t for t in tool_schemas(drifted) if t["name"] == "get_pod_logs"]
    assert list(logs["input_schema"]["properties"]) == ["pod_name"]
    assert tool_schemas(load_config(Path("configs/haiku-v1.yaml"))) is TOOLS
    fixture = {"tickets": {}, "pods": {}, "logs": {"web-1": "x"}}
    client = ToolClient.in_process(create_tool_server(fixture, lambda q: []))
    response = client.call("get_pod_logs", {"pod_name": "web-1"})
    assert (response.status, response.text) == (422, "error 422: missing argument 'pod'")
