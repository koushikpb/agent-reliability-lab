import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

from arlab.agent import AgentResult, run_agent
from arlab.config import RunConfig, system_prompt
from arlab.llm import Provider, RecordingStore, ReplayMiss
from arlab.retrieval import search
from arlab.sandbox import SearchFn, ToolClient, create_tool_server


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    ticket_id: str
    cluster: dict[str, Any]
    expected: dict[str, Any]


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    final_status: str
    result: AgentResult


def load_scenario(path: Path) -> Scenario:
    spec = yaml.safe_load(path.read_text())
    cluster = json.loads((path.parent / spec["cluster_fixture"]).read_text())
    # The ticket lives in the scenario file so its text sits next to the expected outcome.
    ticket = {**spec["ticket"], "status": "open", "notes": []}
    cluster["tickets"] = {ticket["id"]: ticket}
    return Scenario(spec["id"], spec["category"], ticket["id"], cluster, spec["expected"])


def load_suite(suite_dir: Path) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(suite_dir.glob("*.yaml"))]


def run_scenario(
    scenario: Scenario, provider: Provider, search_docs: SearchFn, config: RunConfig
) -> ScenarioOutcome:
    server = create_tool_server(scenario.cluster, search_docs)
    result = run_agent(
        provider,
        ToolClient.in_process(server),
        system_prompt(config),
        scenario.ticket_id,
        config.agent.max_steps,
    )
    final_status = server.state.cluster["tickets"][scenario.ticket_id]["status"]
    return ScenarioOutcome(scenario.id, final_status, result)


def _search_key(index_version: str, retriever: str, k: int, query: str) -> str:
    raw = json.dumps([index_version, retriever, k, query])
    return "search-" + hashlib.sha256(raw.encode()).hexdigest()


def db_search(conn: psycopg.Connection, retriever: str, k: int) -> SearchFn:
    def search_docs(query: str) -> list[dict[str, str]]:
        hits = search(conn, retriever, query, k)
        rows = conn.execute(
            "SELECT id, heading, text FROM chunks WHERE id = ANY(%s)", ([h.chunk_id for h in hits],)
        ).fetchall()
        by_id = {chunk_id: (heading, text) for chunk_id, heading, text in rows}
        return [
            {"chunk_id": h.chunk_id, "heading": by_id[h.chunk_id][0], "text": by_id[h.chunk_id][1]}
            for h in hits
        ]

    return search_docs


def recording_search(
    inner: SearchFn, store: RecordingStore, index_version: str, retriever: str, k: int
) -> SearchFn:
    def search_docs(query: str) -> list[dict[str, str]]:
        hits = inner(query)
        store.put(_search_key(index_version, retriever, k, query), hits)
        return hits

    return search_docs


def replay_search(store: RecordingStore, index_version: str, retriever: str, k: int) -> SearchFn:
    """Recorded search results, so a replayed suite needs no database or embedding model."""

    def search_docs(query: str) -> list[dict[str, str]]:
        key = _search_key(index_version, retriever, k, query)
        hits: list[dict[str, str]] | None = store.get(key)
        if hits is None:
            raise ReplayMiss(key)
        return hits

    return search_docs
