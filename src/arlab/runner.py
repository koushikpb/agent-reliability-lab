import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

from arlab.agent import AgentResult, run_agent
from arlab.checks import Expected
from arlab.config import RunConfig, system_prompt, tool_schemas
from arlab.llm import Provider, RecordingProvider, RecordingStore, ReplayMiss, ReplayProvider
from arlab.results import ScenarioRow, row_from_result
from arlab.retrieval import search
from arlab.sandbox import SearchFn, ToolClient, create_tool_server
from arlab.tracing import Event, make_tracer, record


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    ticket_id: str
    cluster: dict[str, Any]
    expected: Expected


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    final_status: str
    result: AgentResult
    events: list[Event]
    row: ScenarioRow


class ScenarioCrashed(Exception):
    """A scenario raised before finishing. Carries the events traced so far, so the crash can be
    stored and explained; the original exception is the __cause__."""

    def __init__(self, scenario_id: str, events: list[Event]) -> None:
        super().__init__(scenario_id)
        self.scenario_id = scenario_id
        self.events = events


def load_scenario(path: Path) -> Scenario:
    spec = yaml.safe_load(path.read_text())
    cluster = json.loads((path.parent / spec["cluster_fixture"]).read_text())
    # The ticket lives in the scenario file so its text sits next to the expected outcome.
    ticket = {**spec["ticket"], "status": "open", "notes": []}
    cluster["tickets"] = {ticket["id"]: ticket}
    # Validated at load, so a typo in an expectation fails before any model call is paid for.
    expected = Expected.model_validate(spec["expected"])
    return Scenario(spec["id"], spec["category"], ticket["id"], cluster, expected)


def load_suite(suite_dir: Path) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(suite_dir.glob("*.yaml"))]


def run_scenario(
    scenario: Scenario,
    provider: Provider,
    search_docs: SearchFn,
    config: RunConfig,
    fault: str = "none",
) -> ScenarioOutcome:
    tracer, collector = make_tracer()

    def traced_search(query: str) -> list[dict[str, str]]:
        with tracer.start_as_current_span("retrieval") as span:
            hits = search_docs(query)
            record(span, "retrieval", {"query": query, "chunk_ids": [h["chunk_id"] for h in hits]})
        return hits

    def traced_fault(payload: dict[str, Any]) -> None:
        with tracer.start_as_current_span("fault") as span:
            record(span, "fault", payload)

    server = create_tool_server(scenario.cluster, traced_search, fault, traced_fault)
    try:
        with tracer.start_as_current_span("scenario") as span:
            record(span, "scenario", {"scenario_id": scenario.id, "ticket_id": scenario.ticket_id})
            result = run_agent(
                provider,
                ToolClient.in_process(server),
                system_prompt(config),
                scenario.ticket_id,
                config.agent.max_steps,
                tracer,
                tool_schemas(config),
            )
            record(
                span,
                "scenario",
                {"scenario_id": scenario.id, "ticket_id": scenario.ticket_id, "stop": result.stop},
            )
    except Exception as e:
        # Not swallowed: re-raised with the partial trace attached.
        raise ScenarioCrashed(scenario.id, collector.drain()) from e
    events = collector.drain()
    final_status = server.state.cluster["tickets"][scenario.ticket_id]["status"]
    [scenario_event] = [e for e in events if e.kind == "scenario"]
    row = row_from_result(
        scenario.id, fault, scenario.expected, final_status, result, scenario_event.duration_ms
    )
    return ScenarioOutcome(scenario.id, final_status, result, events, row)


def _search_key(index_version: str, retriever: str, k: int, query: str) -> str:
    raw = json.dumps([index_version, retriever, k, query])
    return "search-" + hashlib.sha256(raw.encode()).hexdigest()


def db_search(conn: psycopg.Connection, retriever: str, k: int, skip: int = 0) -> SearchFn:
    """`skip` drops the top results; only the wrong_chunks seeded bug sets it."""

    def search_docs(query: str) -> list[dict[str, str]]:
        hits = search(conn, retriever, query, k + skip)[skip:]
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


def scenario_io(
    mode: str,
    store_root: Path,
    store_key: str,
    config: RunConfig,
    index_version: str,
    live: Provider | None,
    conn: psycopg.Connection | None,
) -> tuple[Provider, SearchFn]:
    """Provider and search for one scenario. Each scenario records into its own directory: the
    first request is the same for every scenario with the same ticket id, so a shared store would
    let one scenario replay another's responses. A faulted run is keyed `<scenario>@<fault>`."""
    store = RecordingStore(store_root / store_key)
    retriever, k = config.retriever, config.retriever_params.k
    if mode == "replay":
        return ReplayProvider(config.model, store), replay_search(
            store, index_version, retriever, k
        )
    if live is None or conn is None:
        raise ValueError(f"mode {mode} needs a live provider and a database connection")
    search_docs = db_search(conn, retriever, k, k if config.seeded_bug == "wrong_chunks" else 0)
    if mode == "live":
        return live, search_docs
    return (
        RecordingProvider(live, store),
        recording_search(search_docs, store, index_version, retriever, k),
    )
