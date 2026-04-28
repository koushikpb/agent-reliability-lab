import hashlib
import json
import subprocess
import uuid
from pathlib import Path

import psycopg
import yaml
from psycopg.types.json import Jsonb

from arlab.config import RunConfig, config_hash, resolved_params
from arlab.results import ScenarioRow
from arlab.tracing import Event


def git_sha() -> str:
    """HEAD, marked -dirty when the tree has uncommitted changes, which make it unreproducible."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # Untracked files count: a new module the run imports changes behavior just as an edit does.
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def suite_hash(suite_dir: Path) -> str:
    """Hash of the scenario files and the fixtures they reference, so an edited expectation or
    fixture shows up as a new suite. Other files (editor swap files, .DS_Store) don't count."""
    digest = hashlib.sha256()
    for path in sorted(suite_dir.glob("*.yaml")):
        fixture = path.parent / yaml.safe_load(path.read_text())["cluster_fixture"]
        for part in (path, fixture):
            digest.update(part.name.encode())
            digest.update(part.read_bytes())
    return digest.hexdigest()[:16]


def upsert_config(conn: psycopg.Connection, config: RunConfig) -> str:
    key = config_hash(config)
    conn.execute(
        "INSERT INTO configs (hash, model, prompt_version, retriever, params)"
        " VALUES (%s, %s, %s, %s, %s) ON CONFLICT (hash) DO NOTHING",
        (
            key,
            config.model,
            config.prompt_version,
            config.retriever,
            Jsonb(resolved_params(config)),
        ),
    )
    return key


def start_run(
    conn: psycopg.Connection,
    *,
    batch_id: str,
    suite: str,
    suite_hash: str,
    config: RunConfig,
    index_version: str,
    mode: str,
) -> str:
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO runs (id, batch_id, suite, suite_hash, config_hash, index_version, git_sha,"
        " mode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            run_id,
            batch_id,
            suite,
            suite_hash,
            upsert_config(conn, config),
            index_version,
            git_sha(),
            mode,
        ),
    )
    return run_id


def finish_run(conn: psycopg.Connection, run_id: str, status: str) -> None:
    conn.execute("UPDATE runs SET status = %s, finished_at = now() WHERE id = %s", (status, run_id))


def record_scenario(
    conn: psycopg.Connection, run_id: str, row: ScenarioRow, events: list[Event]
) -> None:
    # One transaction per scenario: a crash mid-run leaves whole scenarios, never half a trace.
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO events (run_id, scenario_id, fault, seq, span_id, parent_span_id, kind,"
            " payload, started_at, duration_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    run_id,
                    row.scenario_id,
                    row.fault,
                    e.seq,
                    e.span_id,
                    e.parent_span_id,
                    e.kind,
                    Jsonb(e.payload),
                    e.started_at,
                    e.duration_ms,
                )
                for e in events
            ],
        )
        cur.execute(
            "INSERT INTO scenario_results (run_id, scenario_id, fault, success, final_status,"
            " tool_errors, invalid_calls, redundant_calls, cited_ids, latency_ms, input_tokens,"
            " output_tokens, cost_usd, failure_class)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                run_id,
                row.scenario_id,
                row.fault,
                row.success,
                row.final_status,
                row.tool_errors,
                row.invalid_calls,
                row.redundant_calls,
                row.cited_ids,
                row.latency_ms,
                row.input_tokens,
                row.output_tokens,
                row.cost_usd,
                row.failure_class,
            ),
        )


def record_events(
    conn: psycopg.Connection, run_id: str, scenario_id: str, fault: str, events: list[Event]
) -> None:
    """Events of a scenario that crashed: kept without a result row, so the crash is explainable."""
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO events (run_id, scenario_id, fault, seq, span_id, parent_span_id, kind,"
            " payload, started_at, duration_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    run_id,
                    scenario_id,
                    fault,
                    e.seq,
                    e.span_id,
                    e.parent_span_id,
                    e.kind,
                    Jsonb(e.payload),
                    e.started_at,
                    e.duration_ms,
                )
                for e in events
            ],
        )


def load_events(conn: psycopg.Connection, run_id: str, scenario_id: str, fault: str) -> list[Event]:
    rows = conn.execute(
        "SELECT seq, span_id, parent_span_id, kind, payload, started_at, duration_ms FROM events"
        " WHERE run_id = %s AND scenario_id = %s AND fault = %s ORDER BY seq",
        (run_id, scenario_id, fault),
    ).fetchall()
    return [Event(*r) for r in rows]


def load_row(conn: psycopg.Connection, run_id: str, scenario_id: str, fault: str) -> ScenarioRow:
    row = conn.execute(
        "SELECT scenario_id, fault, success, final_status, tool_errors, invalid_calls,"
        " redundant_calls, cited_ids, latency_ms, input_tokens, output_tokens, cost_usd::float8,"
        " failure_class FROM scenario_results WHERE run_id = %s AND scenario_id = %s AND fault = %s",
        (run_id, scenario_id, fault),
    ).fetchone()
    if row is None:
        raise LookupError(f"no result for {run_id}/{scenario_id}/{fault}")
    return ScenarioRow(*row)


def write_store_meta(root: Path, index_version: str) -> None:
    """Saved with recordings so replay knows the index version without a database."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps({"index_version": index_version}))


def read_store_meta(root: Path) -> str | None:
    path = root / "meta.json"
    return json.loads(path.read_text())["index_version"] if path.exists() else None
