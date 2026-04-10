import psycopg
import pytest

from arlab.db import apply_schema

TABLES = {
    "chunks",
    "ingest_meta",
    "queries",
    "query_labels",
    "configs",
    "runs",
    "scenario_results",
    "events",
}


def test_apply_schema_creates_all_tables(conn):
    apply_schema(conn)
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    assert {r[0] for r in rows} == TABLES


def test_apply_schema_is_idempotent(conn):
    apply_schema(conn)
    apply_schema(conn)


def test_chunk_tsv_is_generated_from_heading_and_text(conn):
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chunks (id, doc_path, heading, text) VALUES (%s, %s, %s, %s)",
        ("t#1", "t.md", "CrashLoopBackOff", "The container keeps restarting."),
    )
    for term in ("restarting", "crashloopbackoff"):
        hit = conn.execute(
            "SELECT id FROM chunks WHERE tsv @@ plainto_tsquery('english', %s)", (term,)
        ).fetchone()
        assert hit == ("t#1",), term


def test_run_requires_stored_config(conn):
    apply_schema(conn)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute(
            "INSERT INTO runs (id, batch_id, suite, suite_hash, config_hash, index_version, git_sha,"
            " mode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            ("r1", "b1", "core", "s", "missing-hash", "i", "abc123", "replay"),
        )
