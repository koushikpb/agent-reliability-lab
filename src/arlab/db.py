import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

DEFAULT_URL = "postgresql://arlab:arlab@localhost:5433/arlab"
# Resolved from the source tree; this is a dev tool, not an installed package.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or database_url(), autocommit=True)
    # The vector type exists only after the schema is applied, e.g. in a fresh test database.
    if conn.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'").fetchone():
        register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())
    register_vector(conn)
