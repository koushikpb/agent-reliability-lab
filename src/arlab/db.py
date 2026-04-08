import os
from pathlib import Path

import psycopg

DEFAULT_URL = "postgresql://arlab:arlab@localhost:5433/arlab"
# Resolved from the source tree; this is a dev tool, not an installed package.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or database_url(), autocommit=True)


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())
