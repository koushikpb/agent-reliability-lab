import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from arlab.db import connect, database_url

TEST_DB = "arlab_test"


# The dev DB already has the tables from docker init, so tests run in a fresh empty database.
# Otherwise they would pass even if apply_schema did nothing. A throwaway schema is not enough:
# with public on the search_path, a missing test table falls through to the dev table.
@pytest.fixture
def conn():
    try:
        admin = connect()
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable, run `docker compose up -d`: {e}")
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    conn = connect(make_conninfo(database_url(), dbname=TEST_DB))
    yield conn
    conn.close()
    admin.execute(f"DROP DATABASE {TEST_DB}")
    admin.close()
