import pytest

from arlab.db import apply_schema
from arlab.labels import load_queries, missing_chunk_ids, store_queries

GOOD = """
- id: q1
  text: pod restarts
  relevant:
    pods#crash-0: 2
    pods#crash-1: 1
"""


def test_load_queries(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(GOOD)
    [q] = load_queries(path)
    assert q.grades == {"pods#crash-0": 2, "pods#crash-1": 1}


@pytest.mark.parametrize(
    "bad",
    [
        GOOD + GOOD,  # duplicate id
        "- id: q1\n  text: t\n  relevant: {}\n",
        "- id: q1\n  text: t\n  relevant:\n    a#b-0: 3\n",
    ],
)
def test_load_queries_rejects_bad_labels(tmp_path, bad):
    path = tmp_path / "q.yaml"
    path.write_text(bad)
    with pytest.raises(ValueError):
        load_queries(path)


def test_missing_chunk_ids_and_store(conn, tmp_path):
    apply_schema(conn)
    conn.execute(
        "INSERT INTO chunks (id, doc_path, heading, text) VALUES ('pods#crash-0', 'p', 'h', 't')"
    )
    path = tmp_path / "q.yaml"
    path.write_text(GOOD)
    queries = load_queries(path)
    assert missing_chunk_ids(conn, queries) == ["pods#crash-1"]
    store_queries(conn, queries)
    store_queries(conn, queries)
    assert conn.execute("SELECT count(*) FROM query_labels").fetchone() == (2,)
