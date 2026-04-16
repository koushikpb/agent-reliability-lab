from dataclasses import dataclass
from pathlib import Path

import psycopg
import yaml


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    text: str
    grades: dict[str, int]  # chunk id -> 1 (relevant) or 2 (highly relevant)
    # Every chunk id a person looked at, relevant or not. Without it, a chunk nobody judged and a
    # chunk judged non-relevant look the same, and unjudged results silently count as misses.
    judged: frozenset[str] | None = None


def load_queries(path: Path) -> list[LabeledQuery]:
    raw = yaml.safe_load(path.read_text()) or []
    queries = [
        LabeledQuery(
            q["id"],
            q["text"],
            dict(q["relevant"]),
            frozenset(q["judged"]) if "judged" in q else None,
        )
        for q in raw
    ]
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate query ids in {path}")
    for q in queries:
        if not q.grades:
            raise ValueError(f"query {q.id} has no relevant chunks")
        if any(g not in (1, 2) for g in q.grades.values()):
            raise ValueError(f"query {q.id} has a grade outside 1-2")
        if q.judged is not None and not set(q.grades) <= q.judged:
            raise ValueError(f"query {q.id} has relevant ids missing from judged")
    return queries


def missing_chunk_ids(conn: psycopg.Connection, queries: list[LabeledQuery]) -> list[str]:
    """Labeled ids with no chunk: a re-chunk renamed them, and recall would drop without an error."""
    labeled = sorted({cid for q in queries for cid in q.grades})
    rows = conn.execute("SELECT id FROM chunks WHERE id = ANY(%s)", (labeled,)).fetchall()
    found = {r[0] for r in rows}
    return [cid for cid in labeled if cid not in found]


def store_queries(conn: psycopg.Connection, queries: list[LabeledQuery]) -> None:
    # The YAML file is the source of truth (reviewed in git); the tables are a copy for SQL.
    with conn.transaction():
        conn.execute("DELETE FROM query_labels")
        conn.execute("DELETE FROM queries")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO queries (id, text) VALUES (%s, %s)", [(q.id, q.text) for q in queries]
            )
            cur.executemany(
                "INSERT INTO query_labels (query_id, chunk_id, grade) VALUES (%s, %s, %s)",
                [(q.id, cid, g) for q in queries for cid, g in q.grades.items()],
            )
