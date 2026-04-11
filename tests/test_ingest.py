import numpy as np

from arlab.chunking import Chunk
from arlab.db import apply_schema
from arlab.ingest import index_version, load_chunks


def fake_chunks(n: int) -> list[Chunk]:
    return [Chunk(f"d#s-{i}", "d", "T > S", f"text number {i}") for i in range(n)]


def test_load_replaces_chunks_builds_hnsw_and_records_version(conn):
    apply_schema(conn)
    load_chunks(conn, fake_chunks(5), np.ones((5, 384), dtype=np.float32))
    load_chunks(conn, fake_chunks(3), np.ones((3, 384), dtype=np.float32))

    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (3,)
    indexdef = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_idx'"
    ).fetchone()
    assert indexdef is not None and "hnsw" in indexdef[0]
    assert conn.execute("SELECT index_version, n_chunks FROM ingest_meta").fetchall() == [
        (index_version(), 3)
    ]


def test_index_version_is_stable():
    assert index_version() == index_version()
    assert len(index_version()) == 12
