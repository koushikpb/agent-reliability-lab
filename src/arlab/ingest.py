import functools
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from arlab import corpus
from arlab.chunking import CHUNKER_VERSION, MAX_WORDS, Chunk, chunk_document

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# pgvector defaults, set explicitly so they are part of the index version. HNSW recall against
# exact search, and the query-time ef_search, are measured in milestone 2.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

Embedder = Callable[[Sequence[str]], np.ndarray]


def index_version() -> str:
    params = {
        "corpus_commit": corpus.COMMIT,
        "sections": corpus.SECTIONS,
        "max_words": MAX_WORDS,
        "chunker_version": CHUNKER_VERSION,
        "embed_model": EMBED_MODEL,
        "embed_revision": EMBED_REVISION,
        "hnsw": [HNSW_M, HNSW_EF_CONSTRUCTION],
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def embedding_input(chunk: Chunk) -> str:
    # The heading path carries context ("Debug Pods > My pod stays pending") the body often lacks.
    return f"{chunk.heading}\n{chunk.text}"


@functools.cache
def _embed_model() -> "SentenceTransformer":
    # Imported here because loading torch takes seconds and most commands don't need it.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    # Normalized so cosine distance and inner product rank the same.
    return _embed_model().encode(list(texts), batch_size=64, normalize_embeddings=True)


def load_chunks(conn: psycopg.Connection, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
    """Replace all chunks in one transaction, then build HNSW over the loaded rows."""
    register_vector(conn)
    with conn.transaction():
        # Dropped first so inserts don't each update the graph.
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        conn.execute("DELETE FROM chunks")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, doc_path, heading, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (c.id, c.doc_path, c.heading, c.text, e)
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )
        conn.execute(
            "CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
            f" WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
        conn.execute("DELETE FROM ingest_meta")
        conn.execute(
            "INSERT INTO ingest_meta (index_version, n_chunks) VALUES (%s, %s)",
            (index_version(), len(chunks)),
        )


def ingest(conn: psycopg.Connection, corpus_dir: Path, embed: Embedder = embed_texts) -> int:
    if not (corpus_dir / corpus.DOCS_ROOT).exists():
        corpus.fetch_corpus(corpus_dir)
    chunks = [c for path, raw in corpus.iter_docs(corpus_dir) for c in chunk_document(path, raw)]
    load_chunks(conn, chunks, embed([embedding_input(c) for c in chunks]))
    return len(chunks)
