import json
import time
from dataclasses import asdict
from pathlib import Path

import typer

from arlab.db import apply_schema, connect

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(corpus_dir: Path = Path("data/corpus")) -> None:
    """Chunk and embed the docs corpus into Postgres."""
    from arlab import ingest as ingest_mod

    started = time.perf_counter()
    with connect() as conn:
        apply_schema(conn)
        n = ingest_mod.ingest(conn, corpus_dir)
    elapsed = time.perf_counter() - started
    typer.echo(f"ingested {n} chunks, index {ingest_mod.index_version()}, {elapsed:.1f}s")


@app.command("retrieval-eval")
def retrieval_eval(
    retriever: str = "all", labels: Path = Path("labels/queries.yaml"), out: Path = Path("reports")
) -> None:
    """Score retrievers on the labeled query set."""
    from arlab.labels import load_queries, missing_chunk_ids, store_queries
    from arlab.retrieval import RETRIEVERS
    from arlab.retrieval_eval import evaluate, markdown_table

    queries = load_queries(labels)
    if not queries:
        raise typer.BadParameter(f"no labeled queries in {labels}")
    names = list(RETRIEVERS) if retriever == "all" else [retriever]
    with connect() as conn:
        missing = missing_chunk_ids(conn, queries)
        if missing:
            typer.echo(f"{len(missing)} labeled chunk ids not in the index, e.g. {missing[:3]}")
            raise typer.Exit(1)
        store_queries(conn, queries)
        version = conn.execute("SELECT index_version FROM ingest_meta").fetchone()
        reports = [evaluate(conn, name, queries) for name in names]
    typer.echo(markdown_table(reports))
    out.mkdir(exist_ok=True)
    path = out / f"retrieval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps(
            {
                "index_version": version[0] if version else None,
                "labels": str(labels),
                "reports": [asdict(r) for r in reports],
            },
            indent=2,
        )
    )
    typer.echo(f"wrote {path}")


@app.command()
def search(query: str, k: int = 10) -> None:
    """Pool top-k candidates from every retriever, for labeling queries."""
    from arlab.retrieval import RETRIEVERS
    from arlab.retrieval import search as run_search

    found: dict[str, list[str]] = {}
    with connect() as conn:
        for name in RETRIEVERS:
            try:
                hits = run_search(conn, name, query, k)
            except NotImplementedError:
                typer.echo(f"skipping {name}: rrf_fuse is not written yet")
                continue
            for rank, hit in enumerate(hits, 1):
                found.setdefault(hit.chunk_id, []).append(f"{name}#{rank}")
        rows = conn.execute(
            "SELECT id, heading, left(text, 160) FROM chunks WHERE id = ANY(%s)", (list(found),)
        ).fetchall()
    for chunk_id, heading, text in sorted(rows, key=lambda r: -len(found[r[0]])):
        typer.echo(
            f"{chunk_id}  [{', '.join(found[chunk_id])}]\n  {heading}\n  {' '.join(text.split())}\n"
        )


@app.command()
def run(suite: str, config: str) -> None:
    """Run a scenario suite against one config."""
    raise NotImplementedError


@app.command()
def gate(baseline: str, candidate: str) -> None:
    """Compare two runs and exit nonzero on regression."""
    raise NotImplementedError
