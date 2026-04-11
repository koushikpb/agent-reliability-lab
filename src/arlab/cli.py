import time
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
def retrieval_eval(retriever: str = "hybrid", k: int = 10) -> None:
    """Score a retriever on the labeled query set."""
    raise NotImplementedError


@app.command()
def run(suite: str, config: str) -> None:
    """Run a scenario suite against one config."""
    raise NotImplementedError


@app.command()
def gate(baseline: str, candidate: str) -> None:
    """Compare two runs and exit nonzero on regression."""
    raise NotImplementedError
