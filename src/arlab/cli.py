import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest() -> None:
    """Chunk and embed the docs corpus into Postgres."""
    raise NotImplementedError


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
