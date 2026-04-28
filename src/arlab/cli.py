import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

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
    from arlab.retrieval import settings as retrieval_settings
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
                "retrieval_settings": retrieval_settings(),
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
    # Paste as the query's judged list after labeling, so unjudged@10 can be measured.
    typer.echo("judged:\n" + "\n".join(f"  - {cid}" for cid in sorted(found)))


@app.command("hnsw-recall")
def hnsw_recall_cmd(labels: Path = Path("labels/queries.yaml"), k: int = 50) -> None:
    """Overlap of HNSW top-k with exact search, for several ef_search values."""
    from arlab.labels import load_queries
    from arlab.retrieval_eval import hnsw_recall

    texts = [q.text for q in load_queries(labels)]
    if not texts:
        raise typer.BadParameter(f"no queries in {labels}")
    with connect() as conn:
        for ef, overlap in hnsw_recall(conn, texts, k, [k, 100, 200]).items():
            typer.echo(f"ef_search={ef}: recall@{k} vs exact = {overlap:.3f}")


@app.command()
def run(
    suite: Annotated[Path, typer.Option(help="scenario suite directory")],
    config: Annotated[Path, typer.Option(help="run config YAML")],
    mode: Annotated[str, typer.Option(help="replay (no spend), record, or live")] = "replay",
    recordings: Path = Path("recordings"),
    save: Annotated[bool, typer.Option(help="store the run and its events in Postgres")] = True,
    batch: Annotated[str, typer.Option(help="batch id grouping repeats of one config")] = "",
) -> None:
    """Run a scenario suite against one config."""
    import uuid

    from arlab.config import config_hash, load_config
    from arlab.llm import AnthropicProvider, ReplayMiss
    from arlab.runner import ScenarioCrashed, load_suite, run_scenario, scenario_io
    from arlab.store import (
        finish_run,
        read_store_meta,
        record_events,
        record_scenario,
        start_run,
        suite_hash,
        write_store_meta,
    )

    if mode not in ("replay", "record", "live"):
        raise typer.BadParameter("mode must be replay, record or live")
    cfg = load_config(config)
    scenarios = load_suite(suite)
    # Recordings are per suite and config hash: a changed prompt or tool schema can't replay old ones.
    store_root = recordings / suite.name / config_hash(cfg)
    # Replay without --save needs no database, which is how CI runs.
    conn = connect() if mode != "replay" or save else None
    if mode == "replay":
        index_version = read_store_meta(store_root)
        if index_version is None:
            typer.echo(f"no recordings at {store_root}; record with --mode record (costs money)")
            raise typer.Exit(1)
    else:
        assert conn is not None
        ingest = conn.execute("SELECT index_version FROM ingest_meta").fetchone()
        if ingest is None:
            raise typer.BadParameter("no ingested index; run `arlab ingest` first")
        index_version = ingest[0]
        if mode == "record":
            write_store_meta(store_root, index_version)
    live = (
        None if mode == "replay" else AnthropicProvider(cfg.model, cfg.effort, cfg.agent.max_tokens)
    )
    run_id = None
    if conn is not None and save:
        run_id = start_run(
            conn,
            batch_id=batch or uuid.uuid4().hex,
            suite=suite.name,
            suite_hash=suite_hash(suite),
            config=cfg,
            index_version=index_version,
            mode=mode,
        )
    typer.echo(f"config {config_hash(cfg)}  index {index_version}  mode {mode}  run {run_id}")
    total_cost = 0.0
    finished = False
    try:
        for scenario in scenarios:
            provider, search_docs = scenario_io(
                mode, store_root, scenario.id, cfg, index_version, live, conn
            )
            try:
                outcome = run_scenario(scenario, provider, search_docs, cfg)
            except ScenarioCrashed as crash:
                if run_id is not None and conn is not None:
                    record_events(conn, run_id, scenario.id, "none", crash.events)
                if isinstance(crash.__cause__, ReplayMiss):
                    typer.echo(
                        f"{scenario.id}: no recording for this request under {store_root}. Record"
                        " it with --mode record, which calls the live model and costs money."
                    )
                    raise typer.Exit(1) from None
                raise
            if run_id is not None and conn is not None:
                record_scenario(conn, run_id, outcome.row, outcome.events)
            row = outcome.row
            total_cost += row.cost_usd
            expected = scenario.expected.final_status
            typer.echo(
                f"{scenario.id:32} {'ok' if row.success else row.failure_class:22} expected={expected}"
                f" got={row.final_status} stop={outcome.result.stop} steps={outcome.result.steps}"
                f" invalid={row.invalid_calls} tool_errors={row.tool_errors}"
                f" cited={len(row.cited_ids)} tokens={row.input_tokens}/{row.output_tokens}"
                f" ${row.cost_usd:.4f}"
            )
        finished = True
    finally:
        # A run that stopped early is marked failed, so it is never compared as if complete.
        if run_id is not None and conn is not None:
            finish_run(conn, run_id, "finished" if finished else "failed")
        if conn is not None:
            conn.close()
    # Replayed runs report what the recorded run cost, not new spend.
    typer.echo(
        f"total ${total_cost:.4f}" + (" (recorded cost, no new spend)" if mode == "replay" else "")
    )


@app.command()
def gate(baseline: str, candidate: str) -> None:
    """Compare two runs and exit nonzero on regression."""
    raise NotImplementedError


@app.command("check-suite")
def check_suite(suite: Path) -> None:
    """Validate every scenario in a suite, and that its chunk ids exist in the current index."""
    from pydantic import ValidationError

    from arlab.runner import load_scenario

    problems = []
    scenarios = []
    for path in sorted(suite.glob("*.yaml")):
        try:
            scenarios.append(load_scenario(path))
        except (ValidationError, KeyError, FileNotFoundError) as e:
            problems.append(f"{path.name}: {e}")
    ids = [s.id for s in scenarios]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    labeled = {cid for s in scenarios for cid in (s.expected.relevant_chunk_ids or [])}
    if labeled:
        with connect() as conn:
            found = {
                r[0]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE id = ANY(%s)", (sorted(labeled),)
                ).fetchall()
            }
        problems += [f"chunk id not in index: {cid}" for cid in sorted(labeled - found)]
    for problem in problems:
        typer.echo(problem)
    typer.echo(f"{len(scenarios)} scenarios, {len(problems)} problems")
    if problems:
        raise typer.Exit(1)
