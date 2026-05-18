import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from arlab.db import apply_schema, connect
from arlab.report import ResultRow

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
    retriever: str = "all",
    labels: Path = Path("labels/queries.yaml"),
    out: Path = Path("reports"),
    min_grade: Annotated[
        int, typer.Option(help="2 scores against highly relevant labels only")
    ] = 1,
) -> None:
    """Score retrievers on the labeled query set."""
    import hashlib

    from arlab.labels import at_min_grade, load_queries, missing_chunk_ids, store_queries
    from arlab.retrieval import RETRIEVERS
    from arlab.retrieval import settings as retrieval_settings
    from arlab.retrieval_eval import evaluate, markdown_table
    from arlab.store import git_sha

    all_queries = load_queries(labels)
    queries = at_min_grade(all_queries, min_grade)
    if not queries:
        raise typer.BadParameter(f"no labeled queries in {labels} at grade >= {min_grade}")
    if len(queries) < len(all_queries):
        typer.echo(
            f"{len(all_queries) - len(queries)} queries have no label at grade >= {min_grade}"
        )
    names = list(RETRIEVERS) if retriever == "all" else [retriever]
    with connect() as conn:
        missing = missing_chunk_ids(conn, queries)
        if missing:
            typer.echo(f"{len(missing)} labeled chunk ids not in the index, e.g. {missing[:3]}")
            raise typer.Exit(1)
        store_queries(conn, all_queries)
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
                # A path alone can't tell two reports apart after a label edit.
                "labels": str(labels),
                "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                "min_grade": min_grade,
                "git_sha": git_sha(),
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
            for rank, hit in enumerate(run_search(conn, name, query, k), 1):
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
    only: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="run only these scenario ids (repeatable)"),
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="also write result rows to this JSON file")
    ] = None,
    fault: Annotated[
        str, typer.Option(help="tool-server fault to inject in every scenario")
    ] = "none",
) -> None:
    """Run a scenario suite against one config."""
    import uuid

    from arlab.compare import write_rows
    from arlab.config import config_hash, load_config
    from arlab.llm import AnthropicProvider, ReplayMiss
    from arlab.report import ResultRow
    from arlab.runner import ScenarioCrashed, load_suite, run_scenario, scenario_io
    from arlab.sandbox import FAULTS
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
    if fault not in FAULTS:
        raise typer.BadParameter(f"fault must be one of {FAULTS}")
    cfg = load_config(config)
    scenarios = load_suite(suite)
    if only:
        unknown = set(only) - {s.id for s in scenarios}
        if unknown:
            raise typer.BadParameter(f"not in {suite}: {sorted(unknown)}")
        # The run keeps the whole suite's hash: the subset is still scored against these files.
        scenarios = [s for s in scenarios if s.id in only]
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
    results: list[ResultRow] = []
    finished = False
    try:
        for scenario in scenarios:
            store_key = scenario.id if fault == "none" else f"{scenario.id}@{fault}"
            provider, search_docs = scenario_io(
                mode, store_root, store_key, cfg, index_version, live, conn
            )
            try:
                outcome = run_scenario(scenario, provider, search_docs, cfg, fault)
            except ScenarioCrashed as crash:
                if run_id is not None and conn is not None:
                    record_events(conn, run_id, scenario.id, fault, crash.events)
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
            results.append(
                ResultRow(
                    run_id or "unsaved",
                    row.scenario_id,
                    row.fault,
                    row.success,
                    row.failure_class,
                    row.latency_ms,
                    row.cost_usd,
                )
            )
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
    if out is not None:
        write_rows(out, results, suite_hash(suite), config_hash(cfg))
    # Replayed runs report what the recorded run cost, not new spend.
    typer.echo(
        f"total ${total_cost:.4f}" + (" (recorded cost, no new spend)" if mode == "replay" else "")
    )


@app.command()
def report(
    batch: Annotated[str, typer.Option(help="batch id grouping repeats of one config")],
    suite: Annotated[Path, typer.Option(help="suite directory, for scenario categories")],
) -> None:
    """Pass rate per run and per category, failure classes, flaky scenarios, cost."""
    from arlab.report import load_batch, markdown, summarize
    from arlab.runner import load_suite
    from arlab.store import suite_hash

    categories = {s.id: s.category for s in load_suite(suite)}
    with connect() as conn:
        rows, hashes = load_batch(conn, batch)
    if not rows:
        raise typer.BadParameter(f"no finished runs in batch {batch}")
    # Categories come from the files now; if the files changed since the run, they may not match.
    if hashes != {suite_hash(suite)}:
        typer.echo(
            f"warning: batch ran with suite hash(es) {sorted(hashes)}, files are now {suite_hash(suite)}"
        )
    typer.echo(markdown(summarize(rows, categories)))


@app.command()
def rescore(
    batch: Annotated[str, typer.Option(help="batch whose finished runs to re-score")],
    suite: Annotated[Path, typer.Option(help="suite directory with the current expectations")],
    config: Annotated[Path, typer.Option(help="the config the batch ran with")],
    into: Annotated[str, typer.Option(help="batch id for the re-scored runs")] = "",
) -> None:
    """Re-score stored traces under the current expectations, without model calls."""
    from arlab.config import load_config
    from arlab.runner import load_suite
    from arlab.store import rescore_run

    expected = {s.id: s.expected for s in load_suite(suite)}
    target = into or f"{batch}-rescored"
    with connect() as conn:
        sources = conn.execute(
            "SELECT id FROM runs WHERE batch_id = %s AND status IN ('finished', 'interrupted') ORDER BY started_at",
            (batch,),
        ).fetchall()
        if not sources:
            raise typer.BadParameter(f"no finished runs in batch {batch}")
        for (source,) in sources:
            run_id, missing = rescore_run(
                conn, source, expected, batch_id=target, suite=suite, config=load_config(config)
            )
            note = f", no trace for {missing}" if missing else ""
            typer.echo(f"{source} -> {run_id}{note}")
    typer.echo(f"re-scored into batch {target}")


def _load_results(source: str) -> tuple[list[ResultRow], set[str]]:
    """Rows from a JSON file written by `run --out`, or from a batch in Postgres."""
    from arlab.compare import read_rows
    from arlab.report import load_batch

    if source.endswith(".json"):
        rows, suite_hash = read_rows(Path(source))
        return rows, {suite_hash}
    with connect() as conn:
        return load_batch(conn, source)


@app.command()
def gate(
    baseline: Annotated[str, typer.Option(help="batch id, or a JSON file from `run --out`")],
    candidate: Annotated[str, typer.Option(help="batch id, or a JSON file from `run --out`")],
    suite: Annotated[Path, typer.Option(help="suite directory, for scenario categories")],
    max_drop: Annotated[
        float, typer.Option(help="pass-rate drop to tolerate, as a fraction of scenarios")
    ] = 0.0,
    max_cost_increase: Annotated[
        float, typer.Option(help="relative rise in cost per successful task to tolerate")
    ] = 0.25,
    strict: Annotated[
        bool, typer.Option(help="block on any scenario that got worse; for replays, not live runs")
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="compare only these scenario ids (repeatable)"),
    ] = None,
) -> None:
    """Compare a candidate with a baseline and exit nonzero on regression."""
    from arlab.compare import compare, gate_failures, markdown
    from arlab.runner import load_suite
    from arlab.store import suite_hash

    base_rows, base_hashes = _load_results(baseline)
    cand_rows, cand_hashes = _load_results(candidate)
    if only:
        base_rows = [r for r in base_rows if r.scenario_id in only]
        cand_rows = [r for r in cand_rows if r.scenario_id in only]
    if not base_rows or not cand_rows:
        raise typer.BadParameter("baseline or candidate has no finished results")
    # Rows scored against different expectations differ for reasons that aren't the config.
    if len(base_hashes | cand_hashes) != 1:
        typer.echo(f"suite hashes differ: baseline {base_hashes}, candidate {cand_hashes}")
        typer.echo("re-score both with `arlab rescore` against the same suite first")
        raise typer.Exit(2)
    # The suite only supplies categories, but a different suite would mislabel them silently.
    if suite_hash(suite) not in base_hashes:
        typer.echo(
            f"{suite} has hash {suite_hash(suite)}, the results were scored with {base_hashes}"
        )
        raise typer.Exit(2)
    categories = {s.id: s.category for s in load_suite(suite)}
    try:
        comparison = compare(base_rows, cand_rows, categories)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(markdown(comparison))
    reasons = gate_failures(comparison, max_drop, max_cost_increase, strict)
    if reasons:
        typer.echo("\nBLOCKED")
        for reason in reasons:
            typer.echo(f"- {reason}")
        raise typer.Exit(1)
    typer.echo("\npassed")


@app.command()
def faults(
    batch: Annotated[str, typer.Option(help="batch with the faulted runs")],
    clean_batch: Annotated[
        str, typer.Option(help="batch with the same config's unfaulted runs, if not in --batch")
    ] = "",
) -> None:
    """Pass rate per injected fault, next to the same scenarios without one, and what the agent
    did after each fault."""
    from arlab.faults import fault_behavior, markdown, summarize_faults
    from arlab.store import load_events, load_row

    with connect() as conn:
        keys = conn.execute(
            "SELECT s.run_id, s.scenario_id, s.fault FROM scenario_results s"
            " JOIN runs r ON r.id = s.run_id WHERE r.status IN ('finished', 'interrupted')"
            "   AND (r.batch_id = %s OR (r.batch_id = %s AND s.fault = 'none'))"
            " ORDER BY r.started_at, s.scenario_id",
            (batch, clean_batch),
        ).fetchall()
        if not keys:
            raise typer.BadParameter(f"no finished runs in batch {batch}")
        results = []
        for run_id, scenario_id, fault in keys:
            row = load_row(conn, run_id, scenario_id, fault)
            behavior = ""
            if fault != "none":
                events = load_events(conn, run_id, scenario_id, fault)
                behavior = fault_behavior(fault, row, events)
            results.append((row, behavior))
    typer.echo(markdown(summarize_faults(results)))


@app.command()
def judge(
    batch: Annotated[str, typer.Option(help="batch whose finished runs to judge")],
    suite: Annotated[Path, typer.Option(help="suite directory the batch ran")],
    limit: Annotated[int, typer.Option(help="most new judge calls to pay for")] = 0,
) -> None:
    """LLM-judge diagnosis, decoy blame and grounding for each result; stored verdicts are reused."""
    import anthropic

    from arlab.judge import JUDGE_HASH, call_judge, judge_prompt, scenario_reference
    from arlab.runner import load_scenario
    from arlab.store import load_events

    paths = {p.stem: p for p in sorted(suite.glob("*.yaml"))}
    client = anthropic.Anthropic()
    spent, new = 0.0, 0
    with connect() as conn:
        todo = conn.execute(
            "SELECT s.run_id, s.scenario_id, s.fault FROM scenario_results s"
            " JOIN runs r ON r.id = s.run_id"
            " LEFT JOIN judgments j ON j.run_id = s.run_id AND j.scenario_id = s.scenario_id"
            "   AND j.fault = s.fault AND j.judge_hash = %s"
            " WHERE r.batch_id = %s AND r.status IN ('finished', 'interrupted') AND j.run_id IS NULL"
            " ORDER BY r.started_at, s.scenario_id",
            (JUDGE_HASH, batch),
        ).fetchall()
        typer.echo(f"judge {JUDGE_HASH}: {len(todo)} results without a verdict")
        for run_id, scenario_id, fault in todo:
            if limit and new >= limit:
                break
            scenario = load_scenario(paths[scenario_id])
            prompt = judge_prompt(
                scenario.cluster["tickets"][scenario.ticket_id],
                scenario_reference(paths[scenario_id]),
                load_events(conn, run_id, scenario_id, fault),
            )
            verdict, raw, cost = call_judge(client, prompt)
            conn.execute(
                "INSERT INTO judgments (run_id, scenario_id, fault, judge_hash, diagnosis_correct,"
                " blames_decoy, grounded, reason, raw, cost_usd)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    run_id,
                    scenario_id,
                    fault,
                    JUDGE_HASH,
                    verdict.diagnosis_correct,
                    verdict.blames_decoy,
                    verdict.grounded,
                    verdict.reason,
                    json.dumps(raw),
                    round(cost, 6),
                ),
            )
            spent += cost
            new += 1
            typer.echo(
                f"{scenario_id:40} diagnosis={verdict.diagnosis_correct} decoy={verdict.blames_decoy}"
                f" grounded={verdict.grounded} ${cost:.4f}"
            )
    typer.echo(f"{new} new judgments, ${spent:.4f}")


@app.command("judge-report")
def judge_report(
    batch: Annotated[str, typer.Option(help="batch whose stored verdicts to summarize")],
) -> None:
    """Judge verdicts next to the deterministic outcome, for the current judge prompt and model."""
    from arlab.judge import JUDGE_HASH, JudgedRow, Verdict, judgment_markdown

    with connect() as conn:
        rows = conn.execute(
            "SELECT s.scenario_id, s.success, s.failure_class, j.reason, j.diagnosis_correct,"
            " j.blames_decoy, j.grounded FROM scenario_results s"
            " JOIN runs r ON r.id = s.run_id"
            " JOIN judgments j ON j.run_id = s.run_id AND j.scenario_id = s.scenario_id"
            "   AND j.fault = s.fault AND j.judge_hash = %s"
            " WHERE r.batch_id = %s AND r.status IN ('finished', 'interrupted')",
            (JUDGE_HASH, batch),
        ).fetchall()
    if not rows:
        raise typer.BadParameter(f"no verdicts from judge {JUDGE_HASH} in batch {batch}")
    judged = [JudgedRow(sid, ok, cls, Verdict(*v)) for sid, ok, cls, *v in rows]
    typer.echo(f"judge {JUDGE_HASH}, {len(judged)} results")
    typer.echo("Not calibrated against human labels until `arlab calibrate` has run.\n")
    typer.echo(judgment_markdown(judged))


@app.command()
def frontier(
    batch: Annotated[list[str], typer.Option(help="LABEL=BATCH, one per config (repeatable)")],
    suite: Annotated[Path, typer.Option(help="suite directory, for scenario categories")],
    out: Annotated[Path, typer.Option(help="SVG file to write")],
) -> None:
    """Plot pass rate against cost per successful task and p50 latency, one dot per config."""
    from arlab.frontier import FrontierPoint, frontier_svg
    from arlab.report import load_batch, summarize
    from arlab.runner import load_suite

    categories = {s.id: s.category for s in load_suite(suite)}
    points = []
    with connect() as conn:
        for spec in batch:
            label, _, batch_id = spec.partition("=")
            summary = summarize(load_batch(conn, batch_id)[0], categories)
            if summary.cost_per_success is None:
                raise typer.BadParameter(f"{batch_id} has no successful task to price")
            rates = summary.repeat_pass_rates
            points.append(
                FrontierPoint(
                    label,
                    sum(rates) / len(rates),
                    summary.cost_per_success,
                    summary.p50_latency_ms / 1000,
                )
            )
            typer.echo(f"{label}: {points[-1]}")
    out.write_text(frontier_svg(points))
    typer.echo(f"wrote {out}")


@app.command("judge-sample")
def judge_sample(
    batch: Annotated[list[str], typer.Option(help="batches to sample judged results from")],
    suite: Annotated[Path, typer.Option(help="suite directory the batches ran")],
    out: Annotated[Path, typer.Option(help="YAML file for a person to label")],
    n: int = 70,
    seed: int = 0,
) -> None:
    """Sample judged results for a person to label, without the judge's verdicts."""
    import random
    from collections import defaultdict

    from arlab.judge import JUDGE_HASH, judge_prompt, scenario_reference
    from arlab.labeling import dump_labels
    from arlab.runner import load_scenario
    from arlab.store import load_events

    if out.exists():
        raise typer.BadParameter(f"{out} exists; it may hold human labels, so it is not replaced")
    paths = {p.stem: p for p in sorted(suite.glob("*.yaml"))}
    with connect() as conn:
        judged = conn.execute(
            "SELECT j.run_id, j.scenario_id, j.fault, s.success FROM judgments j"
            " JOIN scenario_results s USING (run_id, scenario_id, fault)"
            " JOIN runs r ON r.id = j.run_id"
            " WHERE r.batch_id = ANY(%s) AND j.judge_hash = %s"
            " ORDER BY j.run_id, j.scenario_id, j.fault",
            (batch, JUDGE_HASH),
        ).fetchall()
        # Round-robin over category x check result, so rare categories and failures both show up.
        strata: dict[tuple[str, bool], list[tuple[str, str, str]]] = defaultdict(list)
        for run_id, scenario_id, fault, success in judged:
            category = load_scenario(paths[scenario_id]).category
            strata[(category, success)].append((run_id, scenario_id, fault))
        rng = random.Random(seed)
        for bucket in strata.values():
            rng.shuffle(bucket)
        picked: list[tuple[str, str, str]] = []
        while len(picked) < n and any(strata.values()):
            for key in sorted(strata):
                if strata[key] and len(picked) < n:
                    picked.append(strata[key].pop())
        items = []
        for run_id, scenario_id, fault in picked:
            scenario = load_scenario(paths[scenario_id])
            prompt = judge_prompt(
                scenario.cluster["tickets"][scenario.ticket_id],
                scenario_reference(paths[scenario_id]),
                load_events(conn, run_id, scenario_id, fault),
            )
            items.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "fault": fault,
                    "shown_to_judge": prompt,
                    "diagnosis_correct": None,
                    "blames_decoy": None,
                    "grounded": None,
                    "notes": "",
                }
            )
    header = (
        f"# {len(items)} results for a person to label, judge {JUDGE_HASH}. Fill each null with true or\n"
        "# false, following the questions in src/arlab/judge.py (JUDGE_SYSTEM); leave blames_decoy\n"
        "# null when there is no decoy and grounded null when nothing was cited. Use notes for\n"
        "# ambiguous cases. Judge verdicts are not shown, so they cannot anchor the labels.\n"
    )
    out.write_text(dump_labels(items, header))
    typer.echo(f"wrote {len(items)} items to {out}")


@app.command()
def calibrate(
    labels: Annotated[Path, typer.Option(help="YAML from judge-sample, labeled by a person")],
) -> None:
    """Judge agreement with human labels: agreement, Cohen's kappa, false positives and negatives."""
    import yaml

    from arlab.calibration import QUESTIONS, agreement, markdown
    from arlab.judge import JUDGE_HASH

    items = yaml.safe_load(labels.read_text())
    unlabeled = [i["scenario_id"] for i in items if i["diagnosis_correct"] is None]
    if unlabeled:
        raise typer.BadParameter(f"{len(unlabeled)} items have no human diagnosis label yet")
    results, disagreements = [], []
    with connect() as conn:
        for question in QUESTIONS:
            pairs = []
            for item in items:
                found = conn.execute(
                    f"SELECT {question} FROM judgments WHERE run_id = %s AND scenario_id = %s"
                    " AND fault = %s AND judge_hash = %s",
                    (item["run_id"], item["scenario_id"], item["fault"], JUDGE_HASH),
                ).fetchone()
                if found is None:
                    raise typer.BadParameter(
                        f"no judgment for {item['scenario_id']} by {JUDGE_HASH}"
                    )
                human, judged = item[question], found[0]
                if human is None and judged is None:
                    continue
                if human is None or judged is None:
                    # One rater says the question applies and the other doesn't (a decoy the judge
                    # saw, a label left empty): a disagreement, but not one kappa can count.
                    disagreements.append(
                        f"{question}: {item['scenario_id']} ({item['run_id'][:8]}) applies only"
                        f" for the {'judge' if human is None else 'human'}"
                    )
                    continue
                pairs.append((human, judged))
                if human != judged:
                    disagreements.append(
                        f"{question}: {item['scenario_id']} ({item['run_id'][:8]})"
                    )
            if pairs:
                results.append(agreement(question, pairs))
    # judge-sample stratifies by category and check result, so these describe that sample, not the
    # population of results.
    typer.echo(f"{len(items)} items, stratified by category and check result, judge {JUDGE_HASH}")
    typer.echo(markdown(results))
    if disagreements:
        typer.echo("\ndisagreements:\n" + "\n".join(f"- {d}" for d in disagreements))
    noted = [f"{i['scenario_id']}: {i['notes']}" for i in items if i.get("notes")]
    if noted:
        typer.echo("\nmarked by the labeler:\n" + "\n".join(f"- {n}" for n in noted))


@app.command()
def label(
    labels: Annotated[Path, typer.Option(help="YAML from judge-sample")],
    question: Annotated[
        list[str] | None, typer.Option(help="ask only these questions, e.g. diagnosis_correct")
    ] = None,
) -> None:
    """Label judge-sample items one key at a time: y/n per question, s to skip, q to save and quit.
    Saves after every item, so it resumes where it stopped."""
    import click
    import yaml

    from arlab.labeling import PROMPTS, applicable, dump_labels, is_labeled, read_header

    text = labels.read_text()
    header, items = read_header(text), yaml.safe_load(text)
    if question and set(question) - set(PROMPTS):
        raise typer.BadParameter(f"questions are {', '.join(PROMPTS)}")

    def asked(item: dict[str, Any]) -> list[str]:
        return [q for q in applicable(item) if not question or q in question]

    todo = [i for i in items if any(i[q] is None for q in asked(i))]
    for n, item in enumerate(todo, 1):
        click.clear()
        typer.secho(
            f"item {n} of {len(todo)} left ({len(items)} total): {item['scenario_id']}"
            f" fault={item['fault']}",
            bold=True,
        )
        typer.echo(item["shown_to_judge"])
        for q in asked(item):
            typer.secho(f"\n{q}: {PROMPTS[q]} [y/n/s/q] ", bold=True, nl=False)
            key = ""
            while key not in ("y", "n", "s", "q"):
                key = typer.getchar().lower()
            typer.echo(key)
            if key == "q":
                typer.echo(f"saved; {sum(map(is_labeled, items))} of {len(items)} labeled")
                return
            if key == "s":
                break
            item[q] = key == "y"
        note = typer.prompt("note (Enter for none)", default="", show_default=False)
        if note:
            item["notes"] = note
        # Write to a temporary file first, so an interrupted save never truncates the labels.
        tmp = labels.with_suffix(".tmp")
        tmp.write_text(dump_labels(items, header))
        tmp.replace(labels)
    typer.echo(f"done; {sum(map(is_labeled, items))} of {len(items)} labeled")


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
