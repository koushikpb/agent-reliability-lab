-- Loaded by the Postgres container on first start, and by arlab.db.apply_schema.
-- IF NOT EXISTS adds missing tables but never alters existing ones, so a column change needs the table
-- dropped and recreated.
CREATE EXTENSION IF NOT EXISTS vector;

-- id is stable across re-ingests (doc path + heading slug + part) so relevance labels keep pointing
-- at the same chunk. A serial id would silently break every label on re-ingest.
CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,
    doc_path  text NOT NULL,
    heading   text NOT NULL,
    text      text NOT NULL,
    -- Generated so it can never drift from text, unlike a trigger or app-side update.
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || text)) STORED,
    -- 384 dims matches all-MiniLM-L6-v2. Nullable so chunks can be inserted before embedding.
    embedding vector(384)
);

-- The HNSW index on embedding is built by ingest after the bulk load; building it on an empty table
-- makes every insert update the graph.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS queries (
    id   text PRIMARY KEY,
    text text NOT NULL
);

-- Graded so nDCG can use graded gain; Recall and MRR treat any grade >= 1 as relevant.
-- No foreign key to chunks: ingest replaces chunks wholesale, and a cascade would delete labels.
-- retrieval-eval checks instead that every labeled chunk_id exists.
CREATE TABLE IF NOT EXISTS query_labels (
    query_id text NOT NULL REFERENCES queries (id),
    chunk_id text NOT NULL,
    grade    smallint NOT NULL CHECK (grade IN (1, 2)),
    PRIMARY KEY (query_id, chunk_id)
);

-- params is the fully resolved config (prompt text, agent settings, tool schemas, embedder and
-- reranker revisions), and hash = sha256 of its canonical JSON, so the hash can be recomputed.
CREATE TABLE IF NOT EXISTS configs (
    hash           text PRIMARY KEY,
    model          text NOT NULL,
    prompt_version text NOT NULL,
    retriever      text NOT NULL,
    params         jsonb NOT NULL
);

-- Repeats of one config are separate runs that share a batch_id.
CREATE TABLE IF NOT EXISTS runs (
    id            text PRIMARY KEY,
    batch_id      text NOT NULL,
    suite         text NOT NULL,
    suite_hash    text NOT NULL,
    config_hash   text NOT NULL REFERENCES configs (hash),
    -- corpus commit + chunker params + embedder; retrieval results are only comparable within one.
    index_version text NOT NULL,
    git_sha       text NOT NULL,
    mode          text NOT NULL CHECK (mode IN ('live', 'replay')),
    -- Without status, a crashed run and a run in progress both look like finished_at IS NULL.
    status        text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'finished', 'failed')),
    started_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);

CREATE TABLE IF NOT EXISTS scenario_results (
    run_id          text NOT NULL REFERENCES runs (id),
    scenario_id     text NOT NULL,
    fault           text NOT NULL DEFAULT 'none',
    success         boolean NOT NULL,
    final_status    text,
    -- tool_errors are errors the tool server returned (injected or not); invalid_calls are the
    -- agent's own malformed or wrong calls. Mixing them hides whether the agent or the fault failed.
    tool_errors     integer NOT NULL DEFAULT 0,
    invalid_calls   integer NOT NULL DEFAULT 0,
    redundant_calls integer NOT NULL DEFAULT 0,
    cited_ids       text[] NOT NULL DEFAULT '{}',
    latency_ms      integer,
    input_tokens    integer,
    output_tokens   integer,
    cost_usd        numeric(10, 6),
    failure_class   text,
    PRIMARY KEY (run_id, scenario_id, fault)
);

-- seq orders events within a scenario, so a run can be replayed without trusting wall-clock times.
-- span ids mirror the OpenTelemetry tree, e.g. a retrieval inside a tool call. kind has no CHECK
-- because judge calls and faults add kinds later, and IF NOT EXISTS would never widen it.
CREATE TABLE IF NOT EXISTS events (
    run_id         text NOT NULL REFERENCES runs (id),
    scenario_id    text NOT NULL,
    fault          text NOT NULL DEFAULT 'none',
    seq            integer NOT NULL,
    span_id        text NOT NULL,
    parent_span_id text,
    kind           text NOT NULL,
    payload        jsonb NOT NULL,
    started_at     timestamptz NOT NULL,
    duration_ms    integer,
    PRIMARY KEY (run_id, scenario_id, fault, seq)
);
