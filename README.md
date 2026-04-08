# Agent Reliability Lab

Regression testing and fault injection for a tool-using, retrieval-augmented LLM agent. The agent resolves ops
tickets against a sandboxed Kubernetes-like environment using hybrid retrieval (Postgres full-text + pgvector,
fused with RRF, reranked with a cross-encoder) over the Kubernetes docs. Every run is traced and scored on
retrieval quality, task success, tool correctness, grounding, latency and cost.

> Work in progress. Tables get filled in from real runs.

## Retrieval

| Retriever | Recall@10 | MRR | nDCG@10 | p95 ms |
|---|---|---|---|---|
| Full-text | | | | |
| Dense | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + rerank | | | | |

## Agent configurations

| Config | Success | Tool errors | p95 latency | $/task |
|---|---|---|---|---|

## Running locally

```bash
docker compose up -d
uv sync
uv run arlab --help
```

Corpus: Kubernetes documentation, © The Kubernetes Authors, CC BY 4.0.
