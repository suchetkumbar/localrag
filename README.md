# LocalRAG: a RAG evaluation and regression harness

Score every change to a chunker, embedding model, retriever, reranker, prompt or LLM
against a golden dataset, and block pull requests that make answers worse.

> **Status:** work in progress. Done: pluggable pipeline (M1); corpus pipeline and golden
> dataset tooling (M2, human review in progress). Next: metrics, experiment runner, CI gate.
> No results are reported yet; every number that appears here later will link to a
> results file in this repo.

## What exists today

A stateless RAG pipeline where each stage is swappable from YAML, with no code changes:

| Stage | Options |
|---|---|
| Chunking | fixed-size windows with overlap · markdown sections by heading |
| Retrieval | dense (Qdrant, embedded) · BM25 · hybrid (reciprocal rank fusion) |
| Reranking | off · cross-encoder `BAAI/bge-reranker-base` (ONNX) |
| Generation | Ollama (local, default) · any OpenAI-compatible free tier (e.g. Groq) |

Every model call (embeddings, reranker scores, generations) is cached on disk keyed by
its exact inputs, so re-running an experiment is fast, free and reproducible.

Everything is free: models run locally through [Ollama](https://ollama.com), and the
optional hosted generator uses a free API tier.

## Quick start

Requires Python 3.11+ and Ollama.

```bash
ollama pull llama3
ollama pull nomic-embed-text

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

pytest                                              # unit tests, no models needed

python -m rag.corpus fetch && python -m rag.corpus prepare   # FastAPI docs, checksummed
python -m rag.query "How do I declare a path parameter?"
```

The golden dataset workflow (drafting, human review, build) is described in
[`data/golden/README.md`](data/golden/README.md).

`configs/baseline.yaml` documents every option. Copy it to try a variation, for example
`retriever: {kind: hybrid}` or `reranker: {kind: cross_encoder}`.

## Layout

```
rag/          the pipeline package (see DESIGN.md for every design decision)
rag/golden/   golden dataset: schema, LLM drafting, review CLI, build
rag/corpus.py reproducible corpus download and preparation
configs/      one YAML file per experiment
data/         local only (git-ignored): corpus manifest, corpora, golden dataset
tests/        unit tests, one file per component
DESIGN.md     trade-offs, rejected alternatives, known limitations
```

## Development

```bash
ruff check . && ruff format --check . && mypy && pytest
```

CI runs the same four commands on every pull request.
