# LocalRAG: a RAG evaluation and regression harness

Score every change to a chunker, embedding model, retriever, reranker, prompt or LLM
against a golden dataset, and block pull requests that make answers worse.

> **Status:** work in progress. Milestone 1 (pluggable pipeline) is done. The golden
> dataset, metrics, experiment runner and CI gate are next. No results are reported yet;
> every number that appears here later will link to a results file in this repo.

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
python -m rag.query --corpus path/to/markdown "Your question"
```

`configs/baseline.yaml` documents every option. Copy it to try a variation, for example
`retriever: {kind: hybrid}` or `reranker: {kind: cross_encoder}`.

## Layout

```
rag/          the pipeline package (see DESIGN.md for every design decision)
configs/      one YAML file per experiment
tests/        unit tests, one file per component
DESIGN.md     trade-offs, rejected alternatives, known limitations
```

## Development

```bash
ruff check . && ruff format --check . && mypy && pytest
```

CI runs the same four commands on every pull request.
