# LocalRAG design notes

This file records *why* the evaluation harness is built the way it is: the trade-offs
considered, the alternatives rejected, and known limitations. It grows with each
milestone. Numbers quoted here must come from result files in the repo; none exist yet.

## Scope and layout

LocalRAG started as a local document-chat app (FastAPI, ChromaDB, LangChain, Next.js,
an MCP server, Prometheus/Grafana). An evaluation harness needs something different: a
**stateless, deterministic, swappable** pipeline. Milestone 1 built that as a new package,
`rag/`, and the chat app was then removed so the repository contains only what the
harness uses. The original app is preserved in git history (commit `d679fe2` on `main`).

```
rag/
  types.py       Document, Chunk, ScoredChunk, Generation (frozen dataclasses)
  interfaces.py  Protocols: Chunker, Embedder, Retriever, Reranker, Generator
  config.py      YAML schema (pydantic, extra="forbid", discriminated unions)
  factory.py     config -> components (the only place that maps `kind` to a class)
  cache.py       SQLite-backed cache for every model call
  loaders.py     corpus directory -> Documents
  chunkers.py    fixed-size, heading-based
  embedders.py   Ollama, fake, cached wrapper
  retrievers.py  BM25, dense (Qdrant), hybrid (RRF)
  rerankers.py   cross-encoder (fastembed/ONNX), cached wrapper
  generators.py  Ollama, OpenAI-compatible, fake, cached wrapper
  prompts/       versioned prompt templates + context formatting
  pipeline.py    RAGPipeline: index / retrieve / answer
  http.py        HTTP error helper that keeps server error messages
  query.py       `python -m rag.query` manual smoke test
configs/         one YAML file per experiment
tests/           unit tests, one file per component
```

## Problems found in the original pipeline

These are the reasons the harness could not reuse the original app's `ChatService`:

| Problem | Why it breaks evaluation | Fix in `rag/` |
|---|---|---|
| Chunk IDs were random UUIDs (`uuid4` doc IDs, Chroma-generated chunk IDs) | Labels and per-item comparisons across runs are impossible | Deterministic IDs `doc_id#start-end` |
| A 0.3 similarity threshold dropped results | "recall@5" could be computed over 2 results; k silently varies | Retrievers always return the top k; thresholds are not part of retrieval |
| `max_context_tokens` actually counted characters, and the loop used `break` | One long chunk dropped all lower-ranked chunks from the prompt | `max_context_chars` (honest name), skip-and-continue |
| Chat history and DB writes inside answering | Answers depend on previous questions | Stateless `RAGPipeline.answer()` |
| `temperature=0.1`, no seed | Non-deterministic outputs | `temperature=0`, fixed `seed` |
| `rerank: false` in settings.yaml was never read | Dead config hides missing features | Reranker is a real, configurable stage |
| nomic-embed-text used without its task prefixes | The model was trained with `search_query:` / `search_document:` prefixes | Prefixes are explicit embedder config |

## Decisions

### Interfaces are `typing.Protocol`, not abstract base classes
Structural typing means a test fake or a third-party object satisfies an interface just
by having the right methods. No inheritance, no registration. mypy still checks
conformance wherever a component is passed in.

### Configuration: pydantic with `extra="forbid"` and discriminated unions
Every experiment is a YAML file. Each component block has a `kind` that selects the
implementation, and each implementation declares exactly the parameters it accepts.
Unknown keys are errors: a typo like `chunk_szie: 500` must fail loudly, not run a
"500-character" experiment with the default size. `PipelineConfig.fingerprint()` hashes
every result-affecting setting (excluding the name and the cache settings), so two runs
can be checked for config identity.

*Rejected:* a plugin registry with decorators. With about ten implementations, an
explicit `match` statement in `factory.py` is easier to read and to grep.

### Labelling spans, not chunks
The golden dataset (Milestone 2) must work for *every* chunking strategy, but fixed and
heading chunkers produce different chunks, so "relevant_chunk_ids" from one strategy are
meaningless for another. Instead, every `Chunk` carries exact character offsets
(`text == doc.text[start:end]`, enforced by tests) and golden items will label the
supporting **span** in the source document. A chunk is relevant if it overlaps a labelled
span, so the same labels score any chunker. Line endings are normalised to `\n` on load,
so offsets are identical on Windows and Linux CI.

### Chunking
* **Sizes are in characters, not tokens.** Characters are deterministic and need no
  tokenizer dependency. English prose is roughly 4 characters per token, so the default
  1000-character window is about 250 tokens. *Trade-off:* windows hold a varying number of
  tokens, which matters little at these sizes.
* **Fixed-size windows snap to whitespace** so words are not cut in half, but only within
  the part of the window past the overlap, which guarantees progress for any overlap value.
* **The heading chunker** splits at markdown headings (up to `max_heading_level`), ignores
  `#` lines inside fenced code blocks (they are Python comments, not headings), and gives
  each chunk its heading path as `title`, e.g. `Security > OAuth2 > Scopes`. The title is
  prepended for indexing and for the LLM (`Chunk.indexed_text`) but is not part of the
  span, so the offset invariant holds. Oversized sections fall back to fixed windows.

### Retrieval
* **Dense: Qdrant in embedded mode.** `QdrantClient(":memory:")` or `QdrantClient(path=...)`
  runs in-process, so tests and CI need no server. The same code works against a Qdrant
  server via a URL. *Rejected:* pgvector, which needs a running Postgres everywhere,
  including CI. *Rejected:* keeping ChromaDB, whose `hnswlib` native extension was
  intermittently blocked by Windows Application Control on the development machine.
* **BM25 is implemented by hand** (about 60 lines in `retrievers.py`) so every score can be
  derived on a whiteboard; the unit test checks scores against hand-computed values. It
  uses the Lucene idf variant `ln(1 + (N - df + 0.5)/(df + 0.5))`, which stays positive.
  The classic Robertson idf goes negative for terms in more than half the documents and
  would penalise matching them. Duplicate query terms count once. The tokenizer is
  lowercase `\w+`, which keeps `response_model` as one token (important for API docs). No
  stemming or stopwords: idf already down-weights common words, and stemming would merge
  distinct identifiers.
* **Hybrid uses reciprocal rank fusion**, `score(d) = Σ 1/(k + rank)`, with k = 60. RRF only
  looks at ranks, so BM25 scores (unbounded) and cosine similarities ([-1, 1]) need no
  normalisation. *Rejected:* a weighted sum of scores, which needs per-query score
  normalisation and a tuned weight. Both are extra knobs that overfit a small golden set.
* **Deterministic tie-breaking**: every ranking sorts by `(-score, chunk_id)`.
* **Two depths:** `candidate_k` (default 50) is the first-stage depth and the depth of
  the ranking that retrieval metrics score; `top_k` (default 5) is how many chunks the
  LLM sees. They are separate so that recall@10 can be measured while the LLM gets 5.

### Reranking: cross-encoder via fastembed (ONNX)
A cross-encoder reads (query, chunk) jointly, which is more accurate than comparing
independent embeddings but costs one model pass per candidate, so it only reorders the
`candidate_k` first-stage results. `BAAI/bge-reranker-base` runs through fastembed on
onnxruntime. *Rejected:* sentence-transformers, which pulls in PyTorch (about 2 GB). The
model is downloaded and loaded lazily on first use.

### Generation: plain HTTP, local by default, free hosted option
* **Ollama is the default** (local, free, unlimited). Requests set `temperature=0`, a fixed
  `seed` and an explicit `num_ctx`. Ollama silently truncates prompts longer than
  `num_ctx`, which would drop retrieved context with no error. The default is 4096: the
  worst-case prompt is about 2,700 tokens (8000 context chars ≈ 2000 tokens, plus the system
  prompt and 512 answer tokens). 8192 ran out of VRAM on the development GPU (6 GB RTX
  3060 with llama3-8B and the embedder loaded).
* **One `OpenAICompatibleGenerator`** covers free hosted tiers (Groq by default; OpenRouter,
  Gemini and NVIDIA NIM work by changing `base_url`). The API key is read from the env
  var named in config, never from config itself. Free tiers have daily request caps, so
  the full experiment matrix runs locally; the hosted option is for spot comparisons.
* **No vendor SDKs or LangChain.** One `httpx.post` per call is easy to read, debug and
  test with `httpx.MockTransport`. HTTP errors keep the server's message body (Ollama
  reports out-of-memory only there).

### Prompts are versioned files
`rag/prompts/answer_v1.system.txt` and `answer_v1.user.txt`. A prompt change is its own
diff, and a new version is a new file, so old results stay reproducible. The prompt asks
for numbered citations `[n]` (mapped back to `Answer.context[n-1]`) and an **exact refusal
sentence** for unanswerable questions, so refusal can be checked exactly rather than
judged. v1 deliberately has no prompt-injection hardening; Milestone 6 measures the
injection rate and any hardening will be a v2 compared against v1.

### Caching every model call
Embeddings, reranker scores and generations go through a SQLite cache keyed by sha256
of canonical JSON of `(kind, model identity + params, exact input)`.
* Re-runs are nearly free and **reproducible**: the same config and dataset produce
  identical outputs.
* Embeddings are cached **per text**, so re-chunking only embeds chunks whose text changed.
* The rendered prompt is in the generation key, so editing a template automatically
  invalidates exactly the affected entries.
* Three modes: `rw`, `off`, and `readonly`. In readonly a miss raises an error, so CI can
  never silently call a model.
* A cache hit returns the **original** latency and token counts (flagged `cached=True`), so
  operational metrics from a cached re-run still describe the real model call.
* Values are JSON or `.npy` bytes (`allow_pickle=False`); nothing is ever unpickled.

## Known limitations (Milestone 1)

* Retrieval-stage latency on a warm cache measures cache lookups, not model calls. The
  Milestone 3 latency report needs a cold-cache mode for honest p50/p95 numbers.
* The fake embedder is a hashing bag-of-words. It exercises the wiring, not semantics.
* `DenseRetriever` re-embeds the whole corpus on `index()` unless the cache is warm; there
  is no persistent index reuse between processes yet (not needed at ~300 documents).
* The heading chunker understands ATX (`#`) headings only, not setext (`===` underlines).
* Character-based sizing means token counts per chunk vary with content (code is denser).
* Memory: on the development machine (16 GB RAM, 6 GB VRAM) the ONNX reranker (~1 GB in
  the Python process) plus llama3-8B in Ollama can exhaust host RAM. The Milestone 4
  runner should therefore run in phases (retrieve and rerank every item, release the
  reranker, then generate) instead of item by item.
