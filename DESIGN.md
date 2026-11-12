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

## Milestone 2: corpus and golden dataset

### A pinned, checksummed corpus
`rag/corpus.py` resolves a FastAPI release tag to a commit and records the raw URL, size
and sha256 of every file used in `data/manifests/fastapi.json` (kept locally, not in
git; regenerate it with `build-manifest --ref 0.162.5`). `fetch` re-downloads and verifies, and `prepare` builds the final text
and a fingerprint over every prepared document. Each golden item stores that fingerprint,
so a dataset can never be scored silently against a different corpus.

Preparation choices:
* **Code examples are inlined.** FastAPI pages pull code in with
  `{* ../../docs_src/x.py ln[a:b] *}`. Without inlining, the corpus would contain no code,
  which is unrealistic for API documentation. These paths are relative to the mkdocs
  project directory (`docs/en`), not to the page. Resolving them per page was the first
  bug, caught because the manifest builder checks that every include exists.
* **Excluded:** the changelog (about half of all English text), auto-generated
  `reference/` stubs, and `_*.md` tool fixtures. One of those (`_llm-test.md`) produced a
  nonsense draft in the pilot, which is how it was found.
* **Heading anchors stripped** (`## Title { #title }` becomes `## Title`), so they do not
  pollute BM25 tokens or heading paths.

### Drafting by unit number, not by quotation
The drafting model sees each passage as numbered units (sentences, list items, whole code
blocks) and cites evidence by unit number. Offsets then come from our own splitter, so
every evidence span exists verbatim. *Rejected:* asking the model to quote evidence and
fuzzy-matching the quote, because small models paraphrase "verbatim" quotes, and fuzzy
matching would turn that into wrong offsets instead of an error.

Two unit-splitter bugs found in the pilot run:
* A long section is split into windows, and a window can start inside a code block.
  The splitter therefore computes the code-fence state at the window start from the top
  of the document; otherwise code lines are treated as prose sentences.
* "e.g." is not a sentence boundary. Without that rule, an evidence span ended at
  "wrap the WSGI (e.g.".

### Drafts are not ground truth
Drafts are written with `human_verified=false` and only the review CLI can change that.
The schema enforces `human_verified == review in {accepted, edited}`, `build` only takes
verified items, and `load_golden()` raises on unverified items unless explicitly told it
is a smoke test. Rejected drafts stay in `drafts.jsonl` as an audit trail.

### Deterministic, resumable drafting
Section sampling is seeded, generation is temperature 0 and cached, and item IDs are
hashes of the source sections. Re-running `draft` after a crash continues where it
stopped and never duplicates or overwrites reviewed items. That mattered in practice:
the first full run died at item 51 when Windows refused to replace a file another process
(likely the antivirus scanner) briefly held open. The atomic write now retries briefly.

### Steering a small drafter
An 8B model asked for "an unanswerable question" wrote "What is the maximum number
of ..." almost every time. Each prompt now gets a deterministic opening ("Does FastAPI",
"How do I", "Which", ...) chosen from the item ID. Automatic filters discard questions that
mention the passage or unit IDs, numeric answers without a number, multi-hop drafts citing
one passage, and near-duplicates (token Jaccard >= 0.75).

### Relevance from span overlap
Retrieval metrics (Milestone 3) treat a chunk as relevant if it overlaps an evidence span.
This keeps one set of labels valid for every chunker. The exact overlap rule (any overlap
vs a minimum fraction) is defined and tested in Milestone 3.

## Milestone 3: metrics

Code: `rag/eval/`. Every metric that can be computed exactly is computed in our own code
without a model. The LLM judge is used only where meaning has to be assessed.

### Relevance from span overlap (`rag/eval/retrieval.py`)
A chunk covers an evidence span if it holds at least half of the span, or if at least half
of the chunk lies inside the span. The second condition handles evidence longer than a
chunk: a small chunk sitting inside a long evidence passage is relevant. The threshold
(`min_span_coverage`, 0.5) is a config value, and its boundary cases are tested.

### Retrieval metrics are span-level
* **recall@k** is the fraction of an item's evidence spans covered within the top k, not
  "was any relevant chunk retrieved". A multi-hop item that finds one of its two spans
  scores 0.5, which is exactly the failure multi-hop questions exist to expose.
* **MRR** is the reciprocal rank of the first chunk that covers any span.
* **nDCG@10 uses novelty gains.** A chunk earns gain 1 only if it covers a span not
  already covered higher up. Without this, overlapping windows of the same passage would
  each earn gain and inflate nDCG for the fixed-size chunker, whose 150-character overlap
  makes duplicate hits common. IDCG assumes every span is found in the first positions.
* Unanswerable items have no evidence and are excluded from retrieval metrics.

*Rejected:* computing these with RAGAS or DeepEval. Both are LLM-centric frameworks
with heavy dependency trees, and their retrieval metrics assume chunk-level labels. The
span-based definitions above are ~100 lines with hand-computed tests, and every number can
be explained without referring to a library.

### Deterministic answer checks (`rag/eval/answers.py`)
* **Refusal** is the exact refusal sentence from the prompt, or, for short answers only, a
  few common paraphrases ("the context does not mention ..."). The length limit stops a
  long, substantive answer that mentions a gap in passing from counting as a refusal.
  This gives *refusal accuracy* on unanswerable items and the *false refusal rate* on
  answerable ones.
* **Citation validity** is the fraction of cited `[n]` that refer to a passage actually
  shown. **Citation precision** is the fraction of valid citations whose passage covers
  gold evidence. Both are exact span checks, with no judge involved.

### LLM-as-judge with versioned rubrics (`rag/eval/judge.py`, `rag/eval/prompts/`)
* **Correctness** uses three labels: correct (every key fact, no contradiction), partial
  (some key facts, no contradiction), incorrect (wrong, contradicting, or "I don't know").
  They map to 1 / 0.5 / 0. Three labels instead of a 1-10 score: coarse labels are what a
  human can apply consistently, and Cohen's kappa needs categories.
* **Faithfulness** asks the judge to list the answer's claims and mark each as supported
  or not by the passages the generator saw. The score is the fraction supported; the
  binary label (all supported) is what is compared with the human.
* The rubric asks for a reason **before** the label, so the model compares the facts
  before committing to a verdict.
* **Refusals bypass the judge.** A refusal to an answerable question is "incorrect" by the
  rubric's own definition, and a refusal has no claims to check. That saves calls and
  removes a source of judge noise.
* Each rubric's version (`name@hash`) is stored with every verdict. Judge errors (invalid
  JSON, unknown label) are recorded per item and counted, never silently dropped.

*Rejected:* embedding similarity to the reference answer. "The default port is 8000" and
"... is 8080" are near-identical embeddings, but only one is correct. Similarity measures
wording, not facts.

### Is the judge trustworthy? (`rag/eval/agreement.py`, `rag/eval/labels.py`)
The judge is only trusted after it agrees with a human. `python -m rag.eval label` samples
50 graded answers with a fixed seed and shows them **blind**: the labeler never sees the
judge's verdict, so it cannot anchor them. `python -m rag.eval agreement` reports percent
agreement, Cohen's kappa and the confusion matrix for correctness and for faithfulness.
The judge counts as trusted when kappa >= 0.6 for both.

Kappa rather than accuracy: with mostly-correct answers, a judge that always says
"correct" reaches high percent agreement while knowing nothing. Kappa subtracts the
agreement expected by chance from the two raters' label frequencies, and that
always-"correct" judge scores kappa = 0 (this exact case is a unit test).

If kappa < 0.6, the plan is to read the disagreements in the confusion matrix, change
the rubric (a new `_v2` file, never an edit in place), re-judge from cache and recompute.
Re-labeling is not needed, because the human labels do not depend on the rubric.

**Known risk: self-preference.** By default the judge (llama3) is the same model as the
generator, and models tend to rate their own outputs leniently. The kappa check is the
safeguard; switching the judge to a different model (`judge:` in the eval config) is the
remedy if agreement is poor.

### Operational metrics (`rag/eval/ops.py`)
p50/p95/mean per stage (retrieve, rerank, generate, plus the model's own latency),
using the same linear-interpolation percentile as numpy. There are mean prompt and
completion tokens, and cost per query. Cost is 0 for local Ollama models, computed from a
configured price table for others, and reported as *unknown* rather than guessed if a
price is missing.

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
