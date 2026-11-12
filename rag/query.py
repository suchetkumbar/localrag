"""Ask one question with a given config. A manual smoke test, not the eval runner.

python -m rag.query --config configs/baseline.yaml "How do I declare a path parameter?"
python -m rag.query --corpus path/to/markdown/folder "Any question about those files?"

Reads .env first, so GROQ_API_KEY etc. can live there (see .env.example).
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.config import load_config
from rag.factory import build_pipeline
from rag.loaders import load_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--corpus", type=Path, help="override corpus.path from the config")
    parser.add_argument("--show", type=int, default=5, help="number of retrieved chunks to print")
    args = parser.parse_args(argv)
    load_dotenv()

    cfg = load_config(args.config)
    corpus_path = args.corpus or cfg.corpus.path
    docs = load_corpus(corpus_path, cfg.corpus.patterns)
    pipeline = build_pipeline(cfg)
    chunks = pipeline.index(docs)
    print(f"config {cfg.name} ({cfg.fingerprint()}): {len(docs)} docs, {len(chunks)} chunks")

    answer = pipeline.answer(args.question)
    print("\nRetrieved:")
    for rank, sc in enumerate(answer.ranking[: args.show], start=1):
        print(f"  {rank:>2}. {sc.score:8.4f}  {sc.chunk.chunk_id}")
    print(f"\nAnswer ({answer.generation.model}, cached={answer.generation.cached}):")
    print(answer.generation.text)
    timings = ", ".join(f"{k}={v:.0f}" for k, v in answer.timings_ms.items())
    print(f"\nTimings (ms): {timings}; model latency {answer.generation.latency_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
