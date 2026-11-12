"""Golden dataset CLI.

python -m rag.golden draft                 # LLM drafts -> data/golden/drafts.jsonl
python -m rag.golden review [--type numeric]
python -m rag.golden stats
python -m rag.golden build                 # verified items -> data/golden/v1.jsonl
python -m rag.golden validate data/golden/v1.jsonl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rag.cache import DiskCache
from rag.config import CorpusConfig
from rag.console import use_utf8_output
from rag.corpus import read_fingerprint
from rag.factory import build_generator
from rag.golden.card import stats_markdown, update_card
from rag.golden.draft import DraftConfig, draft_items
from rag.golden.review import ReviewSession, apply_quotas, stratified_sample
from rag.golden.schema import (
    QUESTION_TYPES,
    REVIEW_STATUSES,
    GoldenItem,
    QuestionType,
    load_golden,
    read_jsonl,
    validate_against_corpus,
    write_jsonl,
)
from rag.loaders import load_corpus
from rag.types import Document

DRAFTS = Path("data/golden/drafts.jsonl")
DATASET = Path("data/golden/v1.jsonl")
CARD = Path("data/golden/README.md")
# Final composition of v1: 200 items, 10% unanswerable.
QUOTAS: dict[QuestionType, int] = {
    "factual": 110,
    "numeric": 30,
    "multi_hop": 40,
    "unanswerable": 20,
}


def _load_draft_config(path: Path) -> DraftConfig:
    with path.open(encoding="utf-8") as f:
        return DraftConfig.model_validate(yaml.safe_load(f) or {})


def _corpus(cfg: CorpusConfig) -> tuple[dict[str, Document], str]:
    fingerprint = read_fingerprint(cfg.path)
    if fingerprint is None:
        raise SystemExit(
            f"{cfg.path} is not prepared; run `python -m rag.corpus fetch` and `prepare`"
        )
    return {d.doc_id: d for d in load_corpus(cfg.path, cfg.patterns)}, fingerprint


def cmd_draft(args: argparse.Namespace) -> int:
    cfg = _load_draft_config(args.config)
    docs, fingerprint = _corpus(cfg.corpus)
    existing = read_jsonl(args.drafts) if args.drafts.exists() else []
    generator = build_generator(cfg.generator, DiskCache(cfg.cache.path, cfg.cache.mode))
    items: list[GoldenItem] = list(existing)

    def save(item: GoldenItem) -> None:
        items.append(item)
        write_jsonl(args.drafts, items)
        print(f"  [{len(items):>3}] {item.id} {item.question}", flush=True)

    print(f"drafting from {len(docs)} docs; {len(existing)} existing items kept")
    _, stats = draft_items(
        cfg,
        list(docs.values()),
        generator,
        corpus_fingerprint=fingerprint,
        existing=existing,
        on_item=save,
    )
    print(f"attempted: {dict(stats.attempted)}")
    print(f"accepted:  {dict(stats.accepted)}")
    print(f"rejected:  {dict(stats.rejected)}")
    print(f"{args.drafts}: {len(items)} items (all pending review)")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    docs, _ = _corpus(CorpusConfig(path=args.corpus))
    items = read_jsonl(args.drafts)
    session = ReviewSession(
        items=items,
        docs=docs,
        save=lambda current: write_jsonl(args.drafts, current),
        reviewer=args.reviewer,
        quotas=QUOTAS,
    )
    session.run(only_type=args.type)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    items = read_jsonl(args.drafts)
    table = Counter((i.question_type, i.review.status) for i in items)
    statuses = REVIEW_STATUSES
    print(f"{'type':<14}" + "".join(f"{s:>10}" for s in statuses) + f"{'quota':>8}")
    for q in QUESTION_TYPES:
        row = "".join(f"{table[(q, s)]:>10}" for s in statuses)
        print(f"{q:<14}{row}{QUOTAS[q]:>8}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    docs, fingerprint = _corpus(CorpusConfig(path=args.corpus))
    drafts = read_jsonl(args.drafts)
    selected, missing = apply_quotas(drafts, QUOTAS)
    if missing and not args.partial:
        print(f"not enough verified items yet, missing: {missing}. Review more, or pass --partial.")
        return 1
    problems = validate_against_corpus(selected, docs, fingerprint)
    if problems:
        print("\n".join(problems))
        return 1
    write_jsonl(args.out, selected)
    update_card(args.card, stats_markdown(selected, drafts))
    print(f"{args.out}: {len(selected)} verified items; data card updated: {args.card}")
    if missing:
        print(f"PARTIAL build, still missing: {missing}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    subset = stratified_sample(load_golden(args.dataset), args.n, args.seed)
    write_jsonl(args.out, subset)
    counts = Counter(i.question_type for i in subset)
    print(f"{args.out}: {len(subset)} items {dict(sorted(counts.items()))} (seed {args.seed})")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    docs, fingerprint = _corpus(CorpusConfig(path=args.corpus))
    items = load_golden(args.dataset, allow_unverified=args.allow_unverified)
    problems = validate_against_corpus(items, docs, fingerprint)
    for p in problems:
        print(p)
    print(f"{args.dataset}: {len(items)} items, {len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python -m rag.golden",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    corpus_default = CorpusConfig().path

    d = sub.add_parser("draft", help="LLM-draft candidate items (all start unverified)")
    d.add_argument("--config", type=Path, default=Path("configs/drafting.yaml"))
    d.set_defaults(func=cmd_draft)

    r = sub.add_parser("review", help="accept / edit / reject pending drafts")
    r.add_argument("--type", choices=QUESTION_TYPES)
    r.add_argument("--reviewer", default="human", help="name recorded with each decision")
    r.set_defaults(func=cmd_review)

    s = sub.add_parser("stats", help="review progress per type")
    s.set_defaults(func=cmd_stats)

    b = sub.add_parser("build", help="write the verified dataset and update the data card")
    b.add_argument("--out", type=Path, default=DATASET)
    b.add_argument("--card", type=Path, default=CARD)
    b.add_argument("--partial", action="store_true", help="build even if quotas are not met")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="check a dataset against the prepared corpus")
    v.add_argument("dataset", type=Path)
    v.add_argument("--allow-unverified", action="store_true")
    v.set_defaults(func=cmd_validate)

    sm = sub.add_parser("smoke", help="stratified subset of the verified dataset, for CI")
    sm.add_argument("--dataset", type=Path, default=DATASET)
    sm.add_argument("--out", type=Path, default=Path("data/golden/smoke.jsonl"))
    sm.add_argument("--n", type=int, default=40)
    sm.add_argument("--seed", type=int, default=11)
    sm.set_defaults(func=cmd_smoke)

    for p in (d, r, s, b, v):
        p.add_argument("--drafts", type=Path, default=DRAFTS)
        p.add_argument("--corpus", type=Path, default=corpus_default)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
