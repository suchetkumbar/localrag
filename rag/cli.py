"""``rag-eval``: run experiments, compare runs, run the configuration matrix.

rag-eval run --config configs/baseline.yaml --dataset data/golden/v1.jsonl
rag-eval compare results/<runA> results/<runB>
rag-eval matrix --base configs/baseline.yaml --dataset data/golden/v1.jsonl
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rag.config import load_config
from rag.console import use_utf8_output
from rag.eval.evaluate import EvalConfig
from rag.experiment.matrix import matrix_configs, write_summary
from rag.experiment.report import compare, config_diff, render_compare
from rag.experiment.run import RunOptions, load_run, run_experiment

RESULTS = Path("results")


def _eval_config(path: Path | None) -> EvalConfig:
    if path is None:
        return EvalConfig()
    with path.open(encoding="utf-8") as f:
        return EvalConfig.model_validate(yaml.safe_load(f) or {})


def _options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        limit=args.limit,
        retrieval_only=args.retrieval_only,
        judge=not args.no_judge,
        allow_unverified=args.allow_unverified,
    )


def _run_dir(ref: str, root: Path) -> Path:
    path = Path(ref)
    if (path / "metrics.json").exists():
        return path
    if (root / ref / "metrics.json").exists():
        return root / ref
    raise SystemExit(f"no run found at {ref} or {root / ref}")


def cmd_run(args: argparse.Namespace) -> int:
    run_experiment(
        load_config(args.config),
        args.dataset,
        _eval_config(args.eval_config),
        _options(args),
        results_root=args.results,
    )
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    variants = matrix_configs(load_config(args.base))
    if args.only:
        variants = [v for v in variants if args.only in v.name]
    eval_cfg, options = _eval_config(args.eval_config), _options(args)
    run_dirs = []
    for i, cfg in enumerate(variants, start=1):
        print(f"\n=== [{i}/{len(variants)}] {cfg.name}")
        run_dirs.append(
            run_experiment(cfg, args.dataset, eval_cfg, options, results_root=args.results)
        )
    summary = args.results / ("matrix-retrieval.md" if args.retrieval_only else "matrix.md")
    published = write_summary(run_dirs, summary, None if args.only else Path("README.md"))
    print(f"\nsummary: {summary}" + ("" if published else " (README not updated: not publishable)"))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    dir_a, dir_b = _run_dir(args.run_a, args.results), _run_dir(args.run_b, args.results)
    (payload_a, items_a), (payload_b, items_b) = load_run(dir_a), load_run(dir_b)
    deltas, regressions = compare(items_a, items_b)
    cfg_a = yaml.safe_load((dir_a / "config.yaml").read_text(encoding="utf-8"))
    cfg_b = yaml.safe_load((dir_b / "config.yaml").read_text(encoding="utf-8"))
    text = render_compare(
        payload_a["run"]["run_id"],
        payload_b["run"]["run_id"],
        deltas,
        regressions,
        config_diff(cfg_a, cfg_b),
    )
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="rag-eval", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one config on a dataset")
    run.add_argument("--config", type=Path, required=True)
    matrix = sub.add_parser("matrix", help="chunker x retriever x reranker, 12 runs")
    matrix.add_argument("--base", type=Path, default=Path("configs/baseline.yaml"))
    matrix.add_argument("--only", help="run only variants whose name contains this text")
    for p in (run, matrix):
        p.add_argument("--dataset", type=Path, default=Path("data/golden/v1.jsonl"))
        p.add_argument("--eval-config", type=Path, help="judge and pricing (configs/eval.yaml)")
        p.add_argument("--limit", type=int, help="only the first N items (quick checks)")
        p.add_argument("--retrieval-only", action="store_true", help="skip generation and judge")
        p.add_argument("--no-judge", action="store_true", help="generate but do not judge")
        p.add_argument(
            "--allow-unverified", action="store_true", help="smoke tests on unreviewed items"
        )
    run.set_defaults(func=cmd_run)
    matrix.set_defaults(func=cmd_matrix)

    cmp = sub.add_parser("compare", help="diff two runs and list items that got worse")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.add_argument("--out", type=Path, help="also write the comparison as markdown")
    cmp.set_defaults(func=cmd_compare)

    for p in (run, matrix, cmp):
        p.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
