"""Judge validation CLI.

python -m rag.eval label --results results/<run>/items.jsonl      # blind human labels
python -m rag.eval agreement --results results/<run>/items.jsonl  # kappa vs the judge
"""

import argparse
import sys
from pathlib import Path

from rag.console import use_utf8_output
from rag.eval.labels import (
    LabelSession,
    agreement_report,
    dump_report,
    format_report,
    read_labels,
    read_results,
    sample_for_labeling,
    write_labels,
)

LABELS = Path("data/judge/human_labels.jsonl")


def cmd_label(args: argparse.Namespace) -> int:
    results = read_results(args.results)
    sample = sample_for_labeling(results, args.n, args.seed)
    labels = read_labels(args.labels)
    print(f"{len(sample)} items sampled (seed {args.seed}); {len(labels)} already labeled")
    session = LabelSession(
        items=sample,
        labels=labels,
        save=lambda current: write_labels(args.labels, current),
        labeler=args.labeler,
    )
    session.run()
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    report = agreement_report(read_results(args.results), read_labels(args.labels))
    print(format_report(report))
    if args.out:
        dump_report(args.out, report)
        print(f"written: {args.out}")
    return 0 if report.trusted else 1


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label", help="hand-label a seeded sample of answers, blind to the judge")
    lab.add_argument("--n", type=int, default=50)
    lab.add_argument("--seed", type=int, default=7)
    lab.add_argument("--labeler", default="human")
    lab.set_defaults(func=cmd_label)
    agr = sub.add_parser("agreement", help="percent agreement and Cohen's kappa vs the judge")
    agr.add_argument("--out", type=Path, help="also write the report as JSON")
    agr.set_defaults(func=cmd_agreement)
    for p in (lab, agr):
        p.add_argument("--results", type=Path, required=True, help="items.jsonl from a run")
        p.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
