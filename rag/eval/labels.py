"""Human labels for validating the LLM judge, and the judge-human agreement report.

Labeling is blind: the labeler sees the question, the reference, the system's answer and
the passages it was given, but never the judge's verdict, so the judge cannot anchor the
human. The sample is drawn with a fixed seed from items the judge actually graded.
"""

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rag.config import StrictModel
from rag.eval.agreement import Agreement, agreement
from rag.eval.evaluate import ItemResult
from rag.fileio import write_lines_atomic

KAPPA_THRESHOLD = 0.6
CORRECTNESS_KEYS: dict[str, Literal["correct", "partial", "incorrect"]] = {
    "c": "correct",
    "p": "partial",
    "i": "incorrect",
}


class HumanLabel(StrictModel):
    id: str
    correctness: Literal["correct", "partial", "incorrect"] | None
    faithful: bool | None  # None: the answer makes no factual claims
    labeler: str
    labeled_at: datetime
    note: str = ""


def read_results(path: Path) -> list[ItemResult]:
    with path.open(encoding="utf-8") as f:
        return [ItemResult.model_validate_json(line) for line in f if line.strip()]


def write_results(path: Path, results: Sequence[ItemResult]) -> None:
    write_lines_atomic(path, (r.model_dump_json() for r in results))


def read_labels(path: Path) -> dict[str, HumanLabel]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        labels = [HumanLabel.model_validate_json(line) for line in f if line.strip()]
    return {label.id: label for label in labels}


def write_labels(path: Path, labels: Mapping[str, HumanLabel]) -> None:
    ordered = sorted(labels.values(), key=lambda label: label.id)
    write_lines_atomic(path, (label.model_dump_json() for label in ordered))


def sample_for_labeling(results: Sequence[ItemResult], n: int, seed: int) -> list[ItemResult]:
    """Answerable, non-refused items the judge graded, sampled reproducibly."""
    eligible = sorted(
        (
            r
            for r in results
            if r.answerable and not r.refused and r.judge and r.judge.correctness_label
        ),
        key=lambda r: r.id,
    )
    rng = random.Random(seed)
    return rng.sample(eligible, min(n, len(eligible)))


@dataclass
class LabelSession:
    items: Sequence[ItemResult]
    labels: dict[str, HumanLabel]
    save: Callable[[dict[str, HumanLabel]], None]
    ask: Callable[[str], str] = input
    say: Callable[[str], None] = print
    labeler: str = "human"
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    def run(self) -> int:
        todo = [r for r in self.items if r.id not in self.labels]
        done = 0
        for n, r in enumerate(todo, start=1):
            self._show(r, n, len(todo))
            label = self._ask_label(r)
            if label is None:
                break
            self.labels[r.id] = label
            self.save(self.labels)
            done += 1
        self.say(f"\nlabeled {done} items this session; {len(self.labels)} in total")
        return done

    def _show(self, r: ItemResult, n: int, total: int) -> None:
        self.say("\n" + "=" * 78)
        self.say(f"[{n}/{total}] {r.id} ({r.question_type})")
        self.say(f"Q: {r.question}")
        self.say(f"REFERENCE: {r.reference_answer}")
        self.say(f"SYSTEM ANSWER: {r.answer}")
        for i, text in enumerate(r.context_texts, start=1):
            self.say(f"\n  [{i}] {' '.join(text.split())[:700]}")

    def _ask_label(self, r: ItemResult) -> HumanLabel | None:
        while True:
            c = self.ask("correctness vs reference: [c]orrect [p]artial [i]ncorrect, [q]uit > ")
            c = c.strip().lower()
            if c == "q":
                return None
            if c in CORRECTNESS_KEYS:
                break
        while True:
            f = self.ask("every claim supported by the passages? [y]es [n]o [-] no claims > ")
            f = f.strip().lower()
            if f in ("y", "n", "-"):
                break
        note = self.ask("note (optional): ").strip()
        return HumanLabel(
            id=r.id,
            correctness=CORRECTNESS_KEYS[c],
            faithful=None if f == "-" else f == "y",
            labeler=self.labeler,
            labeled_at=self.now(),
            note=note,
        )


@dataclass(frozen=True, slots=True)
class AgreementReport:
    correctness: Agreement | None
    faithfulness: Agreement | None

    @property
    def trusted(self) -> bool:
        parts = [a for a in (self.correctness, self.faithfulness) if a is not None]
        return bool(parts) and all(a.kappa >= KAPPA_THRESHOLD for a in parts)

    def to_dict(self) -> dict[str, Any]:
        def one(a: Agreement | None) -> dict[str, Any] | None:
            if a is None:
                return None
            confusion = [
                {"judge": str(j), "human": str(h), "count": c}
                for (j, h), c in sorted(a.confusion.items(), key=str)
            ]
            return {
                "n": a.n,
                "percent_agreement": a.percent,
                "cohen_kappa": a.kappa,
                "confusion": confusion,
            }

        return {
            "kappa_threshold": KAPPA_THRESHOLD,
            "trusted": self.trusted,
            "correctness": one(self.correctness),
            "faithfulness": one(self.faithfulness),
        }


def agreement_report(
    results: Sequence[ItemResult], labels: Mapping[str, HumanLabel]
) -> AgreementReport:
    judge_c: list[str] = []
    human_c: list[str] = []
    judge_f: list[bool] = []
    human_f: list[bool] = []
    for r in results:
        label = labels.get(r.id)
        if label is None or r.judge is None:
            continue
        if label.correctness is not None and r.judge.correctness_label is not None:
            judge_c.append(r.judge.correctness_label)
            human_c.append(label.correctness)
        if label.faithful is not None and r.judge.faithful is not None:
            judge_f.append(r.judge.faithful)
            human_f.append(label.faithful)
    return AgreementReport(
        correctness=agreement(judge_c, human_c) if judge_c else None,
        faithfulness=agreement(judge_f, human_f) if judge_f else None,
    )


def format_report(report: AgreementReport) -> str:
    lines = []
    for name, a in (("correctness", report.correctness), ("faithfulness", report.faithfulness)):
        if a is None:
            lines.append(f"{name}: no overlapping labels")
            continue
        status = "ok" if a.kappa >= KAPPA_THRESHOLD else f"BELOW {KAPPA_THRESHOLD}"
        lines.append(f"{name}: n={a.n}  agreement={a.percent:.1%}  kappa={a.kappa:.3f}  [{status}]")
        for (j, h), count in sorted(a.confusion.items(), key=str):
            marker = "" if j == h else "   <- disagreement"
            lines.append(f"    judge={j!s:<10} human={h!s:<10} {count}{marker}")
    verdict = "TRUSTED" if report.trusted else "NOT TRUSTED: improve the rubric (see DESIGN.md)"
    lines.append(f"judge: {verdict}")
    return "\n".join(lines)


def dump_report(path: Path, report: AgreementReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n")
