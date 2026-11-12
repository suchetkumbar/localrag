"""LLM-assisted drafting of golden-item *candidates*.

Drafts are never ground truth: they are written with ``human_verified=False`` and status
``pending``, and only the review CLI can change that. The drafter's job is to propose
diverse, well-formed candidates with exact evidence spans so that review is fast.

Determinism: sections are sampled with a fixed seed, generation is temperature 0 and
cached, and item IDs are derived from the source sections. Re-running ``draft`` therefore
reproduces the same candidates and never duplicates or overwrites reviewed items.
"""

import hashlib
import random
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from rag.chunkers import HeadingChunker
from rag.config import (
    CacheConfig,
    CorpusConfig,
    GeneratorConfig,
    OllamaGeneratorConfig,
    SamplingParams,
    StrictModel,
)
from rag.golden.schema import QUESTION_TYPES, Evidence, GoldenItem, Provenance, QuestionType
from rag.golden.units import Unit, merge_adjacent, split_units
from rag.interfaces import Generator
from rag.llm_json import NoJsonObjectError, extract_json_object
from rag.prompts import REFUSAL
from rag.types import Chunk, Document

PROMPTS_DIR = Path(__file__).parent / "prompts"
# Opening words suggested to the drafter, chosen deterministically per item. Without them
# an 8B model asks almost every unanswerable question as "What is the maximum ...".
STYLES: dict[QuestionType, tuple[str, ...]] = {
    "factual": ("How do I", "What", "Why", "When should I", "Which", "Can I", "What happens if"),
    "numeric": ("What",),
    "multi_hop": ("How", "What", "Why", "Can I", "Which"),
    "unanswerable": ("Does FastAPI", "How do I", "Which", "Can I", "What happens if", "Is there"),
}
ID_PREFIX: dict[QuestionType, str] = {
    "factual": "fa",
    "numeric": "nu",
    "multi_hop": "mh",
    "unanswerable": "un",
}
# Questions that leak the drafting setup ("according to the passage", "in u3").
_BANNED = re.compile(r"\b(passages?|the text|this section|the document|[uab]\d+)\b", re.I)
_DIGIT = re.compile(r"\d")
# A number in prose that is not a list marker ("1. Install") or inside a code span.
_PROSE_NUMBER = re.compile(r"(?<![\w.`/-])\d+(?:\.\d+)*(?![\w`/])")
_WORD = re.compile(r"\w+")


DEFAULT_TARGETS: dict[QuestionType, int] = {
    "factual": 120,
    "numeric": 45,
    "multi_hop": 60,
    "unanswerable": 35,
}


class DraftConfig(StrictModel):
    corpus: CorpusConfig = CorpusConfig()
    generator: GeneratorConfig = OllamaGeneratorConfig(
        sampling=SamplingParams(json_mode=True, max_tokens=300)
    )
    # Oversampled versus the final 200-item build quotas: reviewers reject some drafts.
    targets: dict[QuestionType, int] = Field(default_factory=lambda: dict(DEFAULT_TARGETS))
    seed: int = 13
    max_items_per_doc: int = Field(default=4, gt=0)
    max_section_chars: int = Field(default=2500, gt=0)
    min_prose_chars: int = Field(default=300, ge=0)
    max_code_unit_chars: int = Field(default=800, gt=0)
    duplicate_jaccard: float = Field(default=0.75, gt=0, le=1)
    cache: CacheConfig = CacheConfig()


class DraftRejectedError(ValueError):
    """The LLM response is unusable; ``reason`` is a short category for statistics."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PromptSet:
    system: str
    templates: Mapping[QuestionType, str]

    @property
    def version_hash(self) -> str:
        h = hashlib.sha256(self.system.encode())
        for qtype in QUESTION_TYPES:
            h.update(b"\x00" + self.templates[qtype].encode())
        return h.hexdigest()[:12]


def load_prompts(directory: Path = PROMPTS_DIR) -> PromptSet:
    system = (directory / "system.txt").read_text(encoding="utf-8").strip()
    templates = {
        q: (directory / f"{q}.txt").read_text(encoding="utf-8").strip() for q in QUESTION_TYPES
    }
    return PromptSet(system=system, templates=templates)


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True, slots=True)
class Section:
    chunk: Chunk
    units: tuple[Unit, ...]
    prose_chars: int
    has_prose_number: bool


def _is_code(text: str, unit: Unit) -> bool:
    return text.startswith(("```", "~~~"), unit.start)


def candidate_sections(docs: Sequence[Document], cfg: DraftConfig) -> list[Section]:
    """Markdown sections with enough prose to ask about, in corpus order."""
    chunker = HeadingChunker(max_heading_level=3, max_size=cfg.max_section_chars, overlap=0)
    sections: list[Section] = []
    for doc in docs:
        for chunk in chunker.chunk(doc):
            units = split_units(doc.text, chunk.start, chunk.end)
            prose = [u for u in units if not _is_code(doc.text, u)]
            prose_chars = sum(u.end - u.start for u in prose)
            if prose_chars < cfg.min_prose_chars:
                continue
            has_number = any(
                _PROSE_NUMBER.search(doc.text[u.start : u.end]) and not _is_heading(doc.text, u)
                for u in prose
            )
            sections.append(Section(chunk, tuple(units), prose_chars, has_number))
    return sections


def _is_heading(text: str, unit: Unit) -> bool:
    return text.startswith("#", unit.start)


def render_units(
    text: str, units: Sequence[Unit], prefix: str, max_code_chars: int
) -> tuple[str, dict[str, Unit]]:
    lines, labels = [], {}
    for i, unit in enumerate(units, start=1):
        label = f"{prefix}{i}"
        body = text[unit.start : unit.end]
        if _is_code(text, unit) and len(body) > max_code_chars:
            body = body[:max_code_chars] + "\n... (code truncated)"
        lines.append(f"{label}: {body}")
        labels[label] = unit
    return "\n".join(lines), labels


# --------------------------------------------------------------------------- parsing


def extract_json(raw: str) -> dict[str, Any]:
    try:
        return extract_json_object(raw)
    except NoJsonObjectError as exc:
        reason = "no_json" if "no JSON object" in str(exc) else "bad_json"
        raise DraftRejectedError(reason, str(exc)) from exc


def check_question(question: str) -> str:
    question = " ".join(question.split())
    words = len(_WORD.findall(question))
    if not question.endswith("?"):
        raise DraftRejectedError("not_a_question")
    if not 4 <= words <= 40:
        raise DraftRejectedError("question_length", f"{words} words")
    if _BANNED.search(question):
        raise DraftRejectedError("refers_to_source")
    return question


def parse_draft(
    raw: str, qtype: QuestionType, labels: Mapping[str, Unit]
) -> tuple[str, str, list[str]]:
    """Return ``(question, answer, evidence_labels)`` or raise DraftRejectedError."""
    obj = extract_json(raw)
    question = check_question(str(obj.get("question", "")))
    if qtype == "unanswerable":
        return question, REFUSAL, []
    answer = " ".join(str(obj.get("answer", "")).split())
    if not answer:
        raise DraftRejectedError("no_answer")
    cited = obj.get("evidence_units")
    if not isinstance(cited, list) or not cited:
        raise DraftRejectedError("no_evidence")
    evidence = [str(c).strip().lower() for c in cited]
    unknown = [c for c in evidence if c not in labels]
    if unknown:
        raise DraftRejectedError("unknown_units", ",".join(unknown))
    if qtype == "numeric" and not _DIGIT.search(answer):
        raise DraftRejectedError("numeric_without_number")
    if qtype == "multi_hop" and not (
        any(c.startswith("a") for c in evidence) and any(c.startswith("b") for c in evidence)
    ):
        raise DraftRejectedError("single_hop")
    return question, answer, sorted(set(evidence), key=lambda c: (c[0], int(c[1:])))


def evidence_spans(doc: Document, units: Sequence[Unit]) -> list[Evidence]:
    return [
        Evidence(doc_id=doc.doc_id, start=u.start, end=u.end, quote=doc.text[u.start : u.end])
        for u in merge_adjacent(list(units), doc.text)
    ]


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def is_duplicate(question: str, others: Sequence[frozenset[str]], threshold: float) -> bool:
    tokens = _tokens(question)
    for other in others:
        union = tokens | other
        if union and len(tokens & other) / len(union) >= threshold:
            return True
    return False


# --------------------------------------------------------------------------- drafting


@dataclass
class DraftStats:
    attempted: Counter[str] = field(default_factory=Counter)
    accepted: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class _Task:
    qtype: QuestionType
    sections: tuple[Section, ...]


def _item_id(qtype: QuestionType, sections: Sequence[Section]) -> str:
    key = "|".join([qtype, *(s.chunk.chunk_id for s in sections)])
    return f"{ID_PREFIX[qtype]}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _tasks(qtype: QuestionType, sections: Sequence[Section], rng: random.Random) -> Iterator[_Task]:
    order = list(sections)
    rng.shuffle(order)
    if qtype == "numeric":
        order = [s for s in order if s.has_prose_number]
    if qtype != "multi_hop":
        yield from (_Task(qtype, (s,)) for s in order)
        return
    # Multi-hop: pair a section with a different section of the same document. Related
    # sections of one page are the most likely to support a genuine two-step question.
    by_doc: dict[str, list[Section]] = {}
    for s in sections:
        by_doc.setdefault(s.chunk.doc_id, []).append(s)
    for s in order:
        partners = [p for p in by_doc[s.chunk.doc_id] if p.chunk.title != s.chunk.title]
        if partners:
            yield _Task(qtype, (s, rng.choice(partners)))


def draft_items(
    cfg: DraftConfig,
    docs: Sequence[Document],
    generator: Generator,
    *,
    corpus_fingerprint: str,
    existing: Sequence[GoldenItem] = (),
    on_item: Callable[[GoldenItem], None] | None = None,
) -> tuple[list[GoldenItem], DraftStats]:
    """Draft candidates until each type reaches its target. Existing items are kept."""
    prompts = load_prompts()
    doc_by_id = {d.doc_id: d for d in docs}
    sections = candidate_sections(docs, cfg)
    identity = generator.identity
    drafted_by = f"{identity.get('provider', '?')}/{identity.get('model', '?')}"

    items = list(existing)
    known_ids = {i.id for i in items}
    used_chunks = {c for i in items for c in i.provenance.source_chunks}
    per_doc = Counter(c.split("#")[0] for c in used_chunks)
    per_type = Counter(i.question_type for i in items)
    question_tokens = [_tokens(i.question) for i in items]
    stats = DraftStats()

    for qtype in QUESTION_TYPES:
        rng = random.Random(f"{cfg.seed}:{qtype}")
        for task in _tasks(qtype, sections, rng):
            if per_type[qtype] >= cfg.targets.get(qtype, 0):
                break
            chunk_ids = [s.chunk.chunk_id for s in task.sections]
            item_id = _item_id(qtype, task.sections)
            if item_id in known_ids or any(c in used_chunks for c in chunk_ids):
                continue
            if any(per_doc[s.chunk.doc_id] >= cfg.max_items_per_doc for s in task.sections):
                continue
            stats.attempted[qtype] += 1
            try:
                item = _draft_one(cfg, task, item_id, doc_by_id, generator, prompts)
                if is_duplicate(item.question, question_tokens, cfg.duplicate_jaccard):
                    raise DraftRejectedError("duplicate")
            except DraftRejectedError as exc:
                stats.rejected[exc.reason] += 1
                continue
            item = item.model_copy(
                update={
                    "provenance": Provenance(
                        drafted_by=drafted_by,
                        draft_prompt=prompts.version_hash,
                        source_chunks=tuple(chunk_ids),
                    ),
                    "corpus_fingerprint": corpus_fingerprint,
                }
            )
            items.append(item)
            known_ids.add(item.id)
            used_chunks.update(chunk_ids)
            per_doc.update(s.chunk.doc_id for s in task.sections)
            per_type[qtype] += 1
            question_tokens.append(_tokens(item.question))
            stats.accepted[qtype] += 1
            if on_item is not None:
                on_item(item)
    return items, stats


def _draft_one(
    cfg: DraftConfig,
    task: _Task,
    item_id: str,
    doc_by_id: Mapping[str, Document],
    generator: Generator,
    prompts: PromptSet,
) -> GoldenItem:
    template = prompts.templates[task.qtype]
    labels: dict[str, Unit] = {}
    label_doc: dict[str, Document] = {}
    fields: dict[str, str] = {}
    for prefix, section in zip(
        ("a", "b") if task.qtype == "multi_hop" else ("u",), task.sections, strict=True
    ):
        doc = doc_by_id[section.chunk.doc_id]
        rendered, section_labels = render_units(
            doc.text, section.units, prefix, cfg.max_code_unit_chars
        )
        labels.update(section_labels)
        label_doc.update({label: doc for label in section_labels})
        suffix = f"_{prefix}" if task.qtype == "multi_hop" else ""
        fields[f"doc_id{suffix}"] = doc.doc_id
        fields[f"title{suffix}"] = section.chunk.title
        fields[f"units{suffix}"] = rendered
    styles = STYLES[task.qtype]
    fields["style"] = styles[int(item_id.split("-")[1], 16) % len(styles)]
    generation = generator.generate(prompts.system, template.format(**fields))
    question, answer, cited = parse_draft(generation.text, task.qtype, labels)

    evidence: list[Evidence] = []
    for doc_id in dict.fromkeys(label_doc[c].doc_id for c in cited):
        doc = doc_by_id[doc_id]
        evidence.extend(evidence_spans(doc, [labels[c] for c in cited if label_doc[c] is doc]))
    if task.qtype == "multi_hop" and len(evidence) < 2:
        raise DraftRejectedError("single_hop", "evidence collapsed into one span")
    return GoldenItem(
        id=item_id,
        question=question,
        reference_answer=answer,
        question_type=task.qtype,
        evidence=tuple(evidence),
        provenance=Provenance(drafted_by="pending", draft_prompt="pending"),
        corpus_fingerprint="pending",
    )
