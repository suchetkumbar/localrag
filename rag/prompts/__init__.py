"""Versioned prompt templates and context formatting.

Prompts live in text files (``<name>.system.txt`` / ``<name>.user.txt``) rather than
Python strings so a prompt change is a reviewable diff of its own, and a new version is
a new file: old results stay reproducible. The rendered prompt is part of the generation
cache key, so editing a template automatically invalidates the affected cache entries.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag.types import ScoredChunk

PROMPTS_DIR = Path(__file__).parent

# Must match the sentence in answer_v1.system.txt. A fixed refusal string makes refusal
# detection on unanswerable questions an exact check rather than a judgement call.
REFUSAL = "I don't know based on the provided documents."


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    system: str
    user: str

    @property
    def version_hash(self) -> str:
        return hashlib.sha256(f"{self.system}\x00{self.user}".encode()).hexdigest()[:12]

    def render_user(self, question: str, context: str) -> str:
        return self.user.format(question=question, context=context)


def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> PromptTemplate:
    system = (directory / f"{name}.system.txt").read_text(encoding="utf-8").strip()
    user = (directory / f"{name}.user.txt").read_text(encoding="utf-8").strip()
    if "{context}" not in user or "{question}" not in user:
        raise ValueError(f"prompt {name!r}: user template needs {{context}} and {{question}}")
    return PromptTemplate(name=name, system=system, user=user)


def format_context(chunks: list[ScoredChunk], max_chars: int) -> tuple[str, list[ScoredChunk]]:
    """Number the chunks [1], [2], ... within a character budget.

    Returns the context string and the chunks actually included, in order, so that
    citation [n] maps back to ``included[n - 1]``. A chunk that does not fit is skipped
    and later (smaller) ones may still be included. The old implementation stopped at
    the first oversized chunk, silently dropping everything after it.
    """
    parts: list[str] = []
    included: list[ScoredChunk] = []
    used = 0
    for sc in chunks:
        n = len(included) + 1
        entry = f"[{n}] (source: {sc.chunk.doc_id})\n{sc.chunk.indexed_text}"
        cost = len(entry) + (2 if parts else 0)
        if used + cost > max_chars:
            continue
        parts.append(entry)
        included.append(sc)
        used += cost
    return "\n\n".join(parts), included
