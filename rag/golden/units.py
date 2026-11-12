"""Split a span of markdown into small, numbered "units" (sentences, list items, code blocks).

The drafting LLM cites evidence by unit number ("u3") instead of quoting text. Small
models often paraphrase "verbatim" quotes; unit numbers map back to exact offsets that we
computed ourselves, so every evidence span is guaranteed to exist in the corpus.
"""

import re
from dataclasses import dataclass

# MULTILINE so that "^" also matches at a line start when matching from an offset.
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z`*\[(\"])")
# A period after these does not end a sentence ("wrap the WSGI (e.g. Flask) app").
_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx)\.$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Unit:
    start: int
    end: int


def _trimmed(text: str, start: int, end: int) -> Unit | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Unit(start, end) if start < end else None


def fence_at(text: str, offset: int) -> str | None:
    """The fence marker (e.g. "```") open at ``offset``, or None if not inside a code block."""
    marker: str | None = None
    pos = 0
    while pos < offset:
        nl = text.find("\n", pos)
        line_end = len(text) if nl == -1 else nl
        m = _FENCE.match(text, pos, line_end)
        if m:
            if marker is None:
                marker = m.group(1)[0] * 3
            elif m.group(1).startswith(marker):
                marker = None
        pos = line_end + 1
    return marker


def split_units(text: str, start: int = 0, end: int | None = None) -> list[Unit]:
    """Units of ``text[start:end]``: each fenced code block is one unit, every other line
    is split into sentences. Blank lines are dropped. Offsets are absolute.

    The span may start inside a code block (long sections are windowed), so the fence
    state at ``start`` is computed from the beginning of the document.
    """
    end = len(text) if end is None else end
    units: list[Unit] = []
    pos = start
    open_marker = fence_at(text, start)
    fence_start: int | None = start if open_marker else None
    fence_marker = open_marker or ""
    while pos < end:
        nl = text.find("\n", pos, end)
        line_end = end if nl == -1 else nl
        line = text[pos:line_end]
        m = _FENCE.match(line)
        if fence_start is not None:
            if m and m.group(1).startswith(fence_marker):
                unit = _trimmed(text, fence_start, line_end)
                if unit:
                    units.append(unit)
                fence_start = None
        elif m:
            fence_start, fence_marker = pos, m.group(1)[0] * 3
        else:
            seg_start = pos
            for sm in _SENTENCE_END.finditer(text, pos, line_end):
                if _ABBREVIATION.search(text[seg_start : sm.start()]):
                    continue
                unit = _trimmed(text, seg_start, sm.start())
                if unit:
                    units.append(unit)
                seg_start = sm.end()
            unit = _trimmed(text, seg_start, line_end)
            if unit:
                units.append(unit)
        pos = line_end + 1
    if fence_start is not None:  # unterminated fence: keep it as one unit
        unit = _trimmed(text, fence_start, end)
        if unit:
            units.append(unit)
    return units


def merge_adjacent(units: list[Unit], text: str) -> list[Unit]:
    """Merge units separated only by whitespace, so consecutive cited sentences become one
    evidence span instead of several tiny ones."""
    merged: list[Unit] = []
    for u in sorted(units, key=lambda u: u.start):
        if merged and not text[merged[-1].end : u.start].strip():
            merged[-1] = Unit(merged[-1].start, max(merged[-1].end, u.end))
        else:
            merged.append(u)
    return merged
