"""Load a corpus directory into Documents with stable IDs and normalised text."""

import re
from collections.abc import Iterable
from pathlib import Path

from rag.types import Document

_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def normalise_text(text: str) -> str:
    """Normalise line endings so character offsets are identical on every OS.

    Without this, a file checked out with CRLF on Windows has different offsets than the
    same file on Linux CI, and every golden-dataset span would be wrong.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily: only needed for PDF corpora

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path, root: Path) -> Document:
    is_pdf = path.suffix.lower() == ".pdf"
    raw = _read_pdf(path) if is_pdf else path.read_text(encoding="utf-8")
    text = normalise_text(raw)
    doc_id = path.relative_to(root).as_posix()
    match = _H1.search(text)
    title = match.group(1) if match else path.stem
    return Document(doc_id=doc_id, text=text, title=title)


def load_corpus(root: Path | str, patterns: Iterable[str] = ("**/*.md",)) -> list[Document]:
    """Load every file matching ``patterns`` under ``root``, sorted by doc_id."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in root.glob(pattern) if p.is_file())
    docs = [load_document(p, root) for p in paths]
    docs = [d for d in docs if d.text.strip()]
    return sorted(docs, key=lambda d: d.doc_id)
