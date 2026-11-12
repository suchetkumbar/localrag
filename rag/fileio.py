"""Crash-safe file writes."""

import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path


def write_lines_atomic(path: Path, lines: Iterable[str]) -> None:
    """Write ``lines`` (each followed by a newline) so the file is never half-written.

    The data goes to a temporary file in the same directory, which then replaces the
    target in one step. A crash mid-write (or mid-review session) leaves the old file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")
        replace_with_retry(Path(tmp), path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def replace_with_retry(src: Path, dst: Path, attempts: int = 20) -> None:
    """``os.replace`` that tolerates brief locks on Windows.

    On Windows, replacing a file fails with PermissionError while another process (an
    antivirus scanner, a search indexer, an editor) holds it open even momentarily.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
