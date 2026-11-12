"""Console helpers shared by the CLIs."""

import sys


def use_utf8_output() -> None:
    """Print UTF-8 regardless of the console's code page.

    The FastAPI docs contain emoji; the default Windows console encoding (cp1252) cannot
    encode them and would crash the review CLI mid-session.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
