"""Pluggable RAG pipeline used by the evaluation harness.

The package is deliberately independent of the FastAPI app in ``backend/`` so it can be
imported, tested and benchmarked without a web server, a database or LangChain.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
