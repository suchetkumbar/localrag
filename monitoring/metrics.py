"""
Prometheus metrics definitions.
All application metrics are registered here.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry

# Use default registry
REGISTRY = CollectorRegistry(auto_describe=True)

# ---------------------------------------------------------------------------
# Application info
# ---------------------------------------------------------------------------

APP_INFO = Info(
    "localrag_app",
    "LocalRAG application metadata",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Document metrics
# ---------------------------------------------------------------------------

DOCUMENTS_INGESTED_TOTAL = Counter(
    "localrag_documents_ingested_total",
    "Total number of documents ingested",
    ["source", "file_type"],
    registry=REGISTRY,
)

DOCUMENTS_INGESTION_ERRORS_TOTAL = Counter(
    "localrag_documents_ingestion_errors_total",
    "Total number of document ingestion errors",
    ["file_type", "error_type"],
    registry=REGISTRY,
)

DOCUMENTS_TOTAL = Gauge(
    "localrag_documents_total",
    "Current total number of documents in the store",
    registry=REGISTRY,
)

CHUNKS_TOTAL = Gauge(
    "localrag_chunks_total",
    "Current total number of chunks in the vector store",
    registry=REGISTRY,
)

INGESTION_DURATION_SECONDS = Histogram(
    "localrag_ingestion_duration_seconds",
    "Time taken to ingest a document",
    ["file_type"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Query metrics
# ---------------------------------------------------------------------------

QUERIES_TOTAL = Counter(
    "localrag_queries_total",
    "Total number of queries processed",
    ["session_id"],
    registry=REGISTRY,
)

QUERY_ERRORS_TOTAL = Counter(
    "localrag_query_errors_total",
    "Total number of query errors",
    ["error_type"],
    registry=REGISTRY,
)

QUERY_DURATION_SECONDS = Histogram(
    "localrag_query_duration_seconds",
    "End-to-end query duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0],
    registry=REGISTRY,
)

RETRIEVAL_CHUNKS_RETURNED = Histogram(
    "localrag_retrieval_chunks_returned",
    "Number of chunks returned per retrieval",
    buckets=[1, 2, 3, 5, 8, 10],
    registry=REGISTRY,
)

LLM_DURATION_SECONDS = Histogram(
    "localrag_llm_duration_seconds",
    "Time taken by the LLM to generate a response",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Session metrics
# ---------------------------------------------------------------------------

SESSIONS_ACTIVE = Gauge(
    "localrag_sessions_active",
    "Number of active chat sessions",
    registry=REGISTRY,
)

SESSIONS_CREATED_TOTAL = Counter(
    "localrag_sessions_created_total",
    "Total sessions created",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# HTTP metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "localrag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "localrag_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)
