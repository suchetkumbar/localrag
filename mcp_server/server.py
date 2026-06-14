"""
LocalRAG MCP Server.
Exposes RAG capabilities as MCP tools, resources, and prompts.
Transport: stdio (for local use) or can be bridged via FastAPI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.config import get_settings, ensure_directories
from backend.services.logging_service import setup_logging
from backend.services.vector_store_service import VectorStoreService
from backend.services.ingestion_service import IngestionService
from backend.services.chat_service import ChatService
from backend.services.session_service import SessionService
from database.models import init_engine, create_tables, get_session

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

settings = get_settings()
ensure_directories(settings)
setup_logging(settings.logging.level, settings.log_file_path)

_vector_store = VectorStoreService(settings)
_vector_store.initialize()

_db_engine = init_engine(str(settings.sqlite_path))

_ingestion_svc = IngestionService(settings, _vector_store)
_chat_svc = ChatService(settings, _vector_store)
_session_svc = SessionService()

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

app = Server(settings.mcp.server_name)


# ===========================
# TOOLS
# ===========================

@app.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="search_documents",
                description=(
                    "Search the document knowledge base using semantic similarity. "
                    "Returns the most relevant text chunks with source citations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (1-20).",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Minimum similarity score (0.0-1.0).",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="ingest_document",
                description="Ingest a document file into the knowledge base.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file to ingest.",
                        },
                        "source": {
                            "type": "string",
                            "description": "Source label (e.g. 'upload', 'watcher', 'manual').",
                            "default": "manual",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="list_documents",
                description="List all ingested documents with their metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "Filter by status: 'ready', 'processing', 'error', or 'all'.",
                            "default": "all",
                        }
                    },
                },
            ),
            Tool(
                name="get_document",
                description="Get detailed metadata for a specific document by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document UUID.",
                        }
                    },
                    "required": ["document_id"],
                },
            ),
            Tool(
                name="delete_document",
                description="Delete a document and all its chunks from the knowledge base.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document UUID to delete.",
                        }
                    },
                    "required": ["document_id"],
                },
            ),
            Tool(
                name="ask_question",
                description=(
                    "Ask a question against the knowledge base using RAG. "
                    "Returns an answer with source citations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Chat session ID for conversation history.",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to answer.",
                            "minLength": 1,
                            "maxLength": 4000,
                        },
                    },
                    "required": ["session_id", "question"],
                },
            ),
            Tool(
                name="get_collection_stats",
                description="Get statistics about the vector store collection.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
    )


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    log = logger.bind(tool=name)
    log.info("mcp_tool_called", args=arguments)

    try:
        if name == "search_documents":
            return await _tool_search_documents(arguments)
        elif name == "ingest_document":
            return await _tool_ingest_document(arguments)
        elif name == "list_documents":
            return await _tool_list_documents(arguments)
        elif name == "get_document":
            return await _tool_get_document(arguments)
        elif name == "delete_document":
            return await _tool_delete_document(arguments)
        elif name == "ask_question":
            return await _tool_ask_question(arguments)
        elif name == "get_collection_stats":
            return await _tool_get_collection_stats(arguments)
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )
    except Exception as exc:
        log.error("mcp_tool_error", error=str(exc), exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool error: {exc}")],
            isError=True,
        )


# ===========================
# TOOL IMPLEMENTATIONS
# ===========================

async def _tool_search_documents(args: dict) -> CallToolResult:
    query = args["query"]
    top_k = args.get("top_k", settings.retrieval.top_k)
    threshold = args.get("score_threshold", settings.retrieval.score_threshold)

    results = await _vector_store.similarity_search(query, top_k=top_k, score_threshold=threshold)

    if not results:
        return CallToolResult(
            content=[TextContent(type="text", text="No relevant documents found for this query.")]
        )

    output = []
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        output.append(
            f"[{i}] Source: {m.get('filename', 'unknown')} | "
            f"Chunk: {m.get('chunk_index', 0)} | "
            f"Page: {m.get('page', 0)} | "
            f"Score: {score:.4f}\n"
            f"{doc.page_content}"
        )

    return CallToolResult(
        content=[TextContent(type="text", text="\n\n---\n\n".join(output))]
    )


async def _tool_ingest_document(args: dict) -> CallToolResult:
    file_path = Path(args["file_path"])
    source = args.get("source", "manual")

    if not file_path.exists():
        return CallToolResult(
            content=[TextContent(type="text", text=f"File not found: {file_path}")],
            isError=True,
        )

    if not _ingestion_svc.is_supported(file_path):
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unsupported file type: {file_path.suffix}")],
            isError=True,
        )

    async with get_session() as db:
        doc = await _ingestion_svc.ingest_file(file_path, db, source=source)

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "chunk_count": doc.chunk_count,
                        "status": doc.status,
                    },
                    indent=2,
                ),
            )
        ]
    )


async def _tool_list_documents(args: dict) -> CallToolResult:
    from sqlalchemy import select
    from database.models import Document as DocumentModel

    status_filter = args.get("status", "all")

    async with get_session() as db:
        q = select(DocumentModel).order_by(DocumentModel.created_at.desc())
        if status_filter != "all":
            q = q.where(DocumentModel.status == status_filter)
        result = await db.execute(q)
        docs = result.scalars().all()

    output = [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "chunk_count": d.chunk_count,
            "status": d.status,
            "source": d.source,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(output, indent=2))]
    )


async def _tool_get_document(args: dict) -> CallToolResult:
    from sqlalchemy import select
    from database.models import Document as DocumentModel

    doc_id = args["document_id"]

    async with get_session() as db:
        result = await db.execute(select(DocumentModel).where(DocumentModel.id == doc_id))
        doc = result.scalar_one_or_none()

    if doc is None:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Document not found: {doc_id}")],
            isError=True,
        )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "file_type": doc.file_type,
                        "file_size": doc.file_size,
                        "chunk_count": doc.chunk_count,
                        "status": doc.status,
                        "source": doc.source,
                        "error_message": doc.error_message,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                    },
                    indent=2,
                ),
            )
        ]
    )


async def _tool_delete_document(args: dict) -> CallToolResult:
    from sqlalchemy import select
    from database.models import Document as DocumentModel

    doc_id = args["document_id"]

    async with get_session() as db:
        result = await db.execute(select(DocumentModel).where(DocumentModel.id == doc_id))
        doc = result.scalar_one_or_none()

        if doc is None:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Document not found: {doc_id}")],
                isError=True,
            )

        await _ingestion_svc.delete_document(doc, db)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Document {doc_id} deleted successfully.")]
    )


async def _tool_ask_question(args: dict) -> CallToolResult:
    session_id = args["session_id"]
    question = args["question"]

    # Ensure session exists
    async with get_session() as db:
        session = await _session_svc.get_session(db, session_id)
        if session is None:
            session = await _session_svc.create_session(db, title=question[:60])

    async with get_session() as db:
        answer, sources = await _chat_svc.chat(session_id, question, db)

    output = {
        "answer": answer,
        "sources": [s.to_dict() for s in sources],
        "session_id": session_id,
    }

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(output, indent=2))]
    )


async def _tool_get_collection_stats(args: dict) -> CallToolResult:
    stats = await _vector_store.get_collection_stats()
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(stats, indent=2))]
    )


# ===========================
# RESOURCES
# ===========================

@app.list_resources()
async def list_resources() -> ListResourcesResult:
    return ListResourcesResult(
        resources=[
            Resource(
                uri="documents://list",
                name="Document List",
                description="List of all ingested documents",
                mimeType="application/json",
            ),
            Resource(
                uri="collections://stats",
                name="Collection Statistics",
                description="Vector store collection statistics",
                mimeType="application/json",
            ),
        ]
    )


@app.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    if uri == "documents://list":
        result = await _tool_list_documents({})
        return ReadResourceResult(contents=[result.content[0]])

    if uri == "collections://stats":
        result = await _tool_get_collection_stats({})
        return ReadResourceResult(contents=[result.content[0]])

    if uri.startswith("documents://"):
        doc_id = uri.split("documents://")[1]
        result = await _tool_get_document({"document_id": doc_id})
        return ReadResourceResult(contents=[result.content[0]])

    raise ValueError(f"Unknown resource URI: {uri}")


# ===========================
# PROMPTS
# ===========================

@app.list_prompts()
async def list_prompts() -> ListPromptsResult:
    return ListPromptsResult(
        prompts=[
            Prompt(
                name="rag_answer",
                description="Generate a RAG-based answer with citations for a given question.",
                arguments=[
                    PromptArgument(name="question", description="The question to answer.", required=True),
                    PromptArgument(name="session_id", description="Chat session ID.", required=True),
                ],
            ),
            Prompt(
                name="summarize_document",
                description="Summarize the content of a specific document.",
                arguments=[
                    PromptArgument(name="document_id", description="The document UUID to summarize.", required=True),
                    PromptArgument(name="session_id", description="Chat session ID.", required=True),
                ],
            ),
        ]
    )


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    arguments = arguments or {}

    if name == "rag_answer":
        question = arguments.get("question", "")
        session_id = arguments.get("session_id", "")
        return GetPromptResult(
            description="Answer a question using the RAG knowledge base.",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please answer the following question using information from "
                            f"the knowledge base. Always cite your sources.\n\n"
                            f"Session: {session_id}\n"
                            f"Question: {question}"
                        ),
                    ),
                )
            ],
        )

    if name == "summarize_document":
        doc_id = arguments.get("document_id", "")
        session_id = arguments.get("session_id", "")
        return GetPromptResult(
            description="Summarize a document from the knowledge base.",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please provide a comprehensive summary of document ID: {doc_id}. "
                            f"Include key topics, main points, and important details.\n\n"
                            f"Session: {session_id}"
                        ),
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    await create_tables()
    logger.info("mcp_server_starting", name=settings.mcp.server_name)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
