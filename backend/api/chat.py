"""
Chat API endpoints.
POST /api/sessions/{session_id}/chat       - single-turn Q&A
GET  /api/sessions/{session_id}/chat/stream - SSE streaming
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.api.deps import DBSession, ChatDep, SessionSvcDep
from backend.models.schemas import ChatRequest, ChatResponse, ChatSource

router = APIRouter(prefix="/api/sessions", tags=["chat"])


@router.post("/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str,
    request: ChatRequest,
    db: DBSession,
    chat_svc: ChatDep,
    session_svc: SessionSvcDep,
) -> ChatResponse:
    """Send a message and get a RAG-based answer with citations."""
    session = await session_svc.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    answer, sources = await chat_svc.chat(session_id, request.message, db)

    return ChatResponse(
        answer=answer,
        sources=[ChatSource(**s.to_dict()) for s in sources],
        session_id=session_id,
    )


@router.get("/{session_id}/chat/stream")
async def stream_chat(
    session_id: str,
    message: str,
    db: DBSession,
    chat_svc: ChatDep,
    session_svc: SessionSvcDep,
) -> StreamingResponse:
    """Stream a response token by token using SSE."""
    session = await session_svc.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    async def event_generator():
        async for token in chat_svc.stream_chat(session_id, message, db):
            if token.startswith("\n\n__SOURCES__:"):
                sources_json = token.split("__SOURCES__:")[1]
                yield f"event: sources\ndata: {sources_json}\n\n"
            else:
                safe = json.dumps(token)
                yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
