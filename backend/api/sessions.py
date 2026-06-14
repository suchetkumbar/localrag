"""
Session management API endpoints.
GET    /api/sessions             - list sessions
POST   /api/sessions             - create session
GET    /api/sessions/{id}        - get session with messages
PATCH  /api/sessions/{id}        - rename session
DELETE /api/sessions/{id}        - delete session
"""

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, HTTPException, status

from backend.api.deps import DBSession, SessionSvcDep
from backend.models.schemas import (
    MessageOut,
    SessionCreate,
    SessionOut,
    SessionRename,
    SessionWithMessages,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=List[SessionOut])
async def list_sessions(db: DBSession, session_svc: SessionSvcDep) -> List[SessionOut]:
    sessions = await session_svc.list_sessions(db)
    return [SessionOut.model_validate(s) for s in sessions]


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    db: DBSession,
    session_svc: SessionSvcDep,
) -> SessionOut:
    session = await session_svc.create_session(db, title=body.title)
    return SessionOut.model_validate(session)


@router.get("/{session_id}", response_model=SessionWithMessages)
async def get_session(
    session_id: str,
    db: DBSession,
    session_svc: SessionSvcDep,
) -> SessionWithMessages:
    session = await session_svc.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = await session_svc.get_messages(db, session_id)
    msg_out = []
    for m in messages:
        sources = json.loads(m.sources) if m.sources else None
        msg_out.append(
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=sources,
                created_at=m.created_at,
            )
        )

    return SessionWithMessages(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=msg_out,
    )


@router.patch("/{session_id}", response_model=SessionOut)
async def rename_session(
    session_id: str,
    body: SessionRename,
    db: DBSession,
    session_svc: SessionSvcDep,
) -> SessionOut:
    session = await session_svc.rename_session(db, session_id, body.title)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return SessionOut.model_validate(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: DBSession,
    session_svc: SessionSvcDep,
) -> None:
    deleted = await session_svc.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
