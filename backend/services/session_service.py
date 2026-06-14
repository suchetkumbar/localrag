"""
Session management service.
Creates, retrieves, and deletes chat sessions in SQLite.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Session as SessionModel, Message
from monitoring.metrics import SESSIONS_ACTIVE, SESSIONS_CREATED_TOTAL

logger = structlog.get_logger(__name__)


class SessionService:

    async def create_session(
        self, db: AsyncSession, title: Optional[str] = None
    ) -> SessionModel:
        session = SessionModel(id=str(uuid.uuid4()), title=title or "New Chat")
        db.add(session)
        await db.flush()
        SESSIONS_CREATED_TOTAL.inc()
        SESSIONS_ACTIVE.inc()
        logger.info("session_created", session_id=session.id)
        return session

    async def get_session(
        self, db: AsyncSession, session_id: str
    ) -> Optional[SessionModel]:
        result = await db.execute(
            select(SessionModel).where(SessionModel.id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(self, db: AsyncSession) -> List[SessionModel]:
        result = await db.execute(
            select(SessionModel).order_by(SessionModel.updated_at.desc())
        )
        return list(result.scalars().all())

    async def delete_session(self, db: AsyncSession, session_id: str) -> bool:
        session = await self.get_session(db, session_id)
        if session is None:
            return False
        await db.delete(session)
        await db.flush()
        SESSIONS_ACTIVE.dec()
        logger.info("session_deleted", session_id=session_id)
        return True

    async def get_messages(
        self, db: AsyncSession, session_id: str
    ) -> List[Message]:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        return list(result.scalars().all())

    async def rename_session(
        self, db: AsyncSession, session_id: str, title: str
    ) -> Optional[SessionModel]:
        session = await self.get_session(db, session_id)
        if session is None:
            return None
        session.title = title
        await db.flush()
        return session
