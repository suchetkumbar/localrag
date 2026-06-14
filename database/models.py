"""
Database models and async session management.
Uses SQLAlchemy with aiosqlite for fully async SQLite access.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Session(Base):
    """A chat session (conversation thread)."""

    __tablename__ = "sessions"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: str = Column(String(255), nullable=True)
    created_at: datetime = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at: datetime = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages: list[Message] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at"
    )

    def __repr__(self) -> str:
        return f"<Session id={self.id} title={self.title!r}>"


class Message(Base):
    """A single message in a chat session."""

    __tablename__ = "messages"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    session_id: str = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role: str = Column(String(16), nullable=False)  # "user" | "assistant"
    content: str = Column(Text, nullable=False)
    sources: str = Column(Text, nullable=True)  # JSON array of citations
    created_at: datetime = Column(DateTime, server_default=func.now(), nullable=False)

    session: Session = relationship("Session", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} session={self.session_id}>"


class Document(Base):
    """Metadata record for an ingested document."""

    __tablename__ = "documents"

    id: str = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: str = Column(String(512), nullable=False)
    file_type: str = Column(String(16), nullable=False)
    file_size: int = Column(Integer, nullable=False)
    chunk_count: int = Column(Integer, nullable=False, default=0)
    source: str = Column(String(16), nullable=False)  # "upload" | "watcher"
    status: str = Column(String(16), nullable=False, default="processing")  # "processing"|"ready"|"error"
    error_message: str = Column(Text, nullable=True)
    chroma_ids: str = Column(Text, nullable=True)  # JSON array of chroma chunk IDs
    created_at: datetime = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at: datetime = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(db_path: str) -> AsyncEngine:
    global _engine, _session_factory
    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )
    return _engine


async def create_tables() -> None:
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized. Call init_engine() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    if _engine:
        await _engine.dispose()
