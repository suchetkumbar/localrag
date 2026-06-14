"""
Pydantic schemas for all API request and response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)


class SessionRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[List[dict]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionWithMessages(SessionOut):
    messages: List[MessageOut] = []


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatSource(BaseModel):
    filename: str
    chunk_index: int
    page: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
    session_id: str


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    source: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: List[DocumentOut]
    total: int


# ---------------------------------------------------------------------------
# Health & Stats
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_connected: bool
    chroma_chunks: int
    sqlite_connected: bool


class CollectionStats(BaseModel):
    collection_name: str
    chunk_count: int
    embed_model: str
    persist_dir: str
