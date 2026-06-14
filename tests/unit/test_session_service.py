"""
Unit tests for SessionService.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from backend.services.session_service import SessionService


@pytest.fixture
def session_svc():
    return SessionService()


def make_session(sid="sess-1", title="Test Chat"):
    from database.models import Session as SessionModel
    s = MagicMock(spec=SessionModel)
    s.id = sid
    s.title = title
    s.created_at = datetime(2024, 1, 1)
    s.updated_at = datetime(2024, 1, 1)
    s.messages = []
    return s


# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session(session_svc):
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()

    session = await session_svc.create_session(mock_db, title="My Chat")

    mock_db.add.assert_called_once()
    assert session.title == "My Chat"


@pytest.mark.asyncio
async def test_create_session_default_title(session_svc):
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()

    session = await session_svc.create_session(mock_db)
    assert session.title == "New Chat"


@pytest.mark.asyncio
async def test_get_session_found(session_svc):
    expected = make_session()
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = expected
    mock_db.execute = AsyncMock(return_value=result)

    session = await session_svc.get_session(mock_db, "sess-1")
    assert session.id == "sess-1"


@pytest.mark.asyncio
async def test_get_session_not_found(session_svc):
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=result)

    session = await session_svc.get_session(mock_db, "nonexistent")
    assert session is None


@pytest.mark.asyncio
async def test_list_sessions(session_svc):
    sessions = [make_session("s1"), make_session("s2")]
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = sessions
    mock_db.execute = AsyncMock(return_value=result)

    listing = await session_svc.list_sessions(mock_db)
    assert len(listing) == 2


@pytest.mark.asyncio
async def test_delete_session_success(session_svc):
    s = make_session()
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = s
    mock_db.execute = AsyncMock(return_value=result)
    mock_db.delete = AsyncMock()
    mock_db.flush = AsyncMock()

    deleted = await session_svc.delete_session(mock_db, "sess-1")
    assert deleted is True
    mock_db.delete.assert_awaited_once_with(s)


@pytest.mark.asyncio
async def test_delete_session_not_found(session_svc):
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=result)

    deleted = await session_svc.delete_session(mock_db, "bad-id")
    assert deleted is False


@pytest.mark.asyncio
async def test_rename_session(session_svc):
    s = make_session(title="Old Title")
    mock_db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = s
    mock_db.execute = AsyncMock(return_value=result)
    mock_db.flush = AsyncMock()

    renamed = await session_svc.rename_session(mock_db, "sess-1", "New Title")
    assert renamed.title == "New Title"
