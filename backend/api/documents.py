"""
Document management API endpoints.
POST   /api/documents/upload     - upload + ingest a file
GET    /api/documents            - list all documents
GET    /api/documents/{id}       - get document metadata
DELETE /api/documents/{id}       - delete document
GET    /api/documents/stats      - vector store stats
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from sqlalchemy import select

from backend.api.deps import DBSession, IngestionDep, SettingsDep, VectorStoreDep
from backend.models.schemas import CollectionStats, DocumentListResponse, DocumentOut
from database.models import Document as DocumentModel

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: DBSession = ...,
    ingestion_svc: IngestionDep = ...,
    settings: SettingsDep = ...,
) -> DocumentOut:
    """Upload and ingest a document file."""
    # Validate file type
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ingestion.supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Supported: {settings.ingestion.supported_extensions}",
        )

    # Save to uploads folder
    upload_path = settings.uploads_folder_path / f"{uuid.uuid4()}{suffix}"
    try:
        with upload_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    finally:
        await file.close()

    # Ingest
    doc = await ingestion_svc.ingest_file(upload_path, db_session=db, source="upload")

    # Rename file to doc ID for traceability
    final_path = settings.uploads_folder_path / f"{doc.id}{suffix}"
    upload_path.rename(final_path)

    return DocumentOut.model_validate(doc)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    db: DBSession,
    status_filter: str = "all",  # was: status; shadowed fastapi.status import
) -> DocumentListResponse:
    q = select(DocumentModel).order_by(DocumentModel.created_at.desc())
    if status_filter != "all":
        q = q.where(DocumentModel.status == status_filter)
    result = await db.execute(q)
    docs = result.scalars().all()
    return DocumentListResponse(
        documents=[DocumentOut.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.get("/stats", response_model=CollectionStats)
async def get_stats(vector_store: VectorStoreDep) -> CollectionStats:
    stats = await vector_store.get_collection_stats()
    return CollectionStats(**stats)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, db: DBSession) -> DocumentOut:
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: DBSession,
    ingestion_svc: IngestionDep,
) -> Response:
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await ingestion_svc.delete_document(doc, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
