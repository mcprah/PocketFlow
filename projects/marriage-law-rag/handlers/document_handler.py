#!/usr/bin/env python3
"""
Document Handler - Handles document upload and processing endpoints
"""

from fastapi import BackgroundTasks, File, UploadFile, HTTPException
from typing import Optional
from pathlib import Path
from datetime import datetime
import logging

from api.models import DocumentUploadResponse
from tasks.document_processor import process_document_background

logger = logging.getLogger(__name__)


async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "unknown",
    jurisdiction: str = "unknown",
    authority: Optional[str] = None,
    date_issued: Optional[str] = None,
    # Dependencies injected by main.py
    vector_store=None,
    offline_flow=None,
    broadcast_function=None,
    database_url: str = None
):
    """
    Upload and process a legal document
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        doc_type: Type of document
        jurisdiction: Legal jurisdiction
        authority: Issuing authority (optional)
        date_issued: Date issued (optional)
        vector_store: Vector store instance
        offline_flow: PocketFlow offline indexing flow
        broadcast_function: WebSocket broadcast function
        database_url: Database connection string
    
    Returns:
        DocumentUploadResponse with processing status
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(
            status_code=415, 
            detail="Unsupported file type. Only PDF, DOC, and DOCX files are supported."
        )
    
    # Validate file size (50MB limit)
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum file size is 50MB."
        )
    
    # Save uploaded file temporarily
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Generate unique document ID
    document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    
    # Process document in background
    background_tasks.add_task(
        process_document_background,
        document_id,
        str(file_path),
        doc_type,
        jurisdiction,
        authority,
        date_issued,
        vector_store,
        offline_flow,
        broadcast_function,
        database_url
    )
    
    # Broadcast processing start
    if broadcast_function:
        await broadcast_function({
            "type": "document_processing",
            "document_id": document_id,
            "filename": file.filename,
            "status": "started"
        })
    
    logger.info(f"Document upload started: {document_id} ({file.filename})")
    
    return DocumentUploadResponse(
        document_id=document_id,
        status="processing",
        message=f"Document {file.filename} is being processed",
        filename=file.filename
    )