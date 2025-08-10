#!/usr/bin/env python3
"""
Document Processor - Background task for processing uploaded documents
"""

import os
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


async def process_document_background(
    document_id: str,
    file_path: str,
    doc_type: str,
    jurisdiction: str,
    authority: Optional[str],
    date_issued: Optional[str],
    vector_store,
    offline_flow,
    broadcast_function,
    database_url: str
):
    """
    Background task to process uploaded document
    
    Args:
        document_id: Unique identifier for the document
        file_path: Path to the uploaded file
        doc_type: Type of document (e.g., 'law', 'regulation')
        jurisdiction: Legal jurisdiction
        authority: Issuing authority (optional)
        date_issued: Date the document was issued (optional)
        vector_store: Vector store instance for database operations
        offline_flow: PocketFlow offline indexing flow
        broadcast_function: Function to broadcast WebSocket messages
        database_url: Database connection string
    """
    try:
        # Broadcast processing start status
        await broadcast_function({
            "type": "document_processing",
            "document_id": document_id,
            "status": "extracting",
            "progress": 25
        })
        
        # Prepare shared state for PocketFlow
        shared = {
            "raw_documents": [{
                "path": file_path,
                "doc_type": doc_type,
                "jurisdiction": jurisdiction,
                "authority": authority,
                "date_issued": date_issued,
                "document_id": document_id
            }],
            "db_connection_string": database_url,
            "vector_store": vector_store
        }
        
        # Broadcast processing status
        await broadcast_function({
            "type": "document_processing",
            "document_id": document_id,
            "status": "processing",
            "progress": 50
        })
        
        # Run offline indexing flow
        await offline_flow.run_async(shared)
        
        # Broadcast completion
        await broadcast_function({
            "type": "processing_complete",
            "document_id": document_id,
            "status": "completed",
            "progress": 100,
            "timestamp": datetime.now().isoformat()
        })
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        
        logger.info(f"Successfully processed document: {document_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")
        
        # Broadcast error status
        await broadcast_function({
            "type": "processing_error",
            "document_id": document_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        
        # Clean up temporary file even on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file after error: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file {file_path}: {cleanup_error}")