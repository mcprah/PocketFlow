#!/usr/bin/env python3
"""
Bulk Processing Handler - FastAPI endpoints for bulk document processing
"""

import os
import sys
import logging
from typing import Optional, Dict, List
from fastapi import HTTPException
import asyncio
import uuid

# Add parent directory to path for project imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tasks.bulk_document_processor import (
    process_bulk_documents,
    retry_failed_documents,
    estimate_processing_time
)
from utils.bulk_processor import BulkDocumentProcessor, validate_document_collection

logger = logging.getLogger(__name__)


async def handle_bulk_scan(folder_path: str, max_files: Optional[int] = None) -> Dict:
    """
    Scan folder structure and return document discovery results.
    
    Args:
        folder_path: Path to folder containing documents
        max_files: Maximum number of files to scan (None for unlimited)
        
    Returns:
        Dictionary with scan results and statistics
    """
    
    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder path does not exist: {folder_path}"
        )
    
    if not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {folder_path}"
        )
    
    try:
        # Initialize bulk processor and scan
        bulk_processor = BulkDocumentProcessor()
        documents = bulk_processor.scan_folders(folder_path, max_files=max_files)
        
        if not documents:
            return {
                "status": "no_documents",
                "message": "No supported documents found in the specified folder",
                "folder_path": folder_path,
                "documents": [],
                "stats": {}
            }
        
        # Validate discovered documents
        valid_documents, validation_errors = validate_document_collection(documents)
        
        # Generate processing statistics
        stats = bulk_processor.get_processing_stats(valid_documents)
        
        # Generate time estimates
        time_estimates = estimate_processing_time(valid_documents)
        
        # Organize documents by type for preview
        organized_docs = bulk_processor.organize_by_type(valid_documents)
        
        return {
            "status": "success",
            "folder_path": folder_path,
            "total_discovered": len(documents),
            "total_valid": len(valid_documents),
            "validation_errors": validation_errors[:10],  # First 10 errors
            "documents": valid_documents[:50],  # Preview first 50 documents
            "organized_by_type": {
                doc_type: len(docs) for doc_type, docs in organized_docs.items()
            },
            "processing_stats": stats,
            "time_estimates": time_estimates
        }
        
    except Exception as e:
        logger.error(f"Bulk scan failed for {folder_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scan failed: {str(e)}"
        )


async def handle_bulk_process(
    folder_path: str,
    app_state,  # Pass app.state as parameter
    max_files: Optional[int] = None,
    batch_size: int = 50,
    max_workers: Optional[Dict] = None
) -> Dict:
    """
    Start bulk processing of documents in a folder.
    
    Args:
        folder_path: Path to folder containing documents
        app_state: FastAPI app.state object
        max_files: Maximum number of files to process
        batch_size: Number of documents per processing batch
        max_workers: Worker limits for parallel processing stages
        
    Returns:
        Dictionary with operation details and status
    """
    
    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder path does not exist: {folder_path}"
        )
    
    # Generate unique operation ID
    operation_id = f"bulk_process_{uuid.uuid4().hex[:8]}"
    
    try:
        # Check vector store availability
        if not hasattr(app_state, 'vector_store') or not app_state.vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
        
        # Validate batch_size
        if batch_size < 1 or batch_size > 200:
            raise HTTPException(
                status_code=400,
                detail="batch_size must be between 1 and 200"
            )
        
        # Get broadcast function if available
        broadcast_function = None
        if hasattr(app_state, 'websocket_manager') and app_state.websocket_manager:
            broadcast_function = app_state.websocket_manager.broadcast_to_all
        
        # Get database URL if available
        database_url = getattr(app_state, 'database_url', None)
        
        # Start bulk processing as background task
        asyncio.create_task(
            process_bulk_documents(
                folder_path=folder_path,
                operation_id=operation_id,
                vector_store=app_state.vector_store,
                broadcast_function=broadcast_function,
                database_url=database_url,
                max_files=max_files,
                batch_size=batch_size,
                max_workers=max_workers
            )
        )
        
        logger.info(f"Started bulk processing operation: {operation_id}")
        
        return {
            "status": "started",
            "operation_id": operation_id,
            "folder_path": folder_path,
            "max_files": max_files,
            "batch_size": batch_size,
            "max_workers": max_workers or {},
            "message": "Bulk processing started. Monitor progress via WebSocket.",
            "websocket_endpoint": "/ws"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start bulk processing for {folder_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start bulk processing: {str(e)}"
        )


async def handle_bulk_retry(
    operation_id: str,
    failed_documents: List[Dict],
    app_state,  # Pass app.state as parameter
    max_workers: Optional[Dict] = None
) -> Dict:
    """
    Retry processing of previously failed documents.
    
    Args:
        operation_id: Original operation ID for tracking
        failed_documents: List of failed document entries
        app_state: FastAPI app.state object
        max_workers: Worker limits for retry processing
        
    Returns:
        Dictionary with retry operation details
    """
    
    if not failed_documents:
        raise HTTPException(
            status_code=400,
            detail="No failed documents provided for retry"
        )
    
    retry_operation_id = f"{operation_id}_retry_{uuid.uuid4().hex[:4]}"
    
    try:
        # Check vector store availability
        if not hasattr(app_state, 'vector_store') or not app_state.vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
        
        # Get broadcast function if available
        broadcast_function = None
        if hasattr(app_state, 'websocket_manager') and app_state.websocket_manager:
            broadcast_function = app_state.websocket_manager.broadcast_to_all
        
        # Get database URL if available
        database_url = getattr(app_state, 'database_url', None)
        
        # Start retry processing as background task
        asyncio.create_task(
            retry_failed_documents(
                failed_documents=failed_documents,
                operation_id=retry_operation_id,
                vector_store=app_state.vector_store,
                broadcast_function=broadcast_function,
                database_url=database_url,
                max_workers=max_workers
            )
        )
        
        logger.info(f"Started retry operation: {retry_operation_id}")
        
        return {
            "status": "retry_started",
            "retry_operation_id": retry_operation_id,
            "original_operation_id": operation_id,
            "documents_to_retry": len(failed_documents),
            "max_workers": max_workers or {},
            "message": "Retry processing started. Monitor progress via WebSocket."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start retry processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start retry processing: {str(e)}"
        )


async def handle_bulk_stats(app_state) -> Dict:
    """
    Get bulk processing statistics from the database.
    
    Args:
        app_state: FastAPI app.state object
        
    Returns:
        Dictionary with processing statistics
    """
    
    try:
        # Check vector store availability
        if not hasattr(app_state, 'vector_store') or not app_state.vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )
        
        # Get bulk processing statistics
        stats = await app_state.vector_store.get_bulk_processing_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bulk processing stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get processing statistics: {str(e)}"
        )


async def handle_estimate_processing(folder_path: str, max_files: Optional[int] = None) -> Dict:
    """
    Estimate processing time and resources for a document collection.
    
    Args:
        folder_path: Path to folder containing documents
        max_files: Maximum number of files to analyze
        
    Returns:
        Dictionary with processing estimates
    """
    
    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder path does not exist: {folder_path}"
        )
    
    try:
        # Scan folder to get document list
        bulk_processor = BulkDocumentProcessor()
        documents = bulk_processor.scan_folders(folder_path, max_files=max_files)
        
        if not documents:
            return {
                "status": "no_documents",
                "message": "No documents found for estimation",
                "estimates": {}
            }
        
        # Validate documents
        valid_documents, _ = validate_document_collection(documents)
        
        # Generate estimates
        estimates = estimate_processing_time(valid_documents)
        
        # Add processing recommendations
        recommendations = {
            "optimal_batch_size": estimates["recommended_batch_size"],
            "suggested_max_workers": {
                "extract": min(4, max(2, len(valid_documents) // 50)),
                "chunk": min(8, max(4, len(valid_documents) // 25)),
                "embed": 2,  # Limited by API rate limits
                "store": min(4, max(2, len(valid_documents) // 50))
            },
            "processing_tips": []
        }
        
        # Add specific tips based on collection size
        if len(valid_documents) > 1000:
            recommendations["processing_tips"].extend([
                "Consider processing during off-peak hours",
                "Monitor API rate limits for embedding generation",
                "Ensure sufficient database connection pool size"
            ])
        
        if estimates["total_size_mb"] > 500:
            recommendations["processing_tips"].extend([
                "Large files detected - extraction may take longer",
                "Consider increasing memory allocation for workers"
            ])
        
        return {
            "status": "success",
            "folder_path": folder_path,
            "estimates": estimates,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Processing estimation failed for {folder_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Estimation failed: {str(e)}"
        )