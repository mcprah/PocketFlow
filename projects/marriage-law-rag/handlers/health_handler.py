#!/usr/bin/env python3
"""
Health Handler - Handles system health monitoring endpoints
"""

from fastapi import HTTPException
from datetime import datetime
import logging

from api.models import HealthResponse

logger = logging.getLogger(__name__)


async def health_check(vector_store=None):
    """
    Perform system health check
    
    Args:
        vector_store: Vector store instance to check
    
    Returns:
        HealthResponse with system status information
    """
    try:
        # Check database connection status
        db_status = "connected" if vector_store else "disconnected"
        
        # Get document and chunk counts if vector store is available
        doc_count = 0
        chunk_count = 0
        
        if vector_store:
            try:
                doc_count = await vector_store.get_document_count()
                chunk_count = await vector_store.get_chunk_count()
            except Exception as e:
                logger.warning(f"Failed to get counts from vector store: {e}")
                db_status = "error"
        
        # Determine overall system status
        overall_status = "healthy" if db_status == "connected" else "unhealthy"
        vector_store_status = "operational" if vector_store else "unavailable"
        
        health_response = HealthResponse(
            status=overall_status,
            database=db_status,
            vector_store=vector_store_status,
            document_count=doc_count,
            chunk_count=chunk_count,
            last_update=datetime.now().isoformat()
        )
        
        logger.info(f"Health check completed: {overall_status} (docs: {doc_count}, chunks: {chunk_count})")
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")