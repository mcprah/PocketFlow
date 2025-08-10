"""
Additional API routes for the Marriage Law RAG system
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
import logging
from datetime import datetime

from .models import *

logger = logging.getLogger(__name__)

def setup_routes(app):
    """Setup additional API routes"""
    
    # Document management routes
    documents_router = APIRouter(prefix="/documents", tags=["documents"])
    
    @documents_router.get("/", response_model=DocumentListResponse)
    async def list_documents(
        doc_type: Optional[str] = Query(None, description="Filter by document type"),
        jurisdiction: Optional[str] = Query(None, description="Filter by jurisdiction"),
        limit: int = Query(50, ge=1, le=200, description="Number of results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        sort: str = Query("date", description="Sort by: date, title, authority_level")
    ):
        """List processed documents with filtering and pagination"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            # Build filters
            filters = {}
            if doc_type:
                filters["doc_type"] = doc_type
            if jurisdiction:
                filters["jurisdiction"] = jurisdiction
            
            documents = await vector_store.list_documents(
                filters=filters,
                limit=limit,
                offset=offset,
                sort=sort
            )
            
            total_count = await vector_store.get_document_count(filters)
            
            return DocumentListResponse(
                documents=documents,
                total_count=total_count,
                limit=limit,
                offset=offset,
                filters_applied=filters if filters else None
            )
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve documents")
    
    @documents_router.get("/{document_id}", response_model=DocumentDetailResponse)
    async def get_document(document_id: str):
        """Get detailed information about a specific document"""
        if not validate_document_id(document_id):
            raise HTTPException(status_code=400, detail="Invalid document ID format")
        
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            document = await vector_store.get_document(document_id)
            if not document:
                raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
            
            # Get document chunks
            chunks = await vector_store.get_document_chunks(document_id)
            
            # Find related documents (simplified - by similar legal concepts)
            related_docs = await vector_store.get_related_documents(document_id, limit=5)
            
            return DocumentDetailResponse(
                document=document,
                chunks=chunks,
                related_documents=[doc["document_id"] for doc in related_docs]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve document")
    
    @documents_router.delete("/{document_id}")
    async def delete_document(document_id: str):
        """Delete a document and all associated chunks"""
        if not validate_document_id(document_id):
            raise HTTPException(status_code=400, detail="Invalid document ID format")
        
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            # Check if document exists
            document = await vector_store.get_document(document_id)
            if not document:
                raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
            
            # Delete document and chunks
            deleted_chunks = await vector_store.delete_document(document_id)
            
            return {
                "message": f"Document {document_id} successfully deleted",
                "deleted_chunks": deleted_chunks
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete document")
    
    # Query enhancement routes
    query_router = APIRouter(prefix="/query", tags=["query"])
    
    @query_router.get("/suggestions")
    async def get_query_suggestions(
        jurisdiction: Optional[str] = Query(None, description="Focus suggestions on jurisdiction"),
        doc_type: Optional[str] = Query(None, description="Focus suggestions on document type"),
        limit: int = Query(10, ge=1, le=20, description="Number of suggestions")
    ):
        """Get suggested queries based on document content"""
        try:
            # Sample suggestions - in production, these would be generated dynamically
            base_suggestions = [
                "What are the requirements for marriage in California?",
                "How is property divided in divorce cases?",
                "What constitutes community property?",
                "Grounds for annulment vs divorce",
                "Same-sex marriage legal precedents",
                "Child custody determination factors",
                "Prenuptial agreement enforceability",
                "Legal separation vs divorce",
                "Marriage license requirements",
                "Domestic partnership benefits"
            ]
            
            # Filter suggestions based on jurisdiction/doc_type
            filtered_suggestions = base_suggestions
            if jurisdiction:
                filtered_suggestions = [
                    s for s in filtered_suggestions 
                    if jurisdiction.lower() in s.lower() or "marriage" in s.lower()
                ]
            
            return {
                "suggestions": filtered_suggestions[:limit]
            }
            
        except Exception as e:
            logger.error(f"Failed to get query suggestions: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve suggestions")
    
    @query_router.get("/history")
    async def get_query_history(
        limit: int = Query(20, ge=1, le=100, description="Number of recent queries"),
        user_id: Optional[str] = Query(None, description="Filter by user ID")
    ):
        """Get recent query history"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            history = await vector_store.get_query_history(limit=limit, user_id=user_id)
            
            return {
                "queries": history,
                "total_count": len(history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get query history: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve query history")
    
    # System administration routes
    admin_router = APIRouter(prefix="/admin", tags=["administration"])
    
    @admin_router.get("/stats", response_model=SystemStatsResponse)
    async def get_system_stats():
        """Get detailed system statistics"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            # Get various statistics
            doc_stats = await vector_store.get_document_statistics()
            processing_stats = await vector_store.get_processing_statistics()
            query_stats = await vector_store.get_query_statistics()
            
            return SystemStatsResponse(
                document_stats=doc_stats,
                processing_stats=processing_stats,
                query_stats=query_stats
            )
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
    
    @admin_router.post("/reindex")
    async def reindex_documents(
        document_ids: Optional[List[str]] = None,
        force: bool = Query(False, description="Force reindexing even if up to date")
    ):
        """Reindex documents (admin only)"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            if document_ids:
                # Reindex specific documents
                for doc_id in document_ids:
                    if not validate_document_id(doc_id):
                        raise HTTPException(status_code=400, detail=f"Invalid document ID: {doc_id}")
                
                results = await vector_store.reindex_documents(document_ids, force=force)
            else:
                # Reindex all documents
                results = await vector_store.reindex_all_documents(force=force)
            
            return {
                "message": "Reindexing completed",
                "results": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to reindex documents: {e}")
            raise HTTPException(status_code=500, detail="Reindexing failed")
    
    @admin_router.get("/config")
    async def get_system_config():
        """Get current system configuration"""
        try:
            config = SystemConfig()
            return config.dict()
        except Exception as e:
            logger.error(f"Failed to get system config: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve configuration")
    
    # Legal-specific routes
    legal_router = APIRouter(prefix="/legal", tags=["legal"])
    
    @legal_router.get("/jurisdictions")
    async def get_jurisdictions():
        """Get list of supported jurisdictions"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            jurisdictions = await vector_store.get_jurisdictions()
            return {"jurisdictions": jurisdictions}
            
        except Exception as e:
            logger.error(f"Failed to get jurisdictions: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve jurisdictions")
    
    @legal_router.get("/concepts")
    async def get_legal_concepts(
        limit: int = Query(50, ge=1, le=200, description="Number of concepts"),
        jurisdiction: Optional[str] = Query(None, description="Filter by jurisdiction")
    ):
        """Get list of legal concepts found in documents"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            concepts = await vector_store.get_legal_concepts(limit=limit, jurisdiction=jurisdiction)
            return {"concepts": concepts}
            
        except Exception as e:
            logger.error(f"Failed to get legal concepts: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve legal concepts")
    
    @legal_router.get("/citations")
    async def get_citations(
        limit: int = Query(50, ge=1, le=200, description="Number of citations"),
        doc_type: Optional[str] = Query(None, description="Filter by document type")
    ):
        """Get list of legal citations found in documents"""
        try:
            vector_store = app.state.vector_store
            if not vector_store:
                raise HTTPException(status_code=503, detail="Vector store not available")
            
            citations = await vector_store.get_citations(limit=limit, doc_type=doc_type)
            return {"citations": citations}
            
        except Exception as e:
            logger.error(f"Failed to get citations: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve citations")
    
    # Bulk processing routes
    bulk_router = APIRouter(prefix="/bulk", tags=["bulk_processing"])
    
    from handlers.bulk_handler import (
        handle_bulk_scan,
        handle_bulk_process,
        handle_bulk_retry,
        handle_bulk_stats,
        handle_estimate_processing
    )
    
    @bulk_router.post("/scan")
    async def scan_folder(
        folder_path: str = Query(..., description="Path to folder containing documents"),
        max_files: Optional[int] = Query(None, description="Maximum number of files to scan")
    ):
        """Scan folder structure and discover documents for bulk processing"""
        return await handle_bulk_scan(folder_path, max_files)
    
    @bulk_router.post("/process")
    async def start_bulk_processing(
        folder_path: str = Query(..., description="Path to folder containing documents"),
        max_files: Optional[int] = Query(None, description="Maximum number of files to process"),
        batch_size: int = Query(50, ge=1, le=200, description="Documents per processing batch"),
        max_workers_extract: int = Query(4, ge=1, le=16, description="Max workers for extraction"),
        max_workers_chunk: int = Query(8, ge=1, le=32, description="Max workers for chunking"),
        max_workers_embed: int = Query(2, ge=1, le=8, description="Max workers for embedding"),
        max_workers_store: int = Query(4, ge=1, le=16, description="Max workers for storage")
    ):
        """Start bulk processing of documents in a folder"""
        max_workers = {
            "extract": max_workers_extract,
            "chunk": max_workers_chunk,
            "embed": max_workers_embed,
            "store": max_workers_store
        }
        
        return await handle_bulk_process(
            folder_path=folder_path,
            app_state=app.state,
            max_files=max_files,
            batch_size=batch_size,
            max_workers=max_workers
        )
    
    @bulk_router.post("/retry")
    async def retry_failed_processing(
        operation_id: str = Query(..., description="Original operation ID"),
        failed_documents: List[Dict] = None,
        max_workers_extract: int = Query(2, ge=1, le=8, description="Max workers for extraction"),
        max_workers_chunk: int = Query(4, ge=1, le=16, description="Max workers for chunking"),
        max_workers_embed: int = Query(1, ge=1, le=4, description="Max workers for embedding"),
        max_workers_store: int = Query(2, ge=1, le=8, description="Max workers for storage")
    ):
        """Retry processing of previously failed documents"""
        if not failed_documents:
            raise HTTPException(status_code=400, detail="failed_documents is required")
        
        max_workers = {
            "extract": max_workers_extract,
            "chunk": max_workers_chunk,
            "embed": max_workers_embed,
            "store": max_workers_store
        }
        
        return await handle_bulk_retry(
            operation_id=operation_id,
            failed_documents=failed_documents,
            app_state=app.state,
            max_workers=max_workers
        )
    
    @bulk_router.get("/stats")
    async def get_bulk_processing_stats():
        """Get bulk processing statistics from the database"""
        return await handle_bulk_stats(app.state)
    
    @bulk_router.get("/estimate")
    async def estimate_processing_time(
        folder_path: str = Query(..., description="Path to folder containing documents"),
        max_files: Optional[int] = Query(None, description="Maximum number of files to analyze")
    ):
        """Estimate processing time and resource requirements"""
        return await handle_estimate_processing(folder_path, max_files)
    
    # Include routers in the main app
    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(admin_router)
    app.include_router(legal_router)
    app.include_router(bulk_router)