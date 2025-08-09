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
    
    # Include routers in the main app
    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(admin_router)
    app.include_router(legal_router)