#!/usr/bin/env python3
"""
Query Handler - Handles document query processing endpoints
"""

from fastapi import HTTPException
from datetime import datetime
import json
import logging

from api.models import QueryRequest, QueryResponse, QueryResult

logger = logging.getLogger(__name__)


async def query_documents(
    request: QueryRequest,
    # Dependencies injected by main.py
    vector_store=None,
    online_flow=None
):
    """
    Query the document collection using RAG
    
    Args:
        request: Query request with question and filters
        vector_store: Vector store instance
        online_flow: PocketFlow online query flow
    
    Returns:
        QueryResponse with results and generated answer
    """
    
    # Validate query length
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters long"
        )
    
    try:
        # Prepare shared state for query processing
        shared = {
            "user_query": request.question,
            "jurisdiction_filter": request.jurisdiction_filter,
            "doc_type_filter": request.doc_type_filter,
            "authority_level_min": request.authority_level_min,
            "max_results": request.max_results,
            "vector_store": vector_store,
            "query_results": []
        }
        
        # Run online query flow and measure processing time
        start_time = datetime.now()
        await online_flow.run_async(shared)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Format query results
        results = _format_query_results(shared.get("query_results", []))
        
        # Format citations
        formatted_citations = _format_citations(shared.get("citations", []))
        
        logger.info(f"Query processed successfully: '{request.question[:50]}...' ({len(results)} results)")
        
        return QueryResponse(
            query=request.question,
            results=results,
            generated_answer=shared.get("generated_answer", ""),
            confidence_score=shared.get("confidence_score", 0.0),
            processing_time_ms=int(processing_time),
            citations=formatted_citations
        )
        
    except Exception as e:
        logger.error(f"Query processing failed for '{request.question[:50]}...': {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


def _format_query_results(query_results: list) -> list:
    """
    Format raw query results into QueryResult objects
    
    Args:
        query_results: Raw results from the query flow
    
    Returns:
        List of formatted QueryResult objects
    """
    results = []
    
    for result in query_results:
        # Extract and map document metadata
        raw_metadata = result.get("document_metadata", {})
        
        # Handle string-encoded legal context
        legal_context_data = result.get("legal_context", {})
        if isinstance(legal_context_data, str):
            try:
                legal_context_data = json.loads(legal_context_data)
            except json.JSONDecodeError:
                legal_context_data = {}
        
        # Map document metadata to expected structure
        document_metadata = {
            "document_id": result.get("document_id", raw_metadata.get("document_id", "unknown")),
            "title": raw_metadata.get("title"),
            "doc_type": raw_metadata.get("doc_type") or raw_metadata.get("document_type", "unknown"),
            "jurisdiction": raw_metadata.get("jurisdiction", "unknown"),
            "authority": raw_metadata.get("authority"),
            "authority_level": raw_metadata.get("authority_level"),
            "date_issued": raw_metadata.get("date_issued"),
            "file_path": raw_metadata.get("file_path")
        }
        
        # Map legal context to expected structure
        legal_context = {
            "section": legal_context_data.get("section"),
            "legal_concepts": legal_context_data.get("legal_concepts", []),
            "citations": legal_context_data.get("citations", []),
            "case_number": legal_context_data.get("case_number"),
            "court": legal_context_data.get("court")
        }
        
        results.append(QueryResult(
            chunk_id=result.get("chunk_id", "unknown"),
            document_id=result.get("document_id", "unknown"),
            text=result.get("text", ""),
            similarity_score=result.get("similarity_score", 0.0),
            document_metadata=document_metadata,
            legal_context=legal_context
        ))
    
    return results


def _format_citations(raw_citations: list) -> list:
    """
    Format citations into string representations
    
    Args:
        raw_citations: Raw citation objects
    
    Returns:
        List of formatted citation strings
    """
    formatted_citations = []
    
    for citation in raw_citations:
        if isinstance(citation, dict):
            # Convert citation dict to formatted string
            doc_id = citation.get("document_id", "Unknown")
            chunk_id = citation.get("chunk_id", "Unknown")
            score = citation.get("similarity_score", 0.0)
            formatted_citations.append(f"{doc_id}/{chunk_id} (similarity: {score:.2f})")
        else:
            # Already a string
            formatted_citations.append(str(citation))
    
    return formatted_citations