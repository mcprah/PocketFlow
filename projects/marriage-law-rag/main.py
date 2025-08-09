#!/usr/bin/env python3
"""
Marriage Law RAG System - FastAPI Application
Production-ready RAG system for legal document processing and querying.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime

# Import our custom modules
from flows import create_offline_indexing_flow, create_online_query_flow
from api.models import *
from api.routes import setup_routes
from config.settings import get_settings
from utils.postgres_vector_store import PostgresVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marriage Law RAG System",
    description="Retrieval Augmented Generation system for marriage law research",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get settings
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global state
app.state.vector_store = None
app.state.offline_flow = None
app.state.online_flow = None

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Marriage Law RAG System...")
    
    try:
        # Initialize database connection
        app.state.vector_store = PostgresVectorStore(settings.database_url)
        await app.state.vector_store.connect()
        await app.state.vector_store.create_tables()
        
        # Initialize PocketFlow workflows
        app.state.offline_flow = create_offline_indexing_flow()
        app.state.online_flow = create_online_query_flow()
        
        logger.info("✅ Application initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Marriage Law RAG System...")
    
    # Close database connections
    if app.state.vector_store:
        await app.state.vector_store.close()
    
    logger.info("✅ Application shutdown complete")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if active_connections:
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove broken connections
                if connection in active_connections:
                    active_connections.remove(connection)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    try:
        # Check database connection
        db_status = "connected" if app.state.vector_store else "disconnected"
        
        # Get document counts
        doc_count = await app.state.vector_store.get_document_count() if app.state.vector_store else 0
        chunk_count = await app.state.vector_store.get_chunk_count() if app.state.vector_store else 0
        
        return HealthResponse(
            status="healthy" if db_status == "connected" else "unhealthy",
            database=db_status,
            vector_store="operational" if app.state.vector_store else "unavailable",
            document_count=doc_count,
            chunk_count=chunk_count,
            last_update=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

# Document upload endpoint
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "unknown",
    jurisdiction: str = "unknown",
    authority: Optional[str] = None,
    date_issued: Optional[str] = None
):
    """Upload and process a legal document"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(
            status_code=415, 
            detail="Unsupported file type. Only PDF, DOC, and DOCX files are supported."
        )
    
    # Validate file size (50MB limit)
    file_size = 0
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
    
    # Process document in background
    document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    
    background_tasks.add_task(
        process_document_background,
        document_id,
        str(file_path),
        doc_type,
        jurisdiction,
        authority,
        date_issued
    )
    
    # Broadcast processing start
    await broadcast_message({
        "type": "document_processing",
        "document_id": document_id,
        "filename": file.filename,
        "status": "started"
    })
    
    return DocumentUploadResponse(
        document_id=document_id,
        status="processing",
        message=f"Document {file.filename} is being processed",
        filename=file.filename
    )

async def process_document_background(
    document_id: str,
    file_path: str,
    doc_type: str,
    jurisdiction: str,
    authority: Optional[str],
    date_issued: Optional[str]
):
    """Background task to process uploaded document"""
    try:
        # Broadcast processing status
        await broadcast_message({
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
            "db_connection_string": settings.database_url,
            "vector_store": app.state.vector_store
        }
        
        # Run offline indexing flow
        await app.state.offline_flow.run_async(shared)
        
        # Broadcast completion
        await broadcast_message({
            "type": "processing_complete",
            "document_id": document_id,
            "status": "completed",
            "progress": 100
        })
        
        # Clean up temporary file
        os.remove(file_path)
        
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")
        
        # Broadcast error
        await broadcast_message({
            "type": "processing_error",
            "document_id": document_id,
            "status": "failed",
            "error": str(e)
        })

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document collection"""
    
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters long"
        )
    
    try:
        # Prepare shared state for query
        shared = {
            "user_query": request.question,
            "jurisdiction_filter": request.jurisdiction_filter,
            "doc_type_filter": request.doc_type_filter,
            "authority_level_min": request.authority_level_min,
            "max_results": request.max_results,
            "vector_store": app.state.vector_store,
            "query_results": []
        }
        
        # Run online query flow
        start_time = datetime.now()
        await app.state.online_flow.run_async(shared)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Format response
        results = []
        # print(shared.get("query_results", []))
        for result in shared.get("query_results", []):
            # Extract and map document metadata
            raw_metadata = result.get("document_metadata", {})
            
            # Handle string-encoded legal context
            legal_context_data = result.get("legal_context", {})
            if isinstance(legal_context_data, str):
                try:
                    legal_context_data = json.loads(legal_context_data)
                except:
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
        
        # Format citations as strings
        raw_citations = shared.get("citations", [])
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
        
        return QueryResponse(
            query=request.question,
            results=results,
            generated_answer=shared.get("generated_answer", ""),
            confidence_score=shared.get("confidence_score", 0.0),
            processing_time_ms=int(processing_time),
            citations=formatted_citations
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

# Include additional API routes
setup_routes(app)

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        app,  # Pass the app object directly instead of string
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid subprocess issues
        log_level="info"
    )