#!/usr/bin/env python3
"""
Marriage Law RAG System - FastAPI Application
Production-ready RAG system for legal document processing and querying.
"""

from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import logging

# Import our custom modules
from api.models import *
from api.routes import setup_routes
from config.settings import get_settings
from core import startup_handler, shutdown_handler
from ws import websocket_endpoint, WebSocketManager
from handlers import upload_document, query_documents, health_check

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

# Initialize WebSocket manager
websocket_manager = WebSocketManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    await startup_handler(app, settings)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    await shutdown_handler(app)


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint_handler(websocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_endpoint(websocket)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check_endpoint():
    """System health check"""
    return await health_check(vector_store=app.state.vector_store)


# Main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


# Document upload endpoint
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "unknown",
    jurisdiction: str = "unknown",
    authority: Optional[str] = None,
    date_issued: Optional[str] = None
):
    """Upload and process a legal document"""
    return await upload_document(
        background_tasks=background_tasks,
        file=file,
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        authority=authority,
        date_issued=date_issued,
        vector_store=app.state.vector_store,
        offline_flow=app.state.offline_flow,
        broadcast_function=websocket_manager.broadcast_message,
        database_url=settings.database_url
    )


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents_endpoint(request: QueryRequest):
    """Query the document collection"""
    return await query_documents(
        request=request,
        vector_store=app.state.vector_store,
        online_flow=app.state.online_flow
    )


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