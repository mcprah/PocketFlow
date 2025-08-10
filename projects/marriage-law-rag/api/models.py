"""
Pydantic models for the Marriage Law RAG API
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Legal document types"""
    case = "case"
    statute = "statute"
    article = "article"
    gazette = "gazette"
    regulation = "regulation"

class Jurisdiction(str, Enum):
    """Legal jurisdictions"""
    federal = "federal"
    california = "california"
    new_york = "new_york"
    texas = "texas"
    florida = "florida"
    local = "local"

# Request Models

class QueryRequest(BaseModel):
    """Request model for document queries"""
    question: str = Field(..., min_length=3, max_length=500, description="Legal question to search for")
    jurisdiction_filter: Optional[List[str]] = Field(None, description="Filter by jurisdictions")
    doc_type_filter: Optional[List[str]] = Field(None, description="Filter by document types")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter (start/end)")
    authority_level_min: Optional[int] = Field(1, ge=1, le=10, description="Minimum authority level")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results")
    include_citations: Optional[bool] = Field(True, description="Include formatted citations")
    
    @field_validator('jurisdiction_filter')
    @classmethod
    def validate_jurisdictions(cls, v):
        if v is not None:
            valid_jurisdictions = [j.value for j in Jurisdiction]
            for jurisdiction in v:
                if jurisdiction not in valid_jurisdictions:
                    raise ValueError(f"Invalid jurisdiction: {jurisdiction}")
        return v
    
    @field_validator('doc_type_filter')
    @classmethod
    def validate_doc_types(cls, v):
        if v is not None:
            valid_types = [t.value for t in DocumentType]
            for doc_type in v:
                if doc_type not in valid_types:
                    raise ValueError(f"Invalid document type: {doc_type}")
        return v

class DocumentUploadRequest(BaseModel):
    """Request model for document upload metadata"""
    doc_type: DocumentType = Field(..., description="Type of legal document")
    jurisdiction: str = Field(..., description="Legal jurisdiction")
    authority: Optional[str] = Field(None, description="Issuing authority")
    date_issued: Optional[str] = Field(None, description="Document date (YYYY-MM-DD)")

# Response Models

class DocumentMetadata(BaseModel):
    """Document metadata response"""
    document_id: str
    title: Optional[str] = None
    doc_type: str
    jurisdiction: str
    authority: Optional[str] = None
    authority_level: Optional[int] = None
    date_issued: Optional[str] = None
    file_path: Optional[str] = None

class LegalContext(BaseModel):
    """Legal context information"""
    section: Optional[str] = None
    legal_concepts: List[str] = []
    citations: List[str] = []
    case_number: Optional[str] = None
    court: Optional[str] = None

class QueryResult(BaseModel):
    """Individual query result"""
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    document_metadata: DocumentMetadata
    legal_context: LegalContext

class QueryResponse(BaseModel):
    """Response model for document queries"""
    query: str
    results: List[QueryResult]
    generated_answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int
    citations: List[str] = []
    total_documents_searched: Optional[int] = None

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    status: str = Field(..., description="Processing status")
    message: str
    filename: str
    processing_started_at: Optional[datetime] = None
    estimated_completion: Optional[str] = None

class DocumentProcessingResult(BaseModel):
    """Detailed processing result"""
    document_id: str
    status: str
    extraction_method: Optional[str] = None
    quality_score: Optional[float] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    legal_metadata: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    error_details: Optional[str] = None

class DocumentListResponse(BaseModel):
    """Response for document listing"""
    documents: List[DocumentMetadata]
    total_count: int
    limit: int
    offset: int
    filters_applied: Optional[Dict[str, Any]] = None

class DocumentDetailResponse(BaseModel):
    """Detailed document information"""
    document: DocumentMetadata
    extraction_details: Optional[Dict[str, Any]] = None
    legal_metadata: Optional[Dict[str, Any]] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    related_documents: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """System health check response"""
    status: str = Field(..., description="Overall system status")
    database: str = Field(..., description="Database connection status")  
    vector_store: str = Field(..., description="Vector store status")
    document_count: int = Field(..., description="Total indexed documents")
    chunk_count: int = Field(..., description="Total indexed chunks")
    last_update: str = Field(..., description="Last system update timestamp")

class SystemStatsResponse(BaseModel):
    """System statistics response"""
    document_stats: Dict[str, Any]
    processing_stats: Dict[str, Any]
    query_stats: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    error: Dict[str, Any] = Field(..., description="Error details")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "DOCUMENT_NOT_FOUND",
                    "message": "Document with ID 'doc_123' not found",
                    "details": {
                        "document_id": "doc_123",
                        "suggestion": "Check the document ID and try again"
                    }
                }
            }
        }
    }

# WebSocket Models

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now)

class ProcessingUpdateMessage(BaseModel):
    """Processing status update via WebSocket"""
    type: str = "processing_update"
    document_id: str
    status: str
    progress: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None

# Validation helpers

def validate_document_id(document_id: str) -> bool:
    """Validate document ID format"""
    return document_id.startswith("doc_") and len(document_id) > 4

def validate_chunk_id(chunk_id: str) -> bool:
    """Validate chunk ID format"""
    return chunk_id.startswith("chunk_") and len(chunk_id) > 6

# Configuration Models

class SystemConfig(BaseModel):
    """System configuration model"""
    max_file_size_mb: int = 50
    supported_file_types: List[str] = [".pdf", ".doc", ".docx"]
    max_query_length: int = 500
    max_results_per_query: int = 20
    ocr_enabled: bool = True
    embedding_model: str = "text-embedding-3-small-ada-002"
    vector_dimensions: int = 1536