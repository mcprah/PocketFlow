"""
Configuration settings for the Marriage Law RAG system
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Database settings
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/marriage_law_db",
        env="DATABASE_URL",
        description="PostgreSQL connection string"
    )
    
    # OpenAI settings
    openai_api_key: str = Field(
        ..., 
        env="OPENAI_API_KEY",
        description="OpenAI API key for embeddings and LLM"
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-small-ada-002",
        env="EMBEDDING_MODEL",
        description="OpenAI embedding model to use"
    )
    
    llm_model: str = Field(
        default="gpt-4",
        env="LLM_MODEL", 
        description="OpenAI model for text generation"
    )
    
    # OCR settings
    tesseract_cmd: str = Field(
        default="/usr/bin/tesseract",
        env="TESSERACT_CMD",
        description="Path to Tesseract OCR binary"
    )
    
    ocr_enabled: bool = Field(
        default=True,
        env="OCR_ENABLED",
        description="Enable OCR for scanned documents"
    )
    
    # Application settings
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    
    log_level: str = Field(
        default="info",
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    # File processing settings
    max_file_size_mb: int = Field(
        default=50,
        env="MAX_FILE_SIZE_MB",
        description="Maximum file size for uploads"
    )
    
    upload_dir: str = Field(
        default="uploads",
        env="UPLOAD_DIR",
        description="Directory for uploaded files"
    )
    
    # Vector store settings
    vector_dimensions: int = Field(
        default=1536,
        env="VECTOR_DIMENSIONS",
        description="Vector embedding dimensions"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        env="SIMILARITY_THRESHOLD",
        description="Minimum similarity score for results"
    )
    
    # Query settings
    max_query_length: int = Field(
        default=500,
        env="MAX_QUERY_LENGTH",
        description="Maximum query length in characters"
    )
    
    max_results_per_query: int = Field(
        default=20,
        env="MAX_RESULTS_PER_QUERY",
        description="Maximum results returned per query"
    )
    
    default_results_per_query: int = Field(
        default=5,
        env="DEFAULT_RESULTS_PER_QUERY",
        description="Default number of results per query"
    )
    
    # Security settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000"],
        env="ALLOWED_ORIGINS",
        description="CORS allowed origins"
    )
    
    # Redis settings (optional caching)
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL",
        description="Redis connection string for caching"
    )
    
    cache_enabled: bool = Field(
        default=False,
        env="CACHE_ENABLED",
        description="Enable Redis caching"
    )
    
    cache_ttl_seconds: int = Field(
        default=3600,
        env="CACHE_TTL_SECONDS",
        description="Cache TTL in seconds"
    )
    
    # Background job settings
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_BROKER_URL",
        description="Celery broker URL"
    )
    
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_RESULT_BACKEND",
        description="Celery result backend URL"
    )
    
    # Legal processing settings
    supported_jurisdictions: List[str] = Field(
        default=[
            "federal", "california", "new_york", "texas", "florida",
            "illinois", "pennsylvania", "ohio", "georgia", "north_carolina", "local"
        ],
        description="Supported legal jurisdictions"
    )
    
    supported_document_types: List[str] = Field(
        default=["case", "statute", "article", "gazette", "regulation"],
        description="Supported document types"
    )
    
    authority_levels: dict = Field(
        default={
            "Supreme Court": 10,
            "Federal Circuit": 9,
            "State Supreme Court": 9,
            "Federal District": 8,
            "State Appellate": 8,
            "State Superior": 7,
            "State District": 6,
            "Family Court": 6,
            "Municipal": 4,
            "Local": 3,
            "Administrative": 5
        },
        description="Authority level mappings"
    )
    
    # Processing settings
    chunk_size: int = Field(
        default=1500,
        env="CHUNK_SIZE",
        description="Maximum chunk size in characters"
    )
    
    chunk_overlap: int = Field(
        default=200,
        env="CHUNK_OVERLAP",
        description="Overlap between chunks in characters"
    )
    
    batch_size: int = Field(
        default=10,
        env="BATCH_SIZE",
        description="Batch size for document processing"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(
        default=False,
        env="ENABLE_METRICS",
        description="Enable Prometheus metrics"
    )
    
    metrics_port: int = Field(
        default=9090,
        env="METRICS_PORT",
        description="Port for metrics endpoint"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # Create necessary directories
        Path(_settings.upload_dir).mkdir(exist_ok=True)
        
        # Set Tesseract path if provided
        if _settings.tesseract_cmd and os.path.exists(_settings.tesseract_cmd):
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = _settings.tesseract_cmd
    
    return _settings

# Development configuration
def get_dev_settings() -> Settings:
    """Get development-specific settings"""
    settings = get_settings()
    settings.debug = True
    settings.log_level = "debug"
    return settings

# Production configuration
def get_prod_settings() -> Settings:
    """Get production-specific settings"""
    settings = get_settings()
    settings.debug = False
    settings.log_level = "warning"
    return settings