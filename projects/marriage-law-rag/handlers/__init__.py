#!/usr/bin/env python3
"""
API handlers module for FastAPI endpoints
"""

from .document_handler import upload_document
from .query_handler import query_documents
from .health_handler import health_check

__all__ = ['upload_document', 'query_documents', 'health_check']