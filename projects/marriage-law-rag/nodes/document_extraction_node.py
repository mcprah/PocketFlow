#!/usr/bin/env python3
"""
Document Extraction Node - Extracts text and metadata from uploaded documents
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pocketflow import Node
from utils.enhanced_pdf_extractor import EnhancedPDFExtractor, LegalMetadataExtractor, DocumentValidator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentExtractionNode(Node):
    """Extract text and metadata from uploaded documents"""
    
    def __init__(self):
        super().__init__(max_retries=2)
        self.pdf_extractor = EnhancedPDFExtractor(ocr_enabled=True)
        self.metadata_extractor = LegalMetadataExtractor()
        self.validator = DocumentValidator()
    
    def prep(self, shared):
        return shared.get("raw_documents", [])
    
    def exec(self, documents):
        extracted_docs = []
        failed_docs = []
        
        for doc_info in documents:
            try:
                # Extract text from document
                extraction_result = self.pdf_extractor.extract_text(doc_info["path"])
                
                # Extract legal metadata
                legal_metadata = self.metadata_extractor.extract_metadata(
                    extraction_result["text"],
                    extraction_result.get("metadata", {})
                )
                
                # Validate extraction quality
                validation_result = self.validator.validate_document(extraction_result)
                
                # Combine all information
                processed_doc = {
                    "document_id": doc_info.get("document_id", f"doc_{datetime.now().timestamp()}"),
                    "original_path": doc_info["path"],
                    "text": extraction_result["text"],
                    "extraction_metadata": extraction_result,
                    "legal_metadata": legal_metadata,
                    "validation": validation_result,
                    "doc_type": doc_info.get("doc_type", "unknown"),
                    "jurisdiction": doc_info.get("jurisdiction", "unknown"),
                    "authority": doc_info.get("authority"),
                    "date_issued": doc_info.get("date_issued")
                }
                
                if validation_result["is_valid"]:
                    extracted_docs.append(processed_doc)
                    logger.info(f"Successfully extracted document: {doc_info['path']}")
                else:
                    failed_docs.append({
                        "document": processed_doc,
                        "reason": validation_result["issues"]
                    })
                    logger.warning(f"Document validation failed: {doc_info['path']}")
                    
            except Exception as e:
                failed_docs.append({
                    "document": doc_info,
                    "reason": str(e)
                })
                logger.error(f"Failed to extract document {doc_info['path']}: {e}")
        
        return {
            "extracted_documents": extracted_docs,
            "failed_documents": failed_docs
        }
    
    def post(self, shared, prep_result, exec_result):
        shared["extracted_documents"] = exec_result["extracted_documents"]
        shared["failed_documents"] = exec_result["failed_documents"]
        return "chunk" if exec_result["extracted_documents"] else "failed"
    
    def exec_fallback(self, prep_result, exc):
        logger.error(f"DocumentExtractionNode failed: {exc}")
        return {"extracted_documents": [], "failed_documents": []}