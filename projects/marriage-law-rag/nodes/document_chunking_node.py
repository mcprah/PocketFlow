#!/usr/bin/env python3
"""
Document Chunking Node - Splits documents into chunks for vector storage
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pocketflow import Node
import logging
from typing import List

logger = logging.getLogger(__name__)


class DocumentChunkingNode(Node):
    """Split documents into chunks for vector storage"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(max_retries=1)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def prep(self, shared):
        return shared.get("extracted_documents", [])
    
    def exec(self, documents):
        all_chunks = []
        
        for doc in documents:
            text = doc["text"]
            chunks = self._split_text(text, self.chunk_size, self.chunk_overlap)
            
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "chunk_id": f"{doc['document_id']}_chunk_{i}",
                    "document_id": doc["document_id"],
                    "text": chunk_text,
                    "chunk_index": i,
                    "document_metadata": doc["legal_metadata"],
                    "extraction_metadata": doc["extraction_metadata"]
                }
                all_chunks.append(chunk)
        
        return {"chunks": all_chunks}
    
    def post(self, shared, prep_result, exec_result):
        shared["document_chunks"] = exec_result["chunks"]
        return "embed"
    
    def exec_fallback(self, prep_result, exc):
        logger.error(f"DocumentChunkingNode failed: {exc}")
        return {"chunks": []}
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to split at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks