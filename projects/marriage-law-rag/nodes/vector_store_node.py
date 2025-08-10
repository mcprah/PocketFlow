#!/usr/bin/env python3
"""
Vector Store Node - Stores document chunks and embeddings in PostgreSQL with pgvector
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pocketflow import AsyncNode
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStoreNode(AsyncNode):
    """Store document chunks and embeddings in PostgreSQL with pgvector"""
    
    def __init__(self):
        super().__init__(max_retries=2)
    
    async def prep_async(self, shared):
        return {
            "chunks": shared.get("embedded_chunks", []),
            "vector_store": shared.get("vector_store")
        }
    
    async def exec_async(self, prep_data):
        chunks = prep_data["chunks"]
        vector_store = prep_data["vector_store"]
        
        if not vector_store:
            raise ValueError("Vector store not available")
        
        stored_count = 0
        for chunk in chunks:
            try:
                # Store chunk in vector database
                await vector_store.store_chunk(chunk)
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store chunk {chunk['chunk_id']}: {e}")
        
        return {"stored_count": stored_count}
    
    async def post_async(self, shared, prep_result, exec_result):
        shared["indexing_result"] = {
            "total_chunks": len(prep_result["chunks"]),
            "stored_chunks": exec_result["stored_count"],
            "timestamp": datetime.now().isoformat()
        }
        return None  # End of offline indexing flow
    
    async def exec_fallback_async(self, prep_result, exc):
        logger.error(f"VectorStoreNode failed: {exc}")
        return {"stored_count": 0}