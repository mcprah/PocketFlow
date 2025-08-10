#!/usr/bin/env python3
"""
Vector Search Node - Search for relevant document chunks using vector similarity
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pocketflow import AsyncNode
import logging

logger = logging.getLogger(__name__)


class VectorSearchNode(AsyncNode):
    """Search for relevant document chunks using vector similarity"""
    
    def __init__(self, max_results: int = 10):
        super().__init__(max_retries=2)
        self.max_results = max_results
    
    async def prep_async(self, shared):
        return {
            "query_embedding": shared.get("query_embedding"),
            "vector_store": shared.get("vector_store"),
            "jurisdiction_filter": shared.get("jurisdiction_filter"),
            "doc_type_filter": shared.get("doc_type_filter"),
            "authority_level_min": shared.get("authority_level_min"),
            "max_results": shared.get("max_results", self.max_results)
        }
    
    async def exec_async(self, search_params):
        vector_store = search_params["vector_store"]
        query_embedding = search_params["query_embedding"]
        
        if not vector_store or not query_embedding:
            return {"search_results": []}
        
        # Perform vector similarity search
        results = await vector_store.similarity_search(
            query_embedding=query_embedding,
            limit=search_params["max_results"],
            jurisdiction_filter=search_params.get("jurisdiction_filter"),
            doc_type_filter=search_params.get("doc_type_filter"),
            authority_level_min=search_params.get("authority_level_min")
        )
        
        return {"search_results": results}
    
    async def post_async(self, shared, prep_result, exec_result):
        shared["search_results"] = exec_result["search_results"]
        return "generate" if exec_result["search_results"] else "no_results"
    
    async def exec_fallback_async(self, prep_result, exc):
        logger.error(f"VectorSearchNode failed: {exc}")
        return {"search_results": []}