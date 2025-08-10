#!/usr/bin/env python3
"""
Marriage Law RAG System - PocketFlow Workflow Definitions
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pocketflow import AsyncFlow, BatchFlow, BatchNode
from nodes import (
    DocumentExtractionNode,
    DocumentChunkingNode,
    EmbeddingNode,
    VectorStoreNode,
    QueryEmbeddingNode,
    VectorSearchNode,
    AnswerGenerationNode
)


def create_offline_indexing_flow() -> AsyncFlow:
    """
    Create the offline document indexing flow.
    Processes uploaded documents and stores them in the vector database.
    """
    
    # Create nodes
    extract_node = DocumentExtractionNode()
    chunk_node = DocumentChunkingNode()
    embed_node = EmbeddingNode()
    store_node = VectorStoreNode()
    
    # Create flow connections
    extract_node >> chunk_node >> embed_node >> store_node
    
    # Handle conditional transitions
    extract_node - "chunk" >> chunk_node  # Successful extraction goes to chunking
    chunk_node - "embed" >> embed_node    # Chunking goes to embedding
    embed_node - "store" >> store_node    # Embedding goes to storage
    extract_node - "failed" >> store_node  # Still try to store what we have
    
    # Create and return flow
    return AsyncFlow(start=extract_node)


def create_online_query_flow() -> AsyncFlow:
    """
    Create the online query processing flow.
    Handles user queries and generates answers using RAG.
    """
    
    # Create nodes
    query_embed_node = QueryEmbeddingNode()
    search_node = VectorSearchNode()
    generate_node = AnswerGenerationNode()
    
    # Create flow connections
    query_embed_node >> search_node >> generate_node
    
    # Handle conditional transitions
    query_embed_node - "search" >> search_node    # Successful embedding goes to search
    search_node - "generate" >> generate_node     # Search results go to generation
    search_node - "no_results" >> generate_node   # Generate "no results" answer
    query_embed_node - "failed" >> generate_node  # Direct to generation if embedding fails
    
    # Create and return flow  
    return AsyncFlow(start=query_embed_node)


def create_bulk_indexing_flow(max_workers: dict = None) -> BatchFlow:
    """
    Create the bulk document indexing flow for processing large document collections.
    Uses BatchFlow and BatchNode for parallel processing with configurable worker limits.
    
    Args:
        max_workers: Dictionary specifying worker limits for each stage
                    e.g., {"extract": 4, "chunk": 8, "embed": 2, "store": 4}
    
    Returns:
        BatchFlow configured for bulk document processing
    """
    
    # Default worker configuration optimized for typical hardware
    if max_workers is None:
        max_workers = {
            "extract": 4,    # CPU intensive PDF extraction
            "chunk": 8,      # Fast text processing
            "embed": 2,      # Limited by OpenAI API rate limits  
            "store": 4       # Database connection pool size
        }
    
    # Create batch nodes with worker limits
    extract_batch = BatchNode(
        DocumentExtractionNode(), 
        max_workers=max_workers.get("extract", 4)
    )
    chunk_batch = BatchNode(
        DocumentChunkingNode(), 
        max_workers=max_workers.get("chunk", 8)
    )
    embed_batch = BatchNode(
        EmbeddingNode(), 
        max_workers=max_workers.get("embed", 2)
    )
    store_batch = BatchNode(
        VectorStoreNode(), 
        max_workers=max_workers.get("store", 4)
    )
    
    # Create flow connections - same pattern as single document flow
    extract_batch >> chunk_batch >> embed_batch >> store_batch
    
    # Handle conditional transitions for batch processing
    extract_batch - "chunk" >> chunk_batch      # Successful extractions go to chunking
    chunk_batch - "embed" >> embed_batch        # Chunked documents go to embedding
    embed_batch - "store" >> store_batch        # Embedded chunks go to storage
    extract_batch - "failed" >> store_batch     # Still try to store partially processed docs
    
    # Create and return batch flow
    return BatchFlow(start=extract_batch)