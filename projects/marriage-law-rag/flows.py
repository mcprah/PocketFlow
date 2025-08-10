#!/usr/bin/env python3
"""
Marriage Law RAG System - PocketFlow Workflow Definitions
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pocketflow import AsyncFlow
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