#!/usr/bin/env python3
"""
Embedding Node - Generates embeddings for document chunks
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pocketflow import BatchNode
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

# Initialize OpenAI client lazily
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("Please set a valid OPENAI_API_KEY in your .env file")
        openai_client = OpenAI(api_key=api_key)
    return openai_client


class EmbeddingNode(BatchNode):
    """Generate embeddings for document chunks"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__(max_retries=3)
        self.model = model
    
    def prep(self, shared):
        return shared.get("document_chunks", [])
    
    def exec(self, chunk):
        try:
            # Get embedding for single chunk
            client = get_openai_client()
            response = client.embeddings.create(
                model=self.model,
                input=[chunk["text"]]
            )
            
            # Add embedding to chunk
            chunk["embedding"] = response.data[0].embedding
            return chunk
            
        except Exception as e:
            logger.error(f"Embedding generation failed for chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            # Return chunk without embedding as fallback
            return chunk
    
    def post(self, shared, prep_result, exec_results):
        # exec_results is a list of chunks with embeddings
        shared["embedded_chunks"] = exec_results
        return "store"
    
    def exec_fallback(self, prep_result, exc):
        logger.error(f"EmbeddingNode failed: {exc}")
        # Return chunk without embedding
        return prep_result