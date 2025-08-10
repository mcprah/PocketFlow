#!/usr/bin/env python3
"""
Query Embedding Node - Generates embedding for user query
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pocketflow import Node
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


class QueryEmbeddingNode(Node):
    """Generate embedding for user query"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__(max_retries=3)
        self.model = model
    
    def prep(self, shared):
        return shared.get("user_query", "")
    
    def exec(self, query_text):
        if not query_text:
            return {"query_embedding": None}
        
        client = get_openai_client()
        response = client.embeddings.create(
            model=self.model,
            input=[query_text]
        )
        
        return {"query_embedding": response.data[0].embedding}
    
    def post(self, shared, prep_result, exec_result):
        shared["query_embedding"] = exec_result["query_embedding"]
        return "search" if exec_result["query_embedding"] else "failed"
    
    def exec_fallback(self, prep_result, exc):
        logger.error(f"QueryEmbeddingNode failed: {exc}")
        return {"query_embedding": None}