#!/usr/bin/env python3
"""
Answer Generation Node - Generates answer using retrieved chunks and OpenAI
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


class AnswerGenerationNode(Node):
    """Generate answer using retrieved chunks and OpenAI"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__(max_retries=3)
        self.model = model
    
    def prep(self, shared):
        return {
            "user_query": shared.get("user_query", ""),
            "search_results": shared.get("search_results", [])
        }
    
    def exec(self, prep_data):
        user_query = prep_data["user_query"]
        search_results = prep_data["search_results"]
        
        if not search_results:
            return {
                "generated_answer": "I couldn't find relevant information to answer your question.",
                "confidence_score": 0.0,
                "citations": []
            }
        
        # Build context from search results
        context_parts = []
        citations = []
        
        for i, result in enumerate(search_results[:5]):  # Use top 5 results
            context_parts.append(f"[Document {i+1}]: {result['text']}")
            citations.append({
                "document_id": result["document_id"],
                "chunk_id": result["chunk_id"],
                "similarity_score": result.get("similarity_score", 0.0),
                "legal_context": result.get("document_metadata", {})
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for answer generation
        prompt = f"""
        You are a legal research assistant specializing in marriage law in Ghana. Based on the following legal documents and the user's question, provide a comprehensive and accurate answer.

        User Question: {user_query}

        Relevant Legal Documents:
        {context}

        Instructions:
        1. Provide a clear, accurate answer based solely on the information in the provided documents
        2. If the documents don't contain sufficient information, clearly state this
        3. Include specific references to relevant legal authorities or cases mentioned
        4. Use proper legal terminology and cite specific sections when available
        5. If there are conflicting interpretations, acknowledge them
        6. Maintain objectivity and avoid giving legal advice

        Answer:
        """
        
        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a legal research assistant specializing in marriage law in Ghana. Provide accurate, objective information based on legal documents."},
                {"role": "user", "content": prompt}
            ],
            # max_tokens=1000,
            temperature=0.8
        )
        
        generated_answer = response.choices[0].message.content.strip()
        
        # Calculate confidence based on search result quality
        avg_similarity = sum(r.get("similarity_score", 0) for r in search_results) / len(search_results)
        confidence_score = min(avg_similarity * 1.2, 1.0)  # Scale and cap at 1.0
        
        return {
            "generated_answer": generated_answer,
            "confidence_score": confidence_score,
            "citations": citations
        }
    
    def post(self, shared, prep_result, exec_result):
        shared["generated_answer"] = exec_result["generated_answer"]
        shared["confidence_score"] = exec_result["confidence_score"]
        shared["citations"] = exec_result["citations"]
        shared["query_results"] = prep_result["search_results"]
        return None  # End of online query flow