#!/usr/bin/env python3
"""
Marriage Law RAG System - PocketFlow Workflow Definitions
"""

import sys
import os
# Add the parent directory to Python path to access PocketFlow
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pocketflow import Flow, Node, BatchNode, AsyncNode, AsyncFlow, AsyncBatchNode
from utils.enhanced_pdf_extractor import EnhancedPDFExtractor, LegalMetadataExtractor, DocumentValidator
from utils.postgres_vector_store import PostgresVectorStore
from openai import OpenAI
import logging
import os
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

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


class EmbeddingNode(BatchNode):
    """Generate embeddings for document chunks"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
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


class QueryEmbeddingNode(Node):
    """Generate embedding for user query"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
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