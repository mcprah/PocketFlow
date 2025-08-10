#!/usr/bin/env python3
"""
Marriage Law RAG System - Implementation Example with PostgreSQL + pgvector
This demonstrates how to implement the key utilities and nodes for the system.
"""

from pocketflow import Node, Flow, BatchNode, AsyncParallelBatchNode
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from openai import OpenAI
from enhanced_pdf_utilities import EnhancedPDFExtractor, LegalMetadataExtractor, DocumentValidator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class PostgresVectorStore:
    """PostgreSQL + pgvector operations for legal documents"""
    
    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(self.conn_str, cursor_factory=RealDictCursor)
        return self.conn
    
    def create_tables(self):
        """Create the legal document tables with pgvector"""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create tables (schema from design document)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS legal_documents (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    doc_type VARCHAR(50) NOT NULL,
                    jurisdiction VARCHAR(100) NOT NULL,
                    title TEXT,
                    authority VARCHAR(200),
                    date_issued DATE,
                    authority_level INTEGER CHECK (authority_level >= 1 AND authority_level <= 10),
                    raw_text TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES legal_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section_title VARCHAR(500),
                    embedding vector(1536),
                    citations TEXT[],
                    legal_concepts TEXT[],
                    chunk_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create vector similarity index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
            
            self.conn.commit()
    
    def insert_document(self, doc_data: Dict[str, Any]) -> int:
        """Insert a legal document and return its ID"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO legal_documents 
                (file_path, doc_type, jurisdiction, title, authority, date_issued, authority_level, raw_text, metadata)
                VALUES (%(file_path)s, %(doc_type)s, %(jurisdiction)s, %(title)s, 
                        %(authority)s, %(date_issued)s, %(authority_level)s, %(raw_text)s, %(metadata)s)
                RETURNING id;
            """, doc_data)
            
            doc_id = cur.fetchone()['id']
            self.conn.commit()
            return doc_id
    
    def insert_chunk(self, chunk_data: Dict[str, Any]) -> int:
        """Insert a document chunk with embedding"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO document_chunks 
                (document_id, chunk_index, text, section_title, embedding, citations, legal_concepts, chunk_metadata)
                VALUES (%(document_id)s, %(chunk_index)s, %(text)s, %(section_title)s, 
                        %(embedding)s, %(citations)s, %(legal_concepts)s, %(chunk_metadata)s)
                RETURNING id;
            """, chunk_data)
            
            chunk_id = cur.fetchone()['id']
            self.conn.commit()
            return chunk_id
    
    def similarity_search(self, query_embedding: List[float], 
                         jurisdiction_filter: List[str] = None,
                         doc_type_filter: List[str] = None,
                         limit: int = 5) -> List[Dict]:
        """Perform vector similarity search with metadata filtering"""
        
        # Build dynamic WHERE clause based on filters
        where_conditions = []
        params = {'query_embedding': query_embedding, 'limit': limit}
        
        if jurisdiction_filter:
            where_conditions.append("chunk_metadata->>'jurisdiction' = ANY(%(jurisdiction_filter)s)")
            params['jurisdiction_filter'] = jurisdiction_filter
            
        if doc_type_filter:
            where_conditions.append("chunk_metadata->>'doc_type' = ANY(%(doc_type_filter)s)")
            params['doc_type_filter'] = doc_type_filter
        
        where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
            SELECT 
                dc.id,
                dc.text,
                dc.section_title,
                dc.citations,
                dc.legal_concepts,
                dc.chunk_metadata,
                ld.title,
                ld.authority,
                ld.authority_level,
                1 - (dc.embedding <=> %(query_embedding)s) as similarity
            FROM document_chunks dc
            JOIN legal_documents ld ON dc.document_id = ld.id
            WHERE 1=1 {where_clause}
            ORDER BY dc.embedding <=> %(query_embedding)s
            LIMIT %(limit)s;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


class EmbeddingService:
    """OpenAI embedding service for legal text"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small-ada-002") -> List[float]:
        """Get embedding for a text chunk"""
        response = self.client.embeddings.create(
            model=model,
            input=text.replace("\n", " ")
        )
        return response.data[0].embedding


class LegalChunker:
    """Smart chunking for legal documents with enhanced section detection"""
    
    @staticmethod
    def chunk_by_type(text: str, doc_type: str, legal_metadata: Dict = None, max_chunk_size: int = 1500) -> List[Dict]:
        """
        Chunk text based on document type with legal-aware splitting.
        
        Args:
            text: Document text to chunk
            doc_type: Type of legal document (case, statute, article, etc.)
            legal_metadata: Metadata extracted from document
            max_chunk_size: Maximum characters per chunk
        """
        chunks = []
        
        if doc_type == "case":
            chunks = LegalChunker._chunk_case_law(text, max_chunk_size)
        elif doc_type == "statute":
            chunks = LegalChunker._chunk_statute(text, max_chunk_size)
        elif doc_type == "article":
            chunks = LegalChunker._chunk_article(text, max_chunk_size)
        else:
            chunks = LegalChunker._chunk_generic(text, max_chunk_size)
        
        # Add legal metadata to each chunk
        if legal_metadata:
            for chunk in chunks:
                chunk.update({
                    'jurisdiction': legal_metadata.get('jurisdiction'),
                    'authority': legal_metadata.get('authority'),
                    'authority_level': legal_metadata.get('authority_level', 5),
                    'legal_concepts': legal_metadata.get('legal_concepts', []),
                    'citations': legal_metadata.get('citations', [])
                })
        
        return chunks
    
    @staticmethod
    def _chunk_case_law(text: str, max_chunk_size: int) -> List[Dict]:
        """Chunk case law by legal sections and procedural elements"""
        import re
        
        # Identify major case sections
        section_patterns = [
            r'\n\s*(?:FACTS?|PROCEDURAL HISTORY|BACKGROUND)\s*\n',
            r'\n\s*(?:ISSUE|QUESTION PRESENTED)\s*\n',
            r'\n\s*(?:HOLDING|DECISION|RULING)\s*\n',
            r'\n\s*(?:REASONING|ANALYSIS|DISCUSSION)\s*\n',
            r'\n\s*(?:CONCLUSION|DISPOSITION)\s*\n',
            r'\n\s*(?:DISSENT|CONCUR[A-Z]*)\s*\n'
        ]
        
        # Split by major sections first
        sections = []
        current_pos = 0
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                if match.start() > current_pos:
                    sections.append(text[current_pos:match.start()])
                current_pos = match.start()
        
        # Add remaining text
        if current_pos < len(text):
            sections.append(text[current_pos:])
        
        # Further chunk if sections are too large
        chunks = []
        chunk_index = 0
        
        for section in sections:
            if len(section) <= max_chunk_size:
                chunks.append({
                    "text": section.strip(),
                    "chunk_index": chunk_index,
                    "section_type": "case_section"
                })
                chunk_index += 1
            else:
                # Split large sections by paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk + para) > max_chunk_size and current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "chunk_index": chunk_index,
                            "section_type": "case_section"
                        })
                        chunk_index += 1
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "section_type": "case_section"
                    })
                    chunk_index += 1
        
        return chunks
    
    @staticmethod
    def _chunk_statute(text: str, max_chunk_size: int) -> List[Dict]:
        """Chunk statutes by sections and subsections"""
        import re
        
        # Look for section patterns
        section_pattern = r'\n\s*(?:§|Section|Sec\.?)\s*(\d+(?:\.\d+)*)'
        sections = re.split(section_pattern, text)
        
        chunks = []
        chunk_index = 0
        
        # Process sections
        for i in range(0, len(sections), 2):  # Skip section numbers
            section_text = sections[i]
            if not section_text.strip():
                continue
                
            if len(section_text) <= max_chunk_size:
                chunks.append({
                    "text": section_text.strip(),
                    "chunk_index": chunk_index,
                    "section_type": "statute_section"
                })
                chunk_index += 1
            else:
                # Split by subsections or paragraphs
                subsections = re.split(r'\n\s*\([a-z]\)', section_text)
                current_chunk = ""
                
                for subsection in subsections:
                    if len(current_chunk + subsection) > max_chunk_size and current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "chunk_index": chunk_index,
                            "section_type": "statute_subsection"
                        })
                        chunk_index += 1
                        current_chunk = subsection
                    else:
                        current_chunk += "\n" + subsection if current_chunk else subsection
                
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "section_type": "statute_subsection"
                    })
                    chunk_index += 1
        
        return chunks
    
    @staticmethod
    def _chunk_article(text: str, max_chunk_size: int) -> List[Dict]:
        """Chunk academic articles by sections and paragraphs"""
        import re
        
        # Look for article sections (Roman numerals, letters, numbers)
        section_patterns = [
            r'\n\s*[IVX]+\.\s*[A-Z].*\n',  # Roman numerals
            r'\n\s*[A-Z]\.\s*[A-Z].*\n',   # Letter sections
            r'\n\s*\d+\.\s*[A-Z].*\n'      # Numbered sections
        ]
        
        # Try to split by sections first
        sections = []
        for pattern in section_patterns:
            if re.search(pattern, text):
                sections = re.split(pattern, text)
                break
        
        if not sections:
            # Fallback to paragraph splitting
            sections = text.split('\n\n')
        
        chunks = []
        chunk_index = 0
        current_chunk = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            if len(current_chunk + section) > max_chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "section_type": "article_section"
                })
                chunk_index += 1
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "section_type": "article_section"
            })
        
        return chunks
    
    @staticmethod
    def _chunk_generic(text: str, max_chunk_size: int) -> List[Dict]:
        """Generic chunking for other document types"""
        chunks = []
        chunk_index = 0
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk + para) > max_chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "section_type": "generic_section"
                })
                chunk_index += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "section_type": "generic_section"
            })
        
        return chunks


# =============================================================================
# POCKETFLOW NODES
# =============================================================================

class ParseDocumentsNode(BatchNode):
    """Enhanced PDF/DOC parsing with multiple extraction strategies and OCR"""
    
    def __init__(self, ocr_enabled: bool = True):
        super().__init__()
        self.pdf_extractor = EnhancedPDFExtractor(ocr_enabled=ocr_enabled)
        self.metadata_extractor = LegalMetadataExtractor()
        self.validator = DocumentValidator()
    
    def prep(self, shared):
        return shared["raw_documents"]
    
    def exec(self, doc_info):
        """Extract text and metadata from legal documents"""
        try:
            file_path = doc_info["path"]
            
            # Extract text using multi-strategy approach
            extraction_result = self.pdf_extractor.extract_text(file_path)
            
            # Extract legal metadata
            legal_metadata = self.metadata_extractor.extract_metadata(
                extraction_result['text'],
                extraction_result.get('metadata', {})
            )
            
            # Validate document quality
            validation_result = self.validator.validate_document(extraction_result)
            
            return {
                **doc_info,
                "raw_text": extraction_result['text'],
                "extraction_method": extraction_result['extraction_method'],
                "quality_score": extraction_result['quality_score'],
                "legal_metadata": legal_metadata,
                "validation": validation_result,
                "page_count": extraction_result['page_count'],
                "status": "parsed" if validation_result['is_valid'] else "warning"
            }
            
        except Exception as e:
            return {
                **doc_info,
                "error": str(e),
                "status": "failed"
            }
    
    def post(self, shared, prep_res, exec_res_list):
        shared["parsed_documents"] = exec_res_list
        
        successful = [doc for doc in exec_res_list if doc.get("status") == "parsed"]
        warnings = [doc for doc in exec_res_list if doc.get("status") == "warning"]
        failed = [doc for doc in exec_res_list if doc.get("status") == "failed"]
        
        print(f"✅ Successfully parsed: {len(successful)} documents")
        if warnings:
            print(f"⚠️  Parsed with warnings: {len(warnings)} documents")
        if failed:
            print(f"❌ Failed to parse: {len(failed)} documents")
        
        # Log extraction methods used
        methods = {}
        for doc in successful + warnings:
            method = doc.get("extraction_method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        
        print(f"📊 Extraction methods used: {methods}")
        
        # Log average quality scores
        quality_scores = [doc.get("quality_score", 0) for doc in successful + warnings]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"📈 Average extraction quality: {avg_quality:.2f}")


class ChunkDocumentsNode(BatchNode):
    """Enhanced legal document chunking with metadata integration"""
    
    def prep(self, shared):
        # Only process successfully parsed documents
        parsed_docs = [doc for doc in shared["parsed_documents"] 
                      if doc.get("status") in ["parsed", "warning"]]
        return parsed_docs
    
    def exec(self, doc_info):
        """Create semantically meaningful chunks from legal documents"""
        if doc_info.get("status") not in ["parsed", "warning"]:
            return []
        
        legal_metadata = doc_info.get("legal_metadata", {})
        doc_type = legal_metadata.get("document_type", "unknown")
        
        # Use enhanced chunker with legal metadata
        chunker = LegalChunker()
        chunks = chunker.chunk_by_type(
            doc_info["raw_text"], 
            doc_type,
            legal_metadata
        )
        
        # Add document-level metadata to each chunk
        for chunk in chunks:
            chunk.update({
                "document_id": None,  # Will be set after DB insert
                "doc_type": doc_type,
                "file_path": doc_info["path"],
                "extraction_method": doc_info.get("extraction_method"),
                "extraction_quality": doc_info.get("quality_score", 0.0),
                "page_count": doc_info.get("page_count", 0),
                
                # Legal metadata from extraction
                "case_number": legal_metadata.get("case_number"),
                "date_issued": legal_metadata.get("date_issued"),
                "title": legal_metadata.get("title"),
                
                # Add validation scores for chunk quality assessment
                "document_validation_score": doc_info.get("validation", {}).get("quality_score", 0.0)
            })
        
        return chunks
    
    def post(self, shared, prep_res, exec_res_list):
        # Flatten list of chunk lists
        all_chunks = []
        chunk_stats = {"case": 0, "statute": 0, "article": 0, "unknown": 0}
        
        for chunk_list in exec_res_list:
            all_chunks.extend(chunk_list)
            for chunk in chunk_list:
                doc_type = chunk.get("doc_type", "unknown")
                chunk_stats[doc_type] = chunk_stats.get(doc_type, 0) + 1
        
        shared["document_chunks"] = all_chunks
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(prep_res)} documents")
        print(f"📊 Chunk distribution: {chunk_stats}")
        
        # Calculate average chunk sizes by document type
        type_sizes = {}
        for chunk in all_chunks:
            doc_type = chunk.get("doc_type", "unknown")
            if doc_type not in type_sizes:
                type_sizes[doc_type] = []
            type_sizes[doc_type].append(len(chunk.get("text", "")))
        
        for doc_type, sizes in type_sizes.items():
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                print(f"📏 Average {doc_type} chunk size: {avg_size:.0f} characters")


class EmbedChunksNode(AsyncParallelBatchNode):
    """Generate embeddings for document chunks in parallel"""
    
    def prep(self, shared):
        return shared["document_chunks"]
    
    async def exec_async(self, chunk):
        # In real implementation, this would be async
        embedding_service = EmbeddingService(os.getenv("OPENAI_API_KEY"))
        embedding = embedding_service.get_embedding(chunk["text"])
        
        chunk["embedding"] = embedding
        return chunk
    
    async def post_async(self, shared, prep_res, exec_res_list):
        shared["embedded_chunks"] = exec_res_list
        print(f"✅ Generated {len(exec_res_list)} embeddings")


class StoreInDatabaseNode(Node):
    """Store documents and chunks in PostgreSQL with pgvector"""
    
    def prep(self, shared):
        return shared["embedded_chunks"]
    
    def exec(self, embedded_chunks):
        vector_store = PostgresVectorStore(shared["db_connection_string"])
        vector_store.connect()
        vector_store.create_tables()
        
        # Group chunks by document
        docs_by_path = {}
        for chunk in embedded_chunks:
            path = chunk["file_path"]
            if path not in docs_by_path:
                docs_by_path[path] = {"chunks": [], "doc_info": chunk}
            docs_by_path[path]["chunks"].append(chunk)
        
        chunk_count = 0
        for path, doc_data in docs_by_path.items():
            # Insert document first
            doc_info = doc_data["doc_info"]
            doc_id = vector_store.insert_document({
                "file_path": path,
                "doc_type": doc_info["doc_type"],
                "jurisdiction": doc_info["jurisdiction"],
                "title": path.split("/")[-1],  # Simplified
                "authority": doc_info.get("authority"),
                "date_issued": None,  # Would extract from document
                "authority_level": doc_info.get("authority_level", 5),
                "raw_text": None,  # Could store if needed
                "metadata": {}
            })
            
            # Insert chunks
            for chunk in doc_data["chunks"]:
                vector_store.insert_chunk({
                    "document_id": doc_id,
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "section_title": None,
                    "embedding": chunk["embedding"],
                    "citations": [],  # Would extract citations
                    "legal_concepts": [],  # Would extract concepts
                    "chunk_metadata": {
                        "doc_type": chunk["doc_type"],
                        "jurisdiction": chunk["jurisdiction"],
                        "authority_level": chunk["authority_level"]
                    }
                })
                chunk_count += 1
        
        return {"stored_chunks": chunk_count, "vector_store": vector_store}
    
    def post(self, shared, prep_res, exec_res):
        shared["vector_store"] = exec_res["vector_store"]
        print(f"✅ Stored {exec_res['stored_chunks']} chunks in database")


class EmbedQueryNode(Node):
    """Embed the user query for similarity search"""
    
    def prep(self, shared):
        return shared["user_query"]
    
    def exec(self, query):
        embedding_service = EmbeddingService(os.getenv("OPENAI_API_KEY"))
        return embedding_service.get_embedding(query)
    
    def post(self, shared, prep_res, query_embedding):
        shared["query_embedding"] = query_embedding
        print(f"🔍 Embedded query: {prep_res[:50]}...")


class RetrieveRelevantNode(Node):
    """Retrieve relevant chunks using vector similarity"""
    
    def prep(self, shared):
        return {
            "query_embedding": shared["query_embedding"],
            "jurisdiction_filter": shared.get("jurisdiction_filter"),
            "doc_type_filter": shared.get("doc_type_filter"),
            "vector_store": shared["vector_store"]
        }
    
    def exec(self, search_params):
        vector_store = search_params["vector_store"]
        
        results = vector_store.similarity_search(
            query_embedding=search_params["query_embedding"],
            jurisdiction_filter=search_params["jurisdiction_filter"],
            doc_type_filter=search_params["doc_type_filter"],
            limit=5
        )
        
        return results
    
    def post(self, shared, prep_res, search_results):
        shared["retrieved_chunks"] = search_results
        print(f"📄 Retrieved {len(search_results)} relevant chunks")
        
        # Check if we have sufficient results
        if len(search_results) >= 3:
            return "sufficient"
        else:
            return "expand"


class GenerateAnswerNode(Node):
    """Generate answer using LLM with retrieved context"""
    
    def prep(self, shared):
        return {
            "query": shared["user_query"],
            "context_chunks": shared["retrieved_chunks"][:3]  # Use top 3
        }
    
    def exec(self, inputs):
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(inputs["context_chunks"]):
            context_parts.append(f"""
Document {i+1} ({chunk['chunk_metadata']['doc_type'].title()}):
Authority: {chunk['authority']} (Level: {chunk['authority_level']})
Jurisdiction: {chunk['chunk_metadata']['jurisdiction']}
Content: {chunk['text']}
""")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are a legal research assistant specializing in marriage law. 
Based on the provided legal documents, answer the user's question accurately and cite your sources.

LEGAL CONTEXT:
{context}

QUESTION: {inputs['query']}

INSTRUCTIONS:
1. Provide a clear, accurate answer based only on the provided legal sources
2. Include relevant citations and authority levels
3. Note any jurisdictional limitations
4. If the sources don't fully address the question, state this clearly
5. Add appropriate legal disclaimers

ANSWER:"""

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1  # Lower temperature for more factual responses
        )
        
        return response.choices[0].message.content
    
    def post(self, shared, prep_res, generated_answer):
        shared["generated_answer"] = generated_answer
        print("🤖 Generated legal answer")


# =============================================================================
# FLOW CREATION
# =============================================================================

def create_offline_indexing_flow():
    """Create the offline document indexing flow"""
    parse_node = ParseDocumentsNode()
    chunk_node = ChunkDocumentsNode()
    embed_node = EmbedChunksNode()
    store_node = StoreInDatabaseNode()
    
    # Connect nodes in sequence
    parse_node >> chunk_node >> embed_node >> store_node
    
    return Flow(start=parse_node)


def create_online_query_flow():
    """Create the online query processing flow"""
    embed_query_node = EmbedQueryNode()
    retrieve_node = RetrieveRelevantNode()
    generate_node = GenerateAnswerNode()
    
    # Connect with conditional branching
    embed_query_node >> retrieve_node
    
    # If sufficient results, generate answer
    retrieve_node - "sufficient" >> generate_node
    
    # If not sufficient, could add expansion logic here
    # retrieve_node - "expand" >> expand_search_node >> generate_node
    
    return Flow(start=embed_query_node)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Set up database connection
    DB_CONNECTION = "postgresql://user:password@localhost:5432/marriage_law_db"
    
    # Example offline indexing
    offline_shared = {
        "db_connection_string": DB_CONNECTION,
        "raw_documents": [
            {
                "path": "data/marriage_act_2023.txt",
                "doc_type": "statute",
                "jurisdiction": "federal",
                "authority": "Congress",
                "authority_level": 10
            },
            {
                "path": "data/smith_vs_jones.txt", 
                "doc_type": "case",
                "jurisdiction": "california",
                "authority": "CA Supreme Court",
                "authority_level": 9
            }
        ]
    }
    
    print("🔄 Starting offline indexing...")
    offline_flow = create_offline_indexing_flow()
    offline_flow.run(offline_shared)
    
    # Example online query
    online_shared = {
        "user_query": "What are the legal requirements for same-sex marriage in California?",
        "jurisdiction_filter": ["federal", "california"],
        "doc_type_filter": ["statute", "case"],
        "vector_store": offline_shared["vector_store"]
    }
    
    print("\n🔄 Processing user query...")
    online_flow = create_online_query_flow()
    online_flow.run(online_shared)
    
    print(f"\n📋 FINAL ANSWER:\n{online_shared['generated_answer']}")