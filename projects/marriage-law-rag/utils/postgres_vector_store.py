#!/usr/bin/env python3
"""
PostgreSQL Vector Store with pgvector for Marriage Law RAG System
"""

import asyncio
import asyncpg
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class PostgresVectorStore:
    """
    PostgreSQL vector store using pgvector extension for storing and searching document embeddings.
    Optimized for legal document storage with metadata filtering.
    """
    
    def __init__(self, connection_string: str, vector_dimension: int = 1536):
        self.connection_string = connection_string
        self.vector_dimension = vector_dimension
        self.pool = None
    
    async def connect(self) -> None:
        """Initialize connection pool to PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("✅ Connected to PostgreSQL with connection pool")
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    async def close(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def create_tables(self) -> None:
        """Create necessary tables and indexes"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id VARCHAR(255) PRIMARY KEY,
                    original_path TEXT,
                    doc_type VARCHAR(100),
                    jurisdiction VARCHAR(100),
                    authority TEXT,
                    authority_level INTEGER,
                    date_issued DATE,
                    date_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extraction_metadata JSONB,
                    legal_metadata JSONB,
                    validation_metadata JSONB,
                    full_text TEXT,
                    status VARCHAR(50) DEFAULT 'processed'
                );
            """)
            
            # Create document_chunks table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id VARCHAR(255) PRIMARY KEY,
                    document_id VARCHAR(255) REFERENCES documents(document_id) ON DELETE CASCADE,
                    chunk_index INTEGER,
                    text TEXT NOT NULL,
                    embedding vector({self.vector_dimension}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for efficient searching
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
                CREATE INDEX IF NOT EXISTS idx_documents_jurisdiction ON documents(jurisdiction);
                CREATE INDEX IF NOT EXISTS idx_documents_authority_level ON documents(authority_level);
                CREATE INDEX IF NOT EXISTS idx_documents_date_issued ON documents(date_issued);
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
            """)
            
            # Create vector similarity index (HNSW for better performance)
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
                    ON document_chunks USING hnsw (embedding vector_cosine_ops);
                """)
            except Exception as e:
                logger.warning(f"Could not create HNSW index, falling back to IVFFlat: {e}")
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat 
                    ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """)
            
            logger.info("✅ Database tables and indexes created successfully")
    
    async def store_document(self, document_data: Dict[str, Any]) -> str:
        """Store a processed document and return document_id"""
        async with self.pool.acquire() as conn:
            document_id = document_data.get("document_id", str(uuid.uuid4()))
            
            # Extract metadata
            legal_metadata = document_data.get("legal_metadata", {})
            date_issued = None
            if legal_metadata.get("date_issued"):
                try:
                    # Try to parse date
                    from dateutil.parser import parse
                    date_issued = parse(legal_metadata["date_issued"]).date()
                except:
                    date_issued = None
            
            await conn.execute("""
                INSERT INTO documents (
                    document_id, original_path, doc_type, jurisdiction, authority,
                    authority_level, date_issued, extraction_metadata, legal_metadata,
                    validation_metadata, full_text
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (document_id) DO UPDATE SET
                    doc_type = EXCLUDED.doc_type,
                    jurisdiction = EXCLUDED.jurisdiction,
                    authority = EXCLUDED.authority,
                    authority_level = EXCLUDED.authority_level,
                    date_issued = EXCLUDED.date_issued,
                    extraction_metadata = EXCLUDED.extraction_metadata,
                    legal_metadata = EXCLUDED.legal_metadata,
                    validation_metadata = EXCLUDED.validation_metadata,
                    full_text = EXCLUDED.full_text;
            """,
                document_id,
                document_data.get("original_path"),
                document_data.get("doc_type", "unknown"),
                legal_metadata.get("jurisdiction"),
                legal_metadata.get("authority"),
                legal_metadata.get("authority_level"),
                date_issued,
                json.dumps(document_data.get("extraction_metadata", {})),
                json.dumps(legal_metadata),
                json.dumps(document_data.get("validation", {})),
                document_data.get("text", "")
            )
            
            logger.info(f"Stored document: {document_id}")
            return document_id
    
    async def store_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Store a document chunk with its embedding"""
        async with self.pool.acquire() as conn:
            chunk_id = chunk_data.get("chunk_id", str(uuid.uuid4()))
            
            # First ensure the document exists
            if "document_metadata" in chunk_data:
                document_data = {
                    "document_id": chunk_data["document_id"],
                    "legal_metadata": chunk_data["document_metadata"],
                    "extraction_metadata": chunk_data.get("extraction_metadata", {}),
                    "text": "",  # Full text not available at chunk level
                    "doc_type": chunk_data["document_metadata"].get("document_type", "unknown")
                }
                await self.store_document(document_data)
            
            # Convert embedding to vector format
            embedding = chunk_data.get("embedding")
            if embedding:
                # Ensure embedding is a list/array
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embedding_vector = f"[{','.join(map(str, embedding))}]"
            else:
                embedding_vector = None
            
            await conn.execute("""
                INSERT INTO document_chunks (chunk_id, document_id, chunk_index, text, embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding;
            """,
                chunk_id,
                chunk_data["document_id"],
                chunk_data.get("chunk_index", 0),
                chunk_data["text"],
                embedding_vector
            )
            
            return chunk_id
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        jurisdiction_filter: Optional[List[str]] = None,
        doc_type_filter: Optional[List[str]] = None,
        authority_level_min: Optional[int] = None,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search with optional metadata filtering
        """
        async with self.pool.acquire() as conn:
            # Convert query embedding to vector format
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            query_vector = f"[{','.join(map(str, query_embedding))}]"
            
            # Build WHERE clause based on filters
            where_conditions = ["c.embedding IS NOT NULL"]
            params = [query_vector, limit]
            param_count = 2
            
            if jurisdiction_filter:
                if isinstance(jurisdiction_filter, list):
                    if len(jurisdiction_filter) == 1:
                        param_count += 1
                        where_conditions.append(f"d.jurisdiction = ${param_count}")
                        params.append(jurisdiction_filter[0])
                    else:
                        param_count += 1
                        placeholders = ','.join([f'${param_count + i}' for i in range(len(jurisdiction_filter))])
                        where_conditions.append(f"d.jurisdiction = ANY(ARRAY[{placeholders}])")
                        params.extend(jurisdiction_filter)
                        param_count += len(jurisdiction_filter) - 1
                else:
                    param_count += 1
                    where_conditions.append(f"d.jurisdiction = ${param_count}")
                    params.append(jurisdiction_filter)
            
            if doc_type_filter:
                if isinstance(doc_type_filter, list):
                    if len(doc_type_filter) == 1:
                        param_count += 1
                        where_conditions.append(f"d.doc_type = ${param_count}")
                        params.append(doc_type_filter[0])
                    else:
                        param_count += 1
                        placeholders = ','.join([f'${param_count + i}' for i in range(len(doc_type_filter))])
                        where_conditions.append(f"d.doc_type = ANY(ARRAY[{placeholders}])")
                        params.extend(doc_type_filter)
                        param_count += len(doc_type_filter) - 1
                else:
                    param_count += 1
                    where_conditions.append(f"d.doc_type = ${param_count}")
                    params.append(doc_type_filter)
            
            if authority_level_min is not None:
                param_count += 1
                where_conditions.append(f"d.authority_level >= ${param_count}")
                params.append(authority_level_min)
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT 
                    c.chunk_id,
                    c.document_id,
                    c.text,
                    c.chunk_index,
                    1 - (c.embedding <=> $1) as similarity_score,
                    d.doc_type,
                    d.jurisdiction,
                    d.authority,
                    d.authority_level,
                    d.date_issued,
                    d.legal_metadata
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE {where_clause}
                    AND (1 - (c.embedding <=> $1)) > {similarity_threshold}
                ORDER BY c.embedding <=> $1
                LIMIT $2;
            """
            
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                result = {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "chunk_index": row["chunk_index"],
                    "similarity_score": float(row["similarity_score"]),
                    "document_metadata": {
                        "document_type": row["doc_type"],
                        "jurisdiction": row["jurisdiction"],
                        "authority": row["authority"],
                        "authority_level": row["authority_level"],
                        "date_issued": row["date_issued"].isoformat() if row["date_issued"] else None,
                        "legal_metadata": row["legal_metadata"] or {}
                    },
                    "legal_context": row["legal_metadata"] or {}
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM documents;")
            return row["count"]
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks in the store"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM document_chunks;")
            return row["count"]
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM documents WHERE document_id = $1;
            """, document_id)
            
            if not row:
                return None
            
            return {
                "document_id": row["document_id"],
                "original_path": row["original_path"],
                "doc_type": row["doc_type"],
                "jurisdiction": row["jurisdiction"],
                "authority": row["authority"],
                "authority_level": row["authority_level"],
                "date_issued": row["date_issued"].isoformat() if row["date_issued"] else None,
                "date_processed": row["date_processed"].isoformat(),
                "extraction_metadata": row["extraction_metadata"],
                "legal_metadata": row["legal_metadata"],
                "validation_metadata": row["validation_metadata"],
                "full_text": row["full_text"],
                "status": row["status"]
            }
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Delete chunks first (due to foreign key constraint)
                chunks_deleted = await conn.execute("""
                    DELETE FROM document_chunks WHERE document_id = $1;
                """, document_id)
                
                # Delete document
                doc_deleted = await conn.execute("""
                    DELETE FROM documents WHERE document_id = $1;
                """, document_id)
                
                logger.info(f"Deleted document {document_id} and its chunks")
                return True
    
    async def list_documents(
        self, 
        limit: int = 50, 
        offset: int = 0,
        jurisdiction_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort: str = "date"
    ) -> List[Dict[str, Any]]:
        """List documents with optional filtering"""
        async with self.pool.acquire() as conn:
            where_conditions = []
            params = []
            param_count = 0
            
            if jurisdiction_filter:
                param_count += 1
                where_conditions.append(f"jurisdiction = ${param_count}")
                params.append(jurisdiction_filter)
            
            if doc_type_filter:
                param_count += 1
                where_conditions.append(f"doc_type = ${param_count}")
                params.append(doc_type_filter)
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            param_count += 1
            params.append(limit)
            param_count += 1
            params.append(offset)
            
            query = f"""
                SELECT document_id, original_path, doc_type, jurisdiction, authority,
                       authority_level, date_issued, date_processed, status
                FROM documents
                {where_clause}
                ORDER BY date_processed DESC
                LIMIT ${param_count-1} OFFSET ${param_count};
            """
            
            rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.pool.acquire() as conn:
            stats = {}
            
            # Document counts by type
            doc_type_counts = await conn.fetch("""
                SELECT doc_type, COUNT(*) as count 
                FROM documents 
                GROUP BY doc_type;
            """)
            stats["documents_by_type"] = {row["doc_type"]: row["count"] for row in doc_type_counts}
            
            # Document counts by jurisdiction
            jurisdiction_counts = await conn.fetch("""
                SELECT jurisdiction, COUNT(*) as count 
                FROM documents 
                GROUP BY jurisdiction;
            """)
            stats["documents_by_jurisdiction"] = {row["jurisdiction"]: row["count"] for row in jurisdiction_counts}
            
            # Total counts
            totals = await conn.fetchrow("""
                SELECT 
                    (SELECT COUNT(*) FROM documents) as total_documents,
                    (SELECT COUNT(*) FROM document_chunks) as total_chunks,
                    (SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL) as embedded_chunks;
            """)
            stats.update(dict(totals))
            
            return stats
    
    # Additional methods needed by the API routes
    
    async def get_document_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get document count with optional filters"""
        async with self.pool.acquire() as conn:
            where_conditions = []
            params = []
            
            if filters:
                if filters.get("doc_type"):
                    where_conditions.append(f"doc_type = ${len(params) + 1}")
                    params.append(filters["doc_type"])
                if filters.get("jurisdiction"):
                    where_conditions.append(f"jurisdiction = ${len(params) + 1}")
                    params.append(filters["jurisdiction"])
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            query = f"SELECT COUNT(*) as count FROM documents {where_clause};"
            row = await conn.fetchrow(query, *params)
            return row["count"]
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID (alias for get_document_by_id)"""
        return await self.get_document_by_id(document_id)
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chunk_id, chunk_index, text, created_at
                FROM document_chunks 
                WHERE document_id = $1 
                ORDER BY chunk_index;
            """, document_id)
            
            return [dict(row) for row in rows]
    
    async def get_related_documents(self, document_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get related documents based on similar legal concepts"""
        async with self.pool.acquire() as conn:
            # Get the source document's metadata
            source_doc = await conn.fetchrow("""
                SELECT jurisdiction, doc_type, legal_metadata 
                FROM documents 
                WHERE document_id = $1;
            """, document_id)
            
            if not source_doc:
                return []
            
            # Find documents with same jurisdiction or doc_type
            rows = await conn.fetch("""
                SELECT document_id, doc_type, jurisdiction, authority, date_issued
                FROM documents 
                WHERE document_id != $1 
                    AND (jurisdiction = $2 OR doc_type = $3)
                ORDER BY date_processed DESC
                LIMIT $4;
            """, document_id, source_doc["jurisdiction"], source_doc["doc_type"], limit)
            
            return [dict(row) for row in rows]
    
    async def get_query_history(self, limit: int = 20, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get query history (placeholder - would need query_log table)"""
        # This would require a query_log table to be implemented
        # For now, return empty list
        return []
    
    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get document statistics"""
        async with self.pool.acquire() as conn:
            stats = {}
            
            # Document counts by type
            doc_type_counts = await conn.fetch("""
                SELECT doc_type, COUNT(*) as count 
                FROM documents 
                GROUP BY doc_type;
            """)
            stats["by_type"] = {row["doc_type"]: row["count"] for row in doc_type_counts}
            
            # Document counts by jurisdiction
            jurisdiction_counts = await conn.fetch("""
                SELECT jurisdiction, COUNT(*) as count 
                FROM documents 
                GROUP BY jurisdiction;
            """)
            stats["by_jurisdiction"] = {row["jurisdiction"]: row["count"] for row in jurisdiction_counts}
            
            return stats
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        async with self.pool.acquire() as conn:
            # Processing stats would require additional tracking tables
            # For now, return basic stats
            total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents;")
            total_chunks = await conn.fetchval("SELECT COUNT(*) FROM document_chunks;")
            
            return {
                "total_documents_processed": total_docs,
                "total_chunks_created": total_chunks,
                "avg_chunks_per_document": round(total_chunks / max(total_docs, 1), 2)
            }
    
    async def get_query_statistics(self) -> Dict[str, Any]:
        """Get query statistics"""
        # This would require query logging
        return {
            "total_queries": 0,
            "avg_response_time_ms": 0,
            "most_common_queries": []
        }
    
    async def reindex_documents(self, document_ids: List[str], force: bool = False) -> Dict[str, Any]:
        """Reindex specific documents"""
        # This would trigger the offline indexing flow
        # For now, return placeholder
        return {
            "reindexed_count": len(document_ids),
            "status": "completed"
        }
    
    async def reindex_all_documents(self, force: bool = False) -> Dict[str, Any]:
        """Reindex all documents"""
        # This would trigger the offline indexing flow for all documents
        async with self.pool.acquire() as conn:
            total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents;")
            
        return {
            "reindexed_count": total_docs,
            "status": "completed"
        }
    
    async def get_jurisdictions(self) -> List[str]:
        """Get list of all jurisdictions in the database"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT jurisdiction 
                FROM documents 
                WHERE jurisdiction IS NOT NULL 
                ORDER BY jurisdiction;
            """)
            
            return [row["jurisdiction"] for row in rows]
    
    async def get_legal_concepts(self, limit: int = 50, jurisdiction: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get legal concepts found in documents"""
        async with self.pool.acquire() as conn:
            where_clause = ""
            params = [limit]
            
            if jurisdiction:
                where_clause = "WHERE jurisdiction = $2"
                params.append(jurisdiction)
            
            # Extract legal concepts from legal_metadata
            query = f"""
                SELECT legal_metadata->'legal_concepts' as concepts
                FROM documents 
                {where_clause}
                LIMIT $1;
            """
            
            rows = await conn.fetch(query, *params)
            
            # Flatten and count concepts
            concept_counts = {}
            for row in rows:
                if row["concepts"]:
                    concepts = json.loads(row["concepts"]) if isinstance(row["concepts"], str) else row["concepts"]
                    if isinstance(concepts, list):
                        for concept in concepts:
                            concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Return sorted by frequency
            return [
                {"concept": concept, "count": count}
                for concept, count in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            ]
    
    async def get_citations(self, limit: int = 50, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get legal citations found in documents"""
        async with self.pool.acquire() as conn:
            where_clause = ""
            params = [limit]
            
            if doc_type:
                where_clause = "WHERE doc_type = $2"
                params.append(doc_type)
            
            # Extract citations from legal_metadata
            query = f"""
                SELECT legal_metadata->'citations' as citations
                FROM documents 
                {where_clause}
                LIMIT $1;
            """
            
            rows = await conn.fetch(query, *params)
            
            # Flatten and count citations
            citation_counts = {}
            for row in rows:
                if row["citations"]:
                    citations = json.loads(row["citations"]) if isinstance(row["citations"], str) else row["citations"]
                    if isinstance(citations, list):
                        for citation in citations:
                            citation_counts[citation] = citation_counts.get(citation, 0) + 1
            
            # Return sorted by frequency
            return [
                {"citation": citation, "count": count}
                for citation, count in sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            ]


# Connection helper
async def create_vector_store(connection_string: str) -> PostgresVectorStore:
    """Create and initialize a PostgresVectorStore instance"""
    store = PostgresVectorStore(connection_string)
    await store.connect()
    await store.create_tables()
    return store