# CLAUDE.md - Marriage Law RAG System

This file provides guidance to Claude Code (claude.ai/code) when working with the Marriage Law RAG System.

## Project Overview

The Marriage Law RAG System is a production-ready Retrieval Augmented Generation application built using the PocketFlow framework. It specializes in processing legal documents related to marriage law in Ghana and provides accurate, context-aware responses to legal queries.

## Core Architecture

### Technology Stack
- **Framework**: PocketFlow for workflow orchestration
- **Backend**: FastAPI for REST API and WebSocket support
- **Database**: PostgreSQL with pgvector for vector storage
- **AI Models**: OpenAI GPT-4.1 for answer generation, text-embedding-3-small for embeddings
- **Document Processing**: Enhanced PDF extraction with OCR support
- **Frontend**: HTML/JavaScript with real-time WebSocket updates

### Key Components

1. **Document Processing Pipeline** (`nodes/`):
   - `DocumentExtractionNode`: Extracts text and metadata from legal documents
   - `DocumentChunkingNode`: Splits documents into overlapping chunks
   - `EmbeddingNode`: Generates vector embeddings for chunks
   - `VectorStoreNode`: Stores chunks in PostgreSQL with pgvector

2. **Query Processing Pipeline** (`nodes/`):
   - `QueryEmbeddingNode`: Generates embedding for user queries
   - `VectorSearchNode`: Performs similarity search with filters
   - `AnswerGenerationNode`: Generates contextual answers using GPT-4

3. **Flow Orchestration** (`flows.py`):
   - `create_offline_indexing_flow()`: Document processing workflow
   - `create_online_query_flow()`: Query processing workflow

4. **API Handlers** (`handlers/`):
   - `document_handler.py`: Document upload and validation logic
   - `query_handler.py`: Query processing and response formatting
   - `health_handler.py`: System health monitoring

5. **Background Tasks** (`tasks/`):
   - `document_processor.py`: Asynchronous document processing

6. **WebSocket Management** (`ws/`):
   - `manager.py`: Real-time communication and broadcasting

7. **Application Core** (`core/`):
   - `app_state.py`: Application startup and shutdown management

8. **API Layer** (`main.py`):
   - FastAPI application configuration and endpoint routing
   - Middleware and dependency injection

9. **Utilities** (`utils/`):
   - `enhanced_pdf_extractor.py`: Advanced PDF processing with legal metadata extraction
   - `postgres_vector_store.py`: PostgreSQL integration with pgvector operations

## Commands for Development

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_query_endpoint.py -v

# Test with coverage
python -m pytest --cov=. tests/
```

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and database URL

# Run development server
python main.py

# Run production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Database Setup
```bash
# Ensure PostgreSQL is running with pgvector extension
# The application will automatically create tables on startup
```

## PocketFlow Implementation Details

### Offline Indexing Flow
```python
extract_node >> chunk_node >> embed_node >> store_node

# Conditional transitions:
extract_node - "chunk" >> chunk_node
chunk_node - "embed" >> embed_node  
embed_node - "store" >> store_node
extract_node - "failed" >> store_node  # Handle failed extractions
```

### Online Query Flow
```python
query_embed_node >> search_node >> generate_node

# Conditional transitions:
query_embed_node - "search" >> search_node
search_node - "generate" >> generate_node
search_node - "no_results" >> generate_node
query_embed_node - "failed" >> generate_node
```

### Shared Store Data Structure

**Offline Indexing**:
```python
shared = {
    "raw_documents": [{"path": str, "doc_type": str, "jurisdiction": str, ...}],
    "extracted_documents": [{"document_id": str, "text": str, "legal_metadata": dict, ...}],
    "document_chunks": [{"chunk_id": str, "text": str, "document_metadata": dict, ...}],
    "embedded_chunks": [{"chunk_id": str, "embedding": List[float], ...}],
    "indexing_result": {"total_chunks": int, "stored_chunks": int}
}
```

**Online Querying**:
```python
shared = {
    "user_query": str,
    "query_embedding": List[float],
    "search_results": [{"chunk_id": str, "similarity_score": float, ...}],
    "generated_answer": str,
    "confidence_score": float,
    "citations": List[dict]
}
```

## Development Guidelines

### Project Structure

The project now follows a modular architecture with clear separation of concerns:

```
├── main.py                        # FastAPI app configuration and routing
├── flows.py                       # PocketFlow workflow orchestration
├── nodes/                         # PocketFlow node implementations
│   ├── __init__.py
│   ├── document_extraction_node.py
│   ├── document_chunking_node.py
│   ├── embedding_node.py
│   ├── vector_store_node.py
│   ├── query_embedding_node.py
│   ├── vector_search_node.py
│   └── answer_generation_node.py
├── handlers/                      # FastAPI endpoint handlers
│   ├── __init__.py
│   ├── document_handler.py
│   ├── query_handler.py
│   └── health_handler.py
├── tasks/                         # Background task processing
│   ├── __init__.py
│   └── document_processor.py
├── ws/                            # Real-time communication
│   ├── __init__.py
│   └── manager.py
├── core/                          # Application lifecycle management
│   ├── __init__.py
│   └── app_state.py
├── utils/                         # Utility modules
│   ├── enhanced_pdf_extractor.py
│   └── postgres_vector_store.py
├── api/                           # API models and routes
│   ├── models.py
│   └── routes.py
└── config/                        # Configuration management
    └── settings.py
```

Each module follows the PocketFlow pattern with proper imports and error handling.

### Adding New Features

1. **New Document Types**: 
   - Modify `DocumentExtractionNode` in `nodes/document_extraction_node.py`
   - Update `LegalMetadataExtractor` for document-specific metadata

2. **Enhanced Filtering**:
   - Modify `VectorSearchNode` in `nodes/vector_search_node.py`
   - Update database schema in `utils/postgres_vector_store.py`

3. **Custom Embeddings**:
   - Replace `EmbeddingNode` in `nodes/embedding_node.py`
   - Ensure vector dimensions match database configuration

4. **New Node Types**:
   - Create new node file in `nodes/` directory
   - Follow existing patterns for imports and error handling
   - Add to `nodes/__init__.py` exports
   - Update flow definitions in `flows.py`

5. **New API Endpoints**:
   - Create handler function in appropriate `handlers/` module
   - Add endpoint wrapper in `main.py` with dependency injection
   - Update `api/models.py` with request/response models

6. **Background Tasks**:
   - Add task functions to `tasks/` modules
   - Use WebSocket manager for real-time updates
   - Follow error handling patterns with proper logging

### Error Handling

- Nodes use PocketFlow's built-in retry mechanism (max_retries parameter)
- Failed documents are tracked in `shared["failed_documents"]`
- Fallback methods provide graceful degradation
- WebSocket broadcasts keep UI updated on processing status

### Security Considerations

- Input validation on file uploads (size, type restrictions)
- Environment variable management for API keys
- Database connection security with proper credentials
- CORS configuration for production deployment

## API Endpoints

### Core Endpoints
- `POST /documents/upload`: Upload legal documents for processing
- `POST /query`: Query the document collection
- `GET /health`: System health and statistics
- `WebSocket /ws`: Real-time processing updates

### Query Parameters
- `jurisdiction_filter`: Filter by legal jurisdiction
- `doc_type_filter`: Filter by document type
- `authority_level_min`: Minimum authority level for results
- `max_results`: Number of results to return

## Production Deployment

### Environment Setup
```bash
# Required environment variables
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://user:pass@host:port/dbname
ALLOWED_ORIGINS=["http://localhost:3000"]
```

### Database Requirements
- PostgreSQL 12+ with pgvector extension
- Proper indexing on embedding columns for performance
- Regular maintenance for optimal vector search performance

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key validity and rate limits
2. **Database Connection**: Verify PostgreSQL is running and accessible
3. **PDF Extraction Failures**: Ensure proper file permissions and OCR dependencies
4. **Memory Issues**: Monitor embedding batch sizes for large documents

### Performance Optimization

- Use connection pooling for database operations
- Implement caching for frequently accessed embeddings
- Consider batch processing for multiple document uploads
- Monitor vector search performance with database indexing