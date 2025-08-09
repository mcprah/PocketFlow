# Marriage Law RAG System - API Reference

Complete API documentation for the Marriage Law RAG system.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required for development. In production, implement appropriate authentication mechanisms.

## Endpoints

### Document Management

#### Upload Document

Upload and process a legal document for indexing.

```http
POST /documents/upload
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | PDF/DOC/DOCX file to upload |
| doc_type | String | Yes | Document type: `case`, `statute`, `article`, `gazette`, `regulation` |
| jurisdiction | String | Yes | Jurisdiction: `federal`, state names, `local` |
| authority | String | No | Issuing authority (e.g., "California Supreme Court") |
| date_issued | String | No | Document date (YYYY-MM-DD format) |

**Example:**

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@smith_v_jones.pdf" \
  -F "doc_type=case" \
  -F "jurisdiction=california" \
  -F "authority=CA Supreme Court" \
  -F "date_issued=2023-05-15"
```

**Response:**

```json
{
  "document_id": "doc_123456",
  "status": "processed",
  "extraction_method": "pdfplumber",
  "quality_score": 0.95,
  "page_count": 15,
  "chunk_count": 23,
  "legal_metadata": {
    "case_number": "2023-CA-001234",
    "court": "California Supreme Court",
    "jurisdiction": "california",
    "document_type": "case",
    "authority_level": 10,
    "legal_concepts": ["marriage", "divorce", "property division"],
    "citations": ["Family Code 2550", "Marriage Cases (2008) 43 Cal.4th 757"]
  },
  "processing_time_ms": 15420
}
```

#### List Documents

Retrieve list of processed documents with metadata.

```http
GET /documents/
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| doc_type | String | Filter by document type |
| jurisdiction | String | Filter by jurisdiction |
| limit | Integer | Number of results (default: 50, max: 200) |
| offset | Integer | Pagination offset (default: 0) |
| sort | String | Sort by: `date`, `title`, `authority_level` |

**Example:**

```bash
curl "http://localhost:8000/documents/?doc_type=case&jurisdiction=california&limit=10"
```

**Response:**

```json
{
  "documents": [
    {
      "document_id": "doc_123456",
      "title": "Smith v. Jones",
      "doc_type": "case",
      "jurisdiction": "california",
      "authority": "CA Supreme Court",
      "authority_level": 10,
      "date_issued": "2023-05-15",
      "chunk_count": 23,
      "legal_concepts": ["marriage", "divorce"],
      "file_path": "/uploads/smith_v_jones.pdf"
    }
  ],
  "total_count": 147,
  "limit": 10,
  "offset": 0
}
```

#### Get Document Details

Retrieve detailed information about a specific document.

```http
GET /documents/{document_id}
```

**Response:**

```json
{
  "document_id": "doc_123456",
  "title": "Smith v. Jones", 
  "doc_type": "case",
  "jurisdiction": "california",
  "authority": "CA Supreme Court",
  "authority_level": 10,
  "date_issued": "2023-05-15",
  "extraction_details": {
    "method": "pdfplumber",
    "quality_score": 0.95,
    "page_count": 15,
    "extraction_time_ms": 8500
  },
  "legal_metadata": {
    "case_number": "2023-CA-001234",
    "legal_concepts": ["marriage", "divorce", "property division"],
    "citations": ["Family Code 2550"],
    "parties": ["Smith", "Jones"]
  },
  "chunks": [
    {
      "chunk_id": "chunk_789",
      "text": "The court held that marital property...",
      "section_type": "holding",
      "page_number": 5
    }
  ]
}
```

#### Delete Document

Remove a document and all associated chunks from the system.

```http
DELETE /documents/{document_id}
```

**Response:**

```json
{
  "message": "Document doc_123456 successfully deleted",
  "deleted_chunks": 23
}
```

### Query System

#### Search Documents

Query the document collection using natural language.

```http
POST /query
```

**Request Body:**

```json
{
  "question": "What are the grounds for divorce in California?",
  "jurisdiction_filter": ["california", "federal"],
  "doc_type_filter": ["case", "statute"],
  "date_range": {
    "start": "2020-01-01",
    "end": "2024-12-31"
  },
  "authority_level_min": 6,
  "max_results": 5,
  "include_citations": true
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| question | String | Yes | Natural language query |
| jurisdiction_filter | Array[String] | No | Filter by jurisdictions |
| doc_type_filter | Array[String] | No | Filter by document types |
| date_range | Object | No | Date range filter |
| authority_level_min | Integer | No | Minimum authority level (1-10) |
| max_results | Integer | No | Maximum results (default: 5, max: 20) |
| include_citations | Boolean | No | Include formatted citations |

**Response:**

```json
{
  "query": "What are the grounds for divorce in California?",
  "results": [
    {
      "chunk_id": "chunk_789",
      "document_id": "doc_123456",
      "text": "California Family Code Section 2310 establishes that the grounds for dissolution of marriage include irreconcilable differences and incurable insanity...",
      "similarity_score": 0.92,
      "document_metadata": {
        "title": "California Family Code",
        "doc_type": "statute",
        "jurisdiction": "california",
        "authority": "California Legislature",
        "authority_level": 9,
        "date_issued": "2023-01-01"
      },
      "legal_context": {
        "section": "Family Code 2310",
        "legal_concepts": ["divorce", "dissolution", "grounds"],
        "citations": ["Family Code 2310", "Family Code 2311"]
      }
    }
  ],
  "generated_answer": "In California, the primary grounds for divorce are: 1) Irreconcilable differences that have caused the irremedial breakdown of the marriage, and 2) Incurable insanity. California is a no-fault divorce state, meaning you don't need to prove wrongdoing by either spouse.",
  "confidence_score": 0.89,
  "processing_time_ms": 2340,
  "citations": [
    "Cal. Fam. Code § 2310 (2023)",
    "Cal. Fam. Code § 2311 (2023)"
  ]
}
```

#### Query Suggestions

Get suggested queries based on document content.

```http
GET /query/suggestions
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| jurisdiction | String | Focus suggestions on jurisdiction |
| doc_type | String | Focus suggestions on document type |
| limit | Integer | Number of suggestions (default: 10) |

**Response:**

```json
{
  "suggestions": [
    "What are the requirements for marriage in California?",
    "How is property divided in divorce cases?",
    "What constitutes community property?",
    "Grounds for annulment vs divorce",
    "Same-sex marriage legal precedents"
  ]
}
```

### System Status

#### Health Check

Check system health and availability.

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "database": "connected",
  "vector_store": "operational",
  "document_count": 1247,
  "chunk_count": 45823,
  "last_update": "2024-01-15T10:30:00Z"
}
```

#### System Statistics

Get detailed system statistics and performance metrics.

```http
GET /stats
```

**Response:**

```json
{
  "document_stats": {
    "total_documents": 1247,
    "by_type": {
      "case": 456,
      "statute": 234, 
      "article": 345,
      "gazette": 123,
      "regulation": 89
    },
    "by_jurisdiction": {
      "federal": 234,
      "california": 567,
      "new_york": 123,
      "other": 323
    }
  },
  "processing_stats": {
    "total_chunks": 45823,
    "avg_chunk_size": 1456,
    "extraction_methods": {
      "pdfplumber": 756,
      "pymupdf": 234,
      "pypdf2": 123,
      "ocr": 134
    }
  },
  "query_stats": {
    "total_queries": 5647,
    "avg_response_time_ms": 1234,
    "popular_topics": [
      "divorce grounds",
      "property division",
      "marriage requirements"
    ]
  }
}
```

## Error Codes

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created (document uploaded) |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (document/resource not found) |
| 413 | Payload Too Large (file size exceeded) |
| 415 | Unsupported Media Type (invalid file format) |
| 422 | Unprocessable Entity (validation errors) |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "Document with ID 'doc_123456' not found",
    "details": {
      "document_id": "doc_123456",
      "suggestion": "Check the document ID and try again"
    }
  }
}
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `INVALID_FILE_FORMAT` | Unsupported file type | Use PDF, DOC, or DOCX files |
| `FILE_TOO_LARGE` | File exceeds size limit | Files must be under 50MB |
| `EXTRACTION_FAILED` | Could not extract text | Check file integrity, try OCR |
| `INVALID_JURISDICTION` | Unknown jurisdiction | Use valid jurisdiction codes |
| `QUERY_TOO_SHORT` | Query is too short | Use at least 3 words |
| `NO_RESULTS_FOUND` | No matching documents | Try broader search terms |

## Rate Limits

- **Document Upload**: 10 files per minute
- **Queries**: 100 requests per minute  
- **Bulk Operations**: 5 requests per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640995200
```

## SDKs and Examples

### Python SDK Example

```python
import requests
import json

class MarriageLawRAG:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_document(self, file_path, doc_type, jurisdiction):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'doc_type': doc_type,
                'jurisdiction': jurisdiction
            }
            response = requests.post(f"{self.base_url}/documents/upload", 
                                   files=files, data=data)
        return response.json()
    
    def query(self, question, filters=None):
        payload = {'question': question}
        if filters:
            payload.update(filters)
        
        response = requests.post(f"{self.base_url}/query",
                               json=payload)
        return response.json()

# Usage
rag = MarriageLawRAG()

# Upload document
result = rag.upload_document(
    "marriage_statute.pdf", 
    "statute", 
    "california"
)

# Query system  
results = rag.query(
    "What are marriage requirements?",
    filters={"jurisdiction_filter": ["california"]}
)
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class MarriageLawRAG {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async uploadDocument(filePath, docType, jurisdiction) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));
        form.append('doc_type', docType);
        form.append('jurisdiction', jurisdiction);
        
        const response = await axios.post(
            `${this.baseUrl}/documents/upload`,
            form,
            { headers: form.getHeaders() }
        );
        
        return response.data;
    }
    
    async query(question, filters = {}) {
        const response = await axios.post(`${this.baseUrl}/query`, {
            question,
            ...filters
        });
        
        return response.data;
    }
}

// Usage
const rag = new MarriageLawRAG();

// Upload and query
(async () => {
    const uploadResult = await rag.uploadDocument(
        'marriage_law.pdf',
        'statute', 
        'california'
    );
    
    const queryResult = await rag.query(
        'What are the marriage requirements?',
        { jurisdiction_filter: ['california'] }
    );
    
    console.log(queryResult.generated_answer);
})();
```

## WebSocket API (Real-time Updates)

For real-time processing updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'document_processing':
            console.log(`Processing: ${data.filename} - ${data.progress}%`);
            break;
        case 'processing_complete':
            console.log(`Complete: ${data.document_id}`);
            break;
    }
};
```