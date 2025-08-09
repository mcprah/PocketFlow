# Marriage Law RAG System - Setup Guide

This guide will walk you through setting up the Marriage Law RAG system for processing legal documents.

## Prerequisites

- Python 3.9+
- PostgreSQL 14+ with pgvector extension
- Tesseract OCR (for scanned documents)
- OpenAI API key

## Installation Steps

### 1. Environment Setup

```bash
# Navigate to the project directory
cd projects/marriage-law-rag/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. PostgreSQL Setup

```bash
# Install PostgreSQL and pgvector (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector

# Create database
sudo -u postgres createdb marriage_law_db
sudo -u postgres psql marriage_law_db -c "CREATE EXTENSION vector;"
```

### 3. Tesseract OCR Setup

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 4. Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/marriage_law_db

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# OCR Configuration
TESSERACT_CMD=/usr/bin/tesseract  # Adjust path as needed

# Application Configuration
DEBUG=true
LOG_LEVEL=info
```

### 5. Database Schema Creation

```bash
# Run database migrations
python -m database.create_schema
```

### 6. Test Installation

```bash
# Run basic functionality test
python -m tests.test_setup

# Test PDF processing
python -c "
from utils.enhanced_pdf_extractor import EnhancedPDFExtractor
extractor = EnhancedPDFExtractor()
print('✅ PDF extractor ready')
"

# Test database connection
python -c "
from utils.postgres_vector_store import PostgresVectorStore
import os
store = PostgresVectorStore(os.getenv('DATABASE_URL'))
store.connect()
print('✅ Database connection successful')
"
```

## Running the Application

### Development Mode

```bash
# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access the application
# Web UI: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Production Mode

```bash
# Start with uvicorn production settings
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Usage

### 1. Upload Documents

```bash
# Via API
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@path/to/marriage_law.pdf" \
  -F "doc_type=statute" \
  -F "jurisdiction=california"
```

### 2. Query the System

```bash
# Via API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the requirements for marriage in California?",
    "jurisdiction_filter": ["california", "federal"]
  }'
```

### 3. Web Interface

Visit `http://localhost:8000` to use the web interface for:
- Document upload and management
- Interactive querying
- Result visualization
- System monitoring

## Troubleshooting

### Common Issues

1. **OCR Not Working**
   - Ensure Tesseract is installed and path is correct in .env
   - Check that required language packs are installed

2. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check DATABASE_URL format
   - Ensure pgvector extension is installed

3. **PDF Extraction Issues**
   - Install additional PDF processing libraries if needed
   - Check file permissions and paths

4. **Memory Issues**
   - Increase available memory for large document processing
   - Consider batch processing for many documents

### Logs

Application logs are available at:
- Development: Console output
- Production: `logs/marriage-law-rag.log`

## Performance Optimization

### For Large Document Collections

1. **Batch Processing**
   ```python
   # Process documents in batches
   python scripts/batch_process.py --batch-size 10 /path/to/documents/
   ```

2. **Database Indexing**
   - Ensure proper indexes are created (handled automatically)
   - Consider partitioning for very large datasets

3. **Caching**
   - Enable Redis caching for frequent queries
   - Configure embedding caching

## Security Considerations

- Keep API keys secure and never commit to version control
- Use environment variables for all sensitive configuration
- Consider rate limiting for production deployments
- Implement proper authentication for document uploads

## Next Steps

- Review the [User Guide](user-guide.md) for detailed usage instructions
- Check [API Reference](api-reference.md) for complete API documentation
- See [Design Document](design.md) for system architecture details