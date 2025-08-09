# Marriage Law RAG System - User Guide

This guide explains how to use the Marriage Law RAG system for legal research and document analysis.

## Overview

The Marriage Law RAG (Retrieval Augmented Generation) system helps legal professionals and researchers quickly find relevant information from a large collection of marriage-related legal documents including:

- Court cases and precedents
- Statutes and legislation  
- Legal articles and commentary
- Government gazettes and announcements
- Regulations and administrative rules

## Getting Started

### Web Interface

1. **Access the System**
   - Navigate to `http://localhost:8000` in your browser
   - You'll see the main dashboard with upload and query options

2. **Upload Documents**
   - Click "Upload Documents" 
   - Select PDF or DOC files
   - Specify document type (case, statute, article, gazette, regulation)
   - Set jurisdiction (federal, state, local)
   - Click "Process" to extract and index the documents

3. **Query Documents**
   - Use the search box to enter your legal question
   - Apply filters for jurisdiction, document type, date range
   - Review results with citations and authority levels

### API Interface

For programmatic access, use the REST API endpoints:

```bash
# Upload a document
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@marriage_statute.pdf" \
  -F "doc_type=statute" \
  -F "jurisdiction=california"

# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are grounds for divorce?",
    "jurisdiction_filter": ["california"],
    "doc_type_filter": ["statute", "case"]
  }'
```

## Document Processing

### Supported File Formats

- **PDF**: Primary format for legal documents
- **DOC/DOCX**: Microsoft Word documents
- **OCR**: Scanned documents (automatically detected)

### Document Types

1. **Cases**: Court decisions and legal precedents
   - Automatically extracts case numbers, courts, dates
   - Identifies parties, holdings, and legal reasoning
   - Ranks by court authority level

2. **Statutes**: Primary legislation and codes  
   - Parses section numbers and subsections
   - Extracts effective dates and amendments
   - Links related provisions

3. **Articles**: Academic papers and legal commentary
   - Identifies authors and publication details
   - Extracts key legal concepts and analysis
   - Lower authority weight than primary sources

4. **Gazettes**: Official government announcements
   - Processes regulatory updates and notices
   - Tracks effective dates and compliance requirements

5. **Regulations**: Administrative rules and procedures
   - Links to enabling statutes
   - Identifies regulatory agencies
   - Tracks implementation timelines

### Quality Indicators

The system provides quality scores for each document:

- **Extraction Quality** (0.0-1.0): How well text was extracted
- **OCR Confidence** (0-100%): Accuracy for scanned documents  
- **Validation Score** (0.0-1.0): Overall document completeness
- **Authority Level** (1-10): Legal precedential value

## Querying the System

### Query Types

1. **Definitional Queries**
   - "What is common law marriage?"
   - "Define marital property"
   - "What constitutes abandonment?"

2. **Procedural Queries**
   - "How to file for divorce in California?"
   - "What forms are needed for marriage license?"
   - "Steps for legal separation"

3. **Legal Analysis**
   - "Grounds for annulment vs divorce"
   - "Same-sex marriage precedents"
   - "Property division in high-asset divorces"

### Search Filters

**Jurisdiction Filters**:
- Federal: Supreme Court, Federal Circuit decisions
- State: State-specific laws and cases  
- Local: County/city ordinances and procedures

**Document Type Filters**:
- Cases: Court decisions and precedents
- Statutes: Primary legislation
- Articles: Secondary sources and commentary
- Regulations: Administrative rules

**Date Range Filters**:
- Recent: Last 5 years
- Modern: Since 2000
- Historical: All available dates

**Authority Level Filters**:
- Binding (8-10): Supreme Court, primary statutes
- Persuasive (6-7): Appellate courts, regulations
- Commentary (1-5): Articles, lower court decisions

### Understanding Results

Each result includes:

1. **Relevant Text**: The specific passage that matches your query
2. **Source Information**: Document title, court/authority, date
3. **Citations**: Legal citations in proper format
4. **Authority Level**: Precedential weight (1-10 scale)
5. **Jurisdiction**: Geographic scope of authority
6. **Similarity Score**: How well the text matches your query

### Best Practices

1. **Use Specific Legal Terms**
   - ✅ "prenuptial agreement validity"
   - ❌ "marriage contract rules"

2. **Include Jurisdiction When Relevant**  
   - ✅ "California community property laws"
   - ✅ "federal same-sex marriage recognition"

3. **Be Specific About Document Types**
   - ✅ "divorce statute requirements" 
   - ✅ "child custody case precedents"

4. **Use Complete Legal Phrases**
   - ✅ "grounds for legal separation"
   - ❌ "separation reasons"

## Advanced Features

### Bulk Document Processing

For large document collections:

```bash
# Process entire directories
python scripts/bulk_process.py /path/to/legal/documents/ \
  --doc-type mixed \
  --jurisdiction federal \
  --batch-size 20
```

### Citation Analysis

The system automatically:
- Extracts and validates legal citations
- Links related cases and statutes
- Identifies overruled or superseded authority
- Tracks legislative changes and amendments

### Export Options

Results can be exported in multiple formats:
- **PDF Report**: Formatted legal memorandum
- **Word Document**: Editable research summary  
- **CSV**: Structured data for analysis
- **JSON**: API-friendly format

### Integration Options

**Legal Research Platforms**:
- Export citations to Westlaw, Lexis formats
- Import research notes and annotations
- Sync with case management systems

**Document Management**:
- Integrate with firm document repositories  
- Automated workflow triggers
- Version control for updated statutes

## Troubleshooting

### Common Issues

1. **Poor Search Results**
   - Try different search terms or synonyms
   - Broaden jurisdiction or document type filters
   - Check spelling of legal terms

2. **Document Upload Failures**
   - Ensure files are under 50MB
   - Check file format (PDF, DOC, DOCX only)
   - Verify file isn't password protected

3. **Slow Query Response**
   - Large document collections may take longer
   - Try more specific queries to reduce search scope
   - Consider using filters to narrow results

### Getting Help

- **Documentation**: Check this guide and API reference
- **Logs**: Review system logs for error details
- **Support**: Contact system administrator for technical issues

## Legal Disclaimers

⚠️ **Important Legal Notice**:

- This system is for research assistance only
- Always verify results with current legal sources
- Consult qualified legal professionals for advice
- Laws change frequently - check for updates
- Not a substitute for professional legal judgment

## Privacy and Security

- Uploaded documents are stored securely
- Query logs help improve system performance
- No document content is shared externally
- Follow your organization's data policies
- Consider classification levels for sensitive documents