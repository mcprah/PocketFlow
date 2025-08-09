#!/usr/bin/env python3
"""
Enhanced PDF Processing Utilities for Marriage Law RAG System
Provides robust PDF extraction with multiple fallback strategies and OCR support.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
from datetime import datetime
import fitz  # pymupdf
import PyPDF2
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPDFExtractor:
    """
    Multi-strategy PDF text extractor with OCR fallback for legal documents.
    Uses PyPDF2, pdfplumber, and pymupdf with intelligent strategy selection.
    """
    
    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled
        self.extraction_strategies = ['pdfplumber', 'pymupdf', 'pypdf2']
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF/DOC with multiple fallback strategies.
        
        Returns:
            Dict containing extracted text, metadata, and extraction details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Handle different file types
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _extract_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using multiple strategies"""
        results = {
            'file_path': str(file_path),
            'extraction_method': None,
            'text': '',
            'page_count': 0,
            'metadata': {},
            'quality_score': 0.0,
            'extraction_details': {}
        }
        
        # Try each extraction strategy
        for strategy in self.extraction_strategies:
            try:
                if strategy == 'pdfplumber':
                    result = self._extract_with_pdfplumber(file_path)
                elif strategy == 'pymupdf':
                    result = self._extract_with_pymupdf(file_path)
                elif strategy == 'pypdf2':
                    result = self._extract_with_pypdf2(file_path)
                
                # Evaluate extraction quality
                quality_score = self._evaluate_extraction_quality(result['text'])
                result['quality_score'] = quality_score
                result['extraction_method'] = strategy
                
                # If quality is good enough, use this result
                if quality_score > 0.7:  # Threshold for acceptable quality
                    results.update(result)
                    break
                elif quality_score > results['quality_score']:
                    # Keep the best result so far
                    results.update(result)
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed for {file_path}: {e}")
                continue
        
        # If all strategies failed or quality is too low, try OCR
        if results['quality_score'] < 0.5 and self.ocr_enabled:
            logger.info(f"Low quality extraction ({results['quality_score']:.2f}), trying OCR...")
            try:
                ocr_result = self._extract_with_ocr(file_path)
                if ocr_result['quality_score'] > results['quality_score']:
                    results.update(ocr_result)
                    results['extraction_method'] = 'ocr'
            except Exception as e:
                logger.warning(f"OCR failed for {file_path}: {e}")
        
        return results
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Dict[str, Any]:
        """Extract using pdfplumber - best for complex layouts"""
        text_pages = []
        metadata = {}
        
        with pdfplumber.open(file_path) as pdf:
            metadata = {
                'page_count': len(pdf.pages),
                'pdf_metadata': pdf.metadata or {}
            }
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        
        full_text = '\n\n'.join(text_pages)
        
        return {
            'text': full_text,
            'page_count': len(text_pages),
            'metadata': metadata,
            'extraction_details': {'method': 'pdfplumber', 'pages_processed': len(text_pages)}
        }
    
    def _extract_with_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract using pymupdf - fast and reliable"""
        text_pages = []
        metadata = {}
        
        doc = fitz.open(file_path)
        metadata = {
            'page_count': doc.page_count,
            'pdf_metadata': doc.metadata
        }
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text_pages.append(page_text)
        
        doc.close()
        full_text = '\n\n'.join(text_pages)
        
        return {
            'text': full_text,
            'page_count': len(text_pages),
            'metadata': metadata,
            'extraction_details': {'method': 'pymupdf', 'pages_processed': len(text_pages)}
        }
    
    def _extract_with_pypdf2(self, file_path: Path) -> Dict[str, Any]:
        """Extract using PyPDF2 - good fallback option"""
        text_pages = []
        metadata = {}
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'page_count': len(pdf_reader.pages),
                'pdf_metadata': pdf_reader.metadata or {}
            }
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_pages.append(page_text)
        
        full_text = '\n\n'.join(text_pages)
        
        return {
            'text': full_text,
            'page_count': len(text_pages),
            'metadata': metadata,
            'extraction_details': {'method': 'pypdf2', 'pages_processed': len(text_pages)}
        }
    
    def _extract_with_ocr(self, file_path: Path) -> Dict[str, Any]:
        """Extract using OCR for scanned documents"""
        # Convert PDF to images
        images = convert_from_path(file_path, dpi=300)  # High DPI for better OCR
        
        text_pages = []
        confidence_scores = []
        
        for i, image in enumerate(images):
            # OCR with confidence data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate confidence
            page_text = pytesseract.image_to_string(image)
            page_confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            
            if page_text.strip():
                text_pages.append(page_text)
                if page_confidences:
                    confidence_scores.append(sum(page_confidences) / len(page_confidences))
        
        full_text = '\n\n'.join(text_pages)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'text': full_text,
            'page_count': len(text_pages),
            'metadata': {'avg_ocr_confidence': avg_confidence},
            'extraction_details': {
                'method': 'ocr',
                'pages_processed': len(text_pages),
                'avg_confidence': avg_confidence
            }
        }
    
    def _extract_from_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX files"""
        doc = Document(file_path)
        
        # Extract all paragraph text
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = '\n\n'.join(paragraphs)
        
        # Extract metadata
        core_props = doc.core_properties
        metadata = {
            'title': core_props.title or '',
            'author': core_props.author or '',
            'created': core_props.created.isoformat() if core_props.created else None,
            'modified': core_props.modified.isoformat() if core_props.modified else None,
        }
        
        return {
            'text': full_text,
            'page_count': len(paragraphs),  # Approximate
            'metadata': metadata,
            'quality_score': 1.0,  # DOCX extraction is typically reliable
            'extraction_details': {'method': 'docx', 'paragraphs_processed': len(paragraphs)}
        }
    
    def _evaluate_extraction_quality(self, text: str) -> float:
        """
        Evaluate the quality of extracted text for legal documents.
        Returns a score from 0.0 to 1.0.
        """
        if not text or len(text.strip()) < 100:
            return 0.0
        
        score = 0.0
        
        # Check for reasonable text length
        if len(text) > 500:
            score += 0.2
        
        # Check for legal document indicators
        legal_indicators = [
            r'\bcourt\b', r'\bjudge\b', r'\bstatute\b', r'\bsection\b',
            r'\bmarriage\b', r'\bdivorce\b', r'\blegal\b', r'\blaw\b',
            r'\bcivil\b', r'\bfamily\b', r'\bcode\b'
        ]
        
        found_indicators = sum(1 for pattern in legal_indicators 
                              if re.search(pattern, text.lower()))
        score += min(found_indicators / len(legal_indicators), 0.3)
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.strip()) > 20]
        if len(valid_sentences) > 10:
            score += 0.2
        
        # Check for reasonable character distribution (not too many garbage chars)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio > 0.6:  # At least 60% letters
            score += 0.3
        
        return min(score, 1.0)


class LegalMetadataExtractor:
    """
    Extract legal-specific metadata from PDF documents and text content.
    Specializes in marriage law documents.
    """
    
    def __init__(self):
        # Patterns for different legal document types
        self.case_patterns = {
            'case_number': [
                r'(?:Case|No\.?|Number)\s*:?\s*(\d{4}-\w+-\d+)',
                r'(\d{2,4}-CV-\d+)',
                r'(?:Civil|Civ\.?)\s*(?:No\.?|Number)\s*:?\s*([\w\d-]+)'
            ],
            'court': [
                r'(?:IN THE|BEFORE THE)\s+(.*?COURT.*?)(?:\n|FOR)',
                r'(.*?COURT.*?)(?:\n|,)',
                r'(?:Court|COURT)\s*:?\s*(.*?)(?:\n|$)'
            ],
            'date': [
                r'(?:Filed|Decided|Date)\s*:?\s*(\w+\s+\d{1,2},\s*\d{4})',
                r'(\d{1,2}\/\d{1,2}\/\d{4})',
                r'(\w+\s+\d{1,2},\s*\d{4})'
            ]
        }
        
        self.statute_patterns = {
            'code_section': [
                r'(?:Section|Sec\.?|§)\s*(\d+(?:\.\d+)*)',
                r'(Family Code\s+\d+)',
                r'(Marriage Act\s+\d+)'
            ],
            'jurisdiction': [
                r'(?:State of|Province of)\s+(\w+)',
                r'(\w+)\s+(?:Family|Marriage|Civil)\s+Code',
                r'Federal\s+(.*?)\s+Act'
            ]
        }
    
    def extract_metadata(self, text: str, pdf_metadata: Dict = None) -> Dict[str, Any]:
        """
        Extract legal metadata from document text and PDF properties.
        
        Args:
            text: Extracted document text
            pdf_metadata: PDF metadata from extraction
            
        Returns:
            Dictionary of extracted legal metadata
        """
        metadata = {
            'document_type': self._classify_document_type(text),
            'jurisdiction': self._extract_jurisdiction(text),
            'authority': self._extract_authority(text),
            'date_issued': self._extract_date(text),
            'case_number': self._extract_case_number(text),
            'legal_concepts': self._extract_legal_concepts(text),
            'citations': self._extract_citations(text),
            'authority_level': None  # Will be determined later
        }
        
        # Add PDF metadata if available
        if pdf_metadata:
            metadata['pdf_metadata'] = pdf_metadata
            # Try to extract title from PDF metadata
            if 'title' in pdf_metadata and pdf_metadata['title']:
                metadata['title'] = pdf_metadata['title']
        
        # Determine authority level based on extracted information
        metadata['authority_level'] = self._determine_authority_level(metadata)
        
        return metadata
    
    def _classify_document_type(self, text: str) -> str:
        """Classify the type of legal document"""
        text_lower = text.lower()
        
        # Case law indicators
        case_indicators = ['plaintiff', 'defendant', 'appellant', 'appellee', 
                          'petitioner', 'respondent', 'judgment', 'opinion']
        
        # Statute indicators  
        statute_indicators = ['section', 'code', 'act', 'chapter', 'subsection']
        
        # Article indicators
        article_indicators = ['law review', 'journal', 'commentary', 'analysis']
        
        case_score = sum(1 for indicator in case_indicators if indicator in text_lower)
        statute_score = sum(1 for indicator in statute_indicators if indicator in text_lower)
        article_score = sum(1 for indicator in article_indicators if indicator in text_lower)
        
        if case_score >= statute_score and case_score >= article_score:
            return 'case'
        elif statute_score >= article_score:
            return 'statute'
        elif article_score > 0:
            return 'article'
        else:
            return 'unknown'
    
    def _extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract jurisdiction from document text"""
        # Common jurisdiction patterns
        patterns = [
            r'(?:State of|Province of|Commonwealth of)\s+(\w+)',
            r'(\w+)\s+(?:Supreme|Superior|District|Family)\s+Court',
            r'United States\s+(\w+)\s+Court',
            r'Federal\s+(?:Court|District)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                jurisdiction = match.group(1) if match.groups() else 'federal'
                return jurisdiction.lower()
        
        # Look for state abbreviations in case citations
        state_abbrev = re.search(r'\b([A-Z]{2})\s+\d{4}\b', text)
        if state_abbrev:
            return state_abbrev.group(1).lower()
        
        return None
    
    def _extract_authority(self, text: str) -> Optional[str]:
        """Extract the issuing authority"""
        # Court patterns
        court_patterns = [
            r'(?:IN THE|BEFORE THE)\s+(.*?COURT[^.\n]*)',
            r'((?:Supreme|Superior|District|Family|Appellate)\s+Court[^.\n]*)',
            r'(?:Court|COURT)\s*:?\s*([^.\n]+)'
        ]
        
        for pattern in court_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Legislature patterns
        legislature_patterns = [
            r'(State Legislature)',
            r'(Congress)',
            r'(Parliament)',
        ]
        
        for pattern in legislature_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract document date"""
        date_patterns = [
            r'(?:Filed|Decided|Dated?|Issued)\s*:?\s*(\w+\s+\d{1,2},\s*\d{4})',
            r'(\d{1,2}\/\d{1,2}\/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\w+\s+\d{1,2},\s*\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_case_number(self, text: str) -> Optional[str]:
        """Extract case number if present"""
        case_patterns = [
            r'(?:Case|No\.?|Number)\s*:?\s*(\d{4}-\w+-\d+)',
            r'(\d{2,4}-CV-\d+)',
            r'(?:Civil|Civ\.?)\s*(?:No\.?|Number)\s*:?\s*([\w\d-]+)',
            r'Docket\s*(?:No\.?|Number)\s*:?\s*([\w\d-]+)'
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract marriage-related legal concepts"""
        marriage_concepts = [
            'marriage', 'matrimony', 'wedding', 'spouse', 'husband', 'wife',
            'divorce', 'dissolution', 'separation', 'annulment',
            'prenuptial', 'postnuptial', 'marital property', 'alimony',
            'child custody', 'child support', 'visitation',
            'domestic partnership', 'civil union', 'common law marriage',
            'same-sex marriage', 'bigamy', 'polygamy'
        ]
        
        found_concepts = []
        text_lower = text.lower()
        
        for concept in marriage_concepts:
            if concept in text_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text"""
        # Common citation patterns
        citation_patterns = [
            r'\d+\s+\w+\.?\s+\d+',  # e.g., "123 Cal. 456"
            r'\w+\s+v\.?\s+\w+',    # e.g., "Smith v. Jones"
            r'Family Code\s+§?\s*\d+',  # e.g., "Family Code 300"
            r'Marriage Act\s+\d+',      # e.g., "Marriage Act 1961"
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def _determine_authority_level(self, metadata: Dict) -> int:
        """
        Determine authority level (1-10) based on court/authority type.
        10 = Supreme Court, 1 = Local administrative
        """
        authority = metadata.get('authority', '').lower()
        
        if 'supreme' in authority:
            return 10
        elif 'appellate' in authority or 'appeals' in authority:
            return 8
        elif 'superior' in authority:
            return 7
        elif 'district' in authority:
            return 6
        elif 'family' in authority:
            return 6
        elif 'municipal' in authority or 'local' in authority:
            return 4
        elif 'congress' in authority or 'legislature' in authority:
            return 9
        else:
            return 5  # Default middle value


class DocumentValidator:
    """
    Validate document extraction quality and completeness for legal documents.
    """
    
    def __init__(self):
        self.min_legal_doc_length = 200
        self.required_legal_elements = [
            'date', 'authority', 'legal_reference'
        ]
    
    def validate_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted document data and return quality report.
        
        Args:
            extracted_data: Result from EnhancedPDFExtractor
            
        Returns:
            Validation report with quality score and recommendations
        """
        text = extracted_data.get('text', '')
        metadata = extracted_data.get('metadata', {})
        
        validation_report = {
            'is_valid': False,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': [],
            'extraction_confidence': extracted_data.get('quality_score', 0.0)
        }
        
        # Check text length
        if len(text) < self.min_legal_doc_length:
            validation_report['issues'].append(f"Document too short: {len(text)} chars")
            validation_report['recommendations'].append("Check if document was properly extracted")
        else:
            validation_report['quality_score'] += 0.3
        
        # Check for legal document characteristics
        legal_score = self._check_legal_characteristics(text)
        validation_report['quality_score'] += legal_score * 0.4
        
        if legal_score < 0.3:
            validation_report['issues'].append("Document may not be a legal document")
            validation_report['recommendations'].append("Verify document type and content")
        
        # Check extraction method reliability
        extraction_method = extracted_data.get('extraction_method')
        if extraction_method == 'ocr':
            ocr_confidence = extracted_data.get('extraction_details', {}).get('avg_confidence', 0)
            if ocr_confidence < 70:
                validation_report['issues'].append(f"Low OCR confidence: {ocr_confidence}%")
                validation_report['recommendations'].append("Consider manual review of OCR results")
            validation_report['quality_score'] += min(ocr_confidence / 100 * 0.3, 0.3)
        else:
            validation_report['quality_score'] += 0.3  # Standard extraction methods are more reliable
        
        # Final validation
        validation_report['is_valid'] = (
            validation_report['quality_score'] > 0.6 and 
            len(validation_report['issues']) == 0
        )
        
        return validation_report
    
    def _check_legal_characteristics(self, text: str) -> float:
        """Check for characteristics typical of legal documents"""
        text_lower = text.lower()
        
        legal_terms = [
            'court', 'judge', 'justice', 'statute', 'law', 'legal',
            'plaintiff', 'defendant', 'case', 'section', 'code',
            'marriage', 'divorce', 'family', 'civil', 'jurisdiction'
        ]
        
        found_terms = sum(1 for term in legal_terms if term in text_lower)
        return min(found_terms / len(legal_terms), 1.0)


# Example usage and testing
if __name__ == "__main__":
    # Initialize extractors
    pdf_extractor = EnhancedPDFExtractor(ocr_enabled=True)
    metadata_extractor = LegalMetadataExtractor()
    validator = DocumentValidator()
    
    # Example usage (would need actual PDF files to test)
    test_file = "sample_marriage_law.pdf"
    
    if os.path.exists(test_file):
        print(f"Processing {test_file}...")
        
        # Extract text
        extraction_result = pdf_extractor.extract_text(test_file)
        print(f"Extraction method: {extraction_result['extraction_method']}")
        print(f"Quality score: {extraction_result['quality_score']:.2f}")
        print(f"Text length: {len(extraction_result['text'])} characters")
        
        # Extract metadata
        legal_metadata = metadata_extractor.extract_metadata(
            extraction_result['text'],
            extraction_result['metadata']
        )
        print(f"Document type: {legal_metadata['document_type']}")
        print(f"Jurisdiction: {legal_metadata['jurisdiction']}")
        print(f"Authority: {legal_metadata['authority']}")
        print(f"Authority level: {legal_metadata['authority_level']}")
        
        # Validate document
        validation = validator.validate_document(extraction_result)
        print(f"Document valid: {validation['is_valid']}")
        print(f"Validation score: {validation['quality_score']:.2f}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")
    else:
        print("No test file found. Create a sample PDF to test the extractors.")