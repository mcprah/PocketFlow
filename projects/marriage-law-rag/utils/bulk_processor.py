#!/usr/bin/env python3
"""
Bulk Document Processor - Utility for scanning and organizing large document collections
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import mimetypes
from datetime import datetime

logger = logging.getLogger(__name__)


class BulkDocumentProcessor:
    """
    Handles bulk document discovery, categorization, and organization
    for processing large document collections across multiple folders.
    """
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.doc', '.docx', '.txt'}
        self.doc_type_keywords = {
            'law': ['law', 'act', 'statute', 'code', 'legislation'],
            'regulation': ['regulation', 'rule', 'policy', 'guideline'],
            'case': ['case', 'judgment', 'decision', 'ruling', 'precedent'],
            'ordinance': ['ordinance', 'bylaw', 'local'],
            'constitution': ['constitution', 'charter', 'fundamental']
        }
        
    def scan_folders(self, base_path: str, max_files: Optional[int] = None) -> List[Dict]:
        """
        Scan folder structure and categorize documents for processing.
        
        Args:
            base_path: Root directory to scan
            max_files: Maximum number of files to process (None for unlimited)
            
        Returns:
            List of document metadata dictionaries
        """
        if not os.path.exists(base_path):
            raise ValueError(f"Base path does not exist: {base_path}")
            
        documents = []
        file_count = 0
        
        logger.info(f"Starting bulk scan of: {base_path}")
        
        for root, dirs, files in os.walk(base_path):
            folder_name = os.path.basename(root)
            relative_path = os.path.relpath(root, base_path)
            
            # Infer document properties from folder structure
            doc_type = self._infer_doc_type_from_folder(folder_name)
            jurisdiction = self._infer_jurisdiction(root)
            authority = self._infer_authority(root)
            
            for file in files:
                if max_files and file_count >= max_files:
                    logger.info(f"Reached maximum file limit: {max_files}")
                    return documents
                    
                file_path = os.path.join(root, file)
                
                if self._is_supported_file(file_path):
                    try:
                        doc_metadata = {
                            "path": file_path,
                            "filename": file,
                            "doc_type": doc_type,
                            "jurisdiction": jurisdiction,
                            "authority": authority,
                            "folder_category": folder_name,
                            "relative_path": relative_path,
                            "file_size": os.path.getsize(file_path),
                            "last_modified": datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat(),
                            "document_id": self._generate_document_id(file_path),
                            "mime_type": mimetypes.guess_type(file_path)[0]
                        }
                        
                        documents.append(doc_metadata)
                        file_count += 1
                        
                    except OSError as e:
                        logger.warning(f"Could not access file {file_path}: {e}")
                        continue
        
        logger.info(f"Discovered {len(documents)} supported documents")
        return documents
    
    def organize_by_type(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize documents by type for targeted processing.
        
        Args:
            documents: List of document metadata
            
        Returns:
            Dictionary with doc_type as keys and document lists as values
        """
        organized = {}
        
        for doc in documents:
            doc_type = doc.get('doc_type', 'unknown')
            if doc_type not in organized:
                organized[doc_type] = []
            organized[doc_type].append(doc)
            
        return organized
    
    def create_processing_batches(
        self, 
        documents: List[Dict], 
        batch_size: int = 50,
        prioritize_by_size: bool = True
    ) -> List[List[Dict]]:
        """
        Create processing batches optimized for memory and performance.
        
        Args:
            documents: List of document metadata
            batch_size: Number of documents per batch
            prioritize_by_size: Sort by file size to balance batch processing load
            
        Returns:
            List of document batches
        """
        if prioritize_by_size:
            # Sort by file size to balance processing load across batches
            documents = sorted(documents, key=lambda x: x.get('file_size', 0))
        
        batches = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batches.append(batch)
            
        logger.info(f"Created {len(batches)} processing batches")
        return batches
    
    def get_processing_stats(self, documents: List[Dict]) -> Dict:
        """
        Generate statistics about the document collection.
        
        Args:
            documents: List of document metadata
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "total_documents": len(documents),
            "total_size_mb": sum(doc.get('file_size', 0) for doc in documents) / (1024 * 1024),
            "doc_types": {},
            "jurisdictions": {},
            "file_extensions": {}
        }
        
        for doc in documents:
            # Count by document type
            doc_type = doc.get('doc_type', 'unknown')
            stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1
            
            # Count by jurisdiction
            jurisdiction = doc.get('jurisdiction', 'unknown')
            stats["jurisdictions"][jurisdiction] = stats["jurisdictions"].get(jurisdiction, 0) + 1
            
            # Count by file extension
            ext = os.path.splitext(doc.get('filename', ''))[1].lower()
            stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1
        
        return stats
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported for processing."""
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_extensions
    
    def _infer_doc_type_from_folder(self, folder_name: str) -> str:
        """
        Infer document type from folder name using keyword matching.
        
        Args:
            folder_name: Name of the folder containing documents
            
        Returns:
            Inferred document type
        """
        folder_lower = folder_name.lower()
        
        for doc_type, keywords in self.doc_type_keywords.items():
            if any(keyword in folder_lower for keyword in keywords):
                return doc_type
                
        return 'law'  # Default to law documents
    
    def _infer_jurisdiction(self, file_path: str) -> str:
        """
        Infer jurisdiction from file path.
        
        Args:
            file_path: Full path to the document
            
        Returns:
            Inferred jurisdiction
        """
        path_lower = file_path.lower()
        
        # Common jurisdiction indicators
        if 'ghana' in path_lower or 'gh' in path_lower:
            return 'Ghana'
        elif 'federal' in path_lower:
            return 'Federal'
        elif 'state' in path_lower:
            return 'State'
        elif 'local' in path_lower or 'municipal' in path_lower:
            return 'Local'
        
        return 'Ghana'  # Default jurisdiction
    
    def _infer_authority(self, file_path: str) -> Optional[str]:
        """
        Infer issuing authority from file path.
        
        Args:
            file_path: Full path to the document
            
        Returns:
            Inferred authority or None
        """
        path_lower = file_path.lower()
        
        # Common authority indicators
        authorities = {
            'supreme court': 'Supreme Court',
            'high court': 'High Court',
            'parliament': 'Parliament',
            'ministry': 'Ministry',
            'assembly': 'National Assembly',
            'commission': 'Commission'
        }
        
        for keyword, authority in authorities.items():
            if keyword in path_lower:
                return authority
                
        return None
    
    def _generate_document_id(self, file_path: str) -> str:
        """
        Generate unique document ID from file path.
        
        Args:
            file_path: Full path to the document
            
        Returns:
            Unique document identifier
        """
        import hashlib
        
        # Create hash from file path and modification time
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        
        try:
            mtime = str(int(os.path.getmtime(file_path)))
        except OSError:
            mtime = "0"
            
        return f"doc_{path_hash}_{mtime}"


def validate_document_collection(documents: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Validate document collection and filter out invalid entries.
    
    Args:
        documents: List of document metadata
        
    Returns:
        Tuple of (valid_documents, error_messages)
    """
    valid_documents = []
    errors = []
    
    for doc in documents:
        try:
            # Check required fields
            required_fields = ['path', 'filename', 'doc_type']
            missing_fields = [field for field in required_fields if not doc.get(field)]
            
            if missing_fields:
                errors.append(f"Missing fields {missing_fields} in document: {doc.get('path', 'unknown')}")
                continue
            
            # Check file exists
            if not os.path.exists(doc['path']):
                errors.append(f"File does not exist: {doc['path']}")
                continue
                
            # Check file is readable
            if not os.access(doc['path'], os.R_OK):
                errors.append(f"File is not readable: {doc['path']}")
                continue
                
            valid_documents.append(doc)
            
        except Exception as e:
            errors.append(f"Error validating document {doc.get('path', 'unknown')}: {e}")
    
    return valid_documents, errors