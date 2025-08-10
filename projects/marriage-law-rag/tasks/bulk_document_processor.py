#!/usr/bin/env python3
"""
Bulk Document Processor - Background task for processing large document collections
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Optional, Callable
from datetime import datetime

# Add the parent directory to Python path to access project modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.bulk_processor import BulkDocumentProcessor, validate_document_collection
from flows import create_bulk_indexing_flow

logger = logging.getLogger(__name__)


class BulkProcessingTracker:
    """
    Tracks progress and state for bulk document processing operations.
    Provides real-time updates via WebSocket and handles error recovery.
    """
    
    def __init__(self, operation_id: str, broadcast_function: Optional[Callable] = None):
        self.operation_id = operation_id
        self.broadcast_function = broadcast_function
        self.start_time = datetime.now()
        self.processed_documents = 0
        self.total_documents = 0
        self.failed_documents = []
        self.current_batch = 0
        self.total_batches = 0
        self.current_stage = "initializing"
        
    async def update_progress(
        self, 
        processed: int = None,
        total: int = None,
        stage: str = None,
        batch_info: dict = None,
        error: str = None
    ):
        """Update processing progress and broadcast to connected clients."""
        
        if processed is not None:
            self.processed_documents = processed
        if total is not None:
            self.total_documents = total
        if stage is not None:
            self.current_stage = stage
        if batch_info:
            self.current_batch = batch_info.get('current_batch', self.current_batch)
            self.total_batches = batch_info.get('total_batches', self.total_batches)
            
        progress_percent = 0
        if self.total_documents > 0:
            progress_percent = (self.processed_documents / self.total_documents) * 100
            
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        progress_data = {
            "type": "bulk_processing_progress",
            "operation_id": self.operation_id,
            "stage": self.current_stage,
            "processed": self.processed_documents,
            "total": self.total_documents,
            "progress_percent": round(progress_percent, 2),
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "elapsed_seconds": round(elapsed_time, 1),
            "failed_count": len(self.failed_documents),
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            progress_data["error"] = error
            progress_data["type"] = "bulk_processing_error"
            
        if self.broadcast_function:
            await self.broadcast_function(progress_data)
            
        logger.info(f"Bulk processing progress: {progress_percent:.1f}% ({self.processed_documents}/{self.total_documents})")
    
    def add_failed_document(self, document: dict, error: str):
        """Record a failed document for later retry."""
        self.failed_documents.append({
            "document": document,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
    async def complete(self, success: bool = True):
        """Mark bulk processing as complete."""
        completion_data = {
            "type": "bulk_processing_complete" if success else "bulk_processing_failed",
            "operation_id": self.operation_id,
            "total_processed": self.processed_documents,
            "total_failed": len(self.failed_documents),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }
        
        if not success:
            completion_data["failed_documents"] = self.failed_documents
            
        if self.broadcast_function:
            await self.broadcast_function(completion_data)


async def process_bulk_documents(
    folder_path: str,
    operation_id: str,
    vector_store,
    broadcast_function: Optional[Callable] = None,
    database_url: str = None,
    max_files: Optional[int] = None,
    batch_size: int = 50,
    max_workers: Optional[Dict] = None
):
    """
    Process a large collection of documents from a folder structure.
    
    Args:
        folder_path: Root folder containing documents to process
        operation_id: Unique identifier for this bulk operation
        vector_store: Vector store instance for database operations
        broadcast_function: Function to broadcast WebSocket messages
        database_url: Database connection string
        max_files: Maximum number of files to process (None for unlimited)
        batch_size: Number of documents per processing batch
        max_workers: Worker limits for parallel processing stages
    """
    
    tracker = BulkProcessingTracker(operation_id, broadcast_function)
    
    try:
        # Initialize bulk document processor
        bulk_processor = BulkDocumentProcessor()
        
        # Stage 1: Scan folder structure
        await tracker.update_progress(stage="scanning")
        logger.info(f"Starting bulk scan of folder: {folder_path}")
        
        documents = bulk_processor.scan_folders(folder_path, max_files=max_files)
        
        if not documents:
            await tracker.update_progress(error="No supported documents found in the specified folder")
            await tracker.complete(success=False)
            return
            
        # Validate discovered documents
        valid_documents, validation_errors = validate_document_collection(documents)
        
        if validation_errors:
            logger.warning(f"Found {len(validation_errors)} validation errors")
            for error in validation_errors[:5]:  # Log first 5 errors
                logger.warning(f"Validation error: {error}")
        
        if not valid_documents:
            await tracker.update_progress(error="No valid documents found after validation")
            await tracker.complete(success=False)
            return
            
        # Log discovery statistics
        stats = bulk_processor.get_processing_stats(valid_documents)
        logger.info(f"Bulk processing stats: {stats}")
        
        await tracker.update_progress(
            total=len(valid_documents),
            stage="organizing"
        )
        
        # Stage 2: Organize into processing batches
        batches = bulk_processor.create_processing_batches(
            valid_documents, 
            batch_size=batch_size,
            prioritize_by_size=True
        )
        
        await tracker.update_progress(
            stage="processing",
            batch_info={
                "total_batches": len(batches),
                "current_batch": 0
            }
        )
        
        # Stage 3: Create bulk processing flow
        bulk_flow = create_bulk_indexing_flow(max_workers=max_workers)
        
        # Stage 4: Process batches
        processed_count = 0
        
        for batch_idx, batch in enumerate(batches):
            try:
                await tracker.update_progress(
                    stage="processing",
                    batch_info={
                        "current_batch": batch_idx + 1,
                        "total_batches": len(batches)
                    }
                )
                
                # Prepare shared state for batch processing
                shared = {
                    "raw_documents": batch,
                    "db_connection_string": database_url,
                    "vector_store": vector_store,
                    "batch_info": {
                        "batch_index": batch_idx,
                        "batch_size": len(batch),
                        "total_batches": len(batches)
                    }
                }
                
                # Process batch using bulk flow
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} documents)")
                await bulk_flow.run_async(shared)
                
                # Update progress
                processed_count += len(batch)
                await tracker.update_progress(processed=processed_count)
                
                # Track any failed documents from this batch
                if 'failed_documents' in shared:
                    for failed_doc in shared['failed_documents']:
                        tracker.add_failed_document(
                            failed_doc['document'], 
                            failed_doc.get('error', 'Unknown error')
                        )
                
                # Brief pause between batches to avoid overwhelming the system
                await asyncio.sleep(0.5)
                
            except Exception as batch_error:
                logger.error(f"Error processing batch {batch_idx + 1}: {batch_error}")
                
                # Mark all documents in this batch as failed
                for doc in batch:
                    tracker.add_failed_document(doc, str(batch_error))
                
                # Continue processing remaining batches
                continue
        
        # Stage 5: Completion
        await tracker.update_progress(stage="completed")
        await tracker.complete(success=True)
        
        logger.info(f"Bulk processing completed. Processed: {processed_count}, Failed: {len(tracker.failed_documents)}")
        
    except Exception as e:
        logger.error(f"Bulk processing failed for operation {operation_id}: {e}")
        await tracker.update_progress(
            stage="failed",
            error=str(e)
        )
        await tracker.complete(success=False)


async def retry_failed_documents(
    failed_documents: List[Dict],
    operation_id: str,
    vector_store,
    broadcast_function: Optional[Callable] = None,
    database_url: str = None,
    max_workers: Optional[Dict] = None
):
    """
    Retry processing of previously failed documents.
    
    Args:
        failed_documents: List of failed document entries with error information
        operation_id: Unique identifier for this retry operation
        vector_store: Vector store instance
        broadcast_function: Function to broadcast WebSocket messages
        database_url: Database connection string
        max_workers: Worker limits for parallel processing
    """
    
    if not failed_documents:
        logger.info("No failed documents to retry")
        return
        
    tracker = BulkProcessingTracker(f"{operation_id}_retry", broadcast_function)
    
    try:
        # Extract document metadata from failed entries
        documents_to_retry = [entry['document'] for entry in failed_documents]
        
        await tracker.update_progress(
            total=len(documents_to_retry),
            stage="retrying"
        )
        
        # Create smaller batches for retry (more conservative approach)
        batch_size = 10
        batches = [documents_to_retry[i:i + batch_size] 
                  for i in range(0, len(documents_to_retry), batch_size)]
        
        bulk_flow = create_bulk_indexing_flow(max_workers=max_workers)
        processed_count = 0
        
        for batch_idx, batch in enumerate(batches):
            try:
                await tracker.update_progress(
                    stage="retrying",
                    batch_info={
                        "current_batch": batch_idx + 1,
                        "total_batches": len(batches)
                    }
                )
                
                shared = {
                    "raw_documents": batch,
                    "db_connection_string": database_url,
                    "vector_store": vector_store,
                    "retry_mode": True,
                    "batch_info": {
                        "batch_index": batch_idx,
                        "batch_size": len(batch),
                        "total_batches": len(batches)
                    }
                }
                
                await bulk_flow.run_async(shared)
                processed_count += len(batch)
                await tracker.update_progress(processed=processed_count)
                
                # Longer pause between retry batches
                await asyncio.sleep(2.0)
                
            except Exception as batch_error:
                logger.error(f"Retry batch {batch_idx + 1} failed: {batch_error}")
                for doc in batch:
                    tracker.add_failed_document(doc, str(batch_error))
        
        await tracker.complete(success=True)
        logger.info(f"Retry processing completed. Processed: {processed_count}, Still failed: {len(tracker.failed_documents)}")
        
    except Exception as e:
        logger.error(f"Retry processing failed: {e}")
        await tracker.complete(success=False)


def estimate_processing_time(documents: List[Dict]) -> Dict:
    """
    Estimate processing time and resource requirements for document collection.
    
    Args:
        documents: List of document metadata
        
    Returns:
        Dictionary with time and resource estimates
    """
    
    total_size_mb = sum(doc.get('file_size', 0) for doc in documents) / (1024 * 1024)
    doc_count = len(documents)
    
    # Rough estimates based on typical processing rates
    estimates = {
        "document_count": doc_count,
        "total_size_mb": round(total_size_mb, 2),
        "estimated_minutes": {
            "extraction": round(doc_count * 0.5),  # ~30 seconds per document
            "chunking": round(doc_count * 0.1),    # ~6 seconds per document  
            "embedding": round(doc_count * 2.0),   # ~2 minutes per document (API limited)
            "storage": round(doc_count * 0.1),     # ~6 seconds per document
        },
        "estimated_total_minutes": round(doc_count * 2.7),  # Conservative estimate
        "recommended_batch_size": min(50, max(10, doc_count // 10)),
        "api_calls_estimate": doc_count * 5  # Rough estimate for OpenAI API calls
    }
    
    return estimates