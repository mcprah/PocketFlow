#!/usr/bin/env python3
"""
Test the /query endpoint directly
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from main import app
from api.models import QueryRequest


class TestQueryEndpoint:
    """Test the /query endpoint functionality"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        
        # Mock the application state
        app.state.vector_store = Mock()
        app.state.online_flow = Mock()
    
    def test_query_endpoint_basic(self):
        """Test basic query functionality"""
        # Mock the flow execution to return expected results
        mock_shared_result = {
            "query_results": [
                {
                    "chunk_id": "test_chunk_1",
                    "document_id": "test_doc_1",
                    "text": "Marriage laws require proper documentation.",
                    "similarity_score": 0.85,
                    "document_metadata": {
                        "document_id": "test_doc_1",
                        "title": "Test Marriage Document",
                        "doc_type": "statute",
                        "jurisdiction": "federal",
                        "authority": "Federal Court",
                        "authority_level": 10,
                        "date_issued": "2023-01-01",
                        "file_path": "/test/path"
                    },
                    "legal_context": {
                        "section": "Section 1",
                        "legal_concepts": ["marriage", "documentation"],
                        "citations": ["Test v. Case"],
                        "case_number": "123-456",
                        "court": "Federal Court"
                    }
                }
            ],
            "generated_answer": "Marriage laws typically require proper documentation and filing.",
            "confidence_score": 0.85,
            "citations": ["Test v. Case"]
        }
        
        def mock_flow_run(shared):
            shared.update(mock_shared_result)
            
        app.state.online_flow.run = Mock(side_effect=mock_flow_run)
        
        # Test the query
        response = self.client.post("/query", json={
            "question": "What are the requirements for marriage?",
            "max_results": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["query"] == "What are the requirements for marriage?"
        assert len(data["results"]) == 1
        assert data["generated_answer"] == "Marriage laws typically require proper documentation and filing."
        assert data["confidence_score"] == 0.85
        assert "processing_time_ms" in data
    
    def test_query_validation_too_short(self):
        """Test query validation for too short queries"""
        response = self.client.post("/query", json={
            "question": "Hi"  # Only 2 characters
        })
        
        # FastAPI returns 422 for validation errors, but our endpoint logic returns 400
        # Let's test that the endpoint logic catches this
        if response.status_code == 422:
            # Pydantic validation error
            assert "at least 3 characters" in str(response.json())
        else:
            assert response.status_code == 400
            assert "Query must be at least 3 characters long" in response.json()["detail"]
    
    def test_query_validation_empty(self):
        """Test query validation for empty queries"""
        response = self.client.post("/query", json={
            "question": ""
        })
        
        # Should return 422 for validation error or 400 for our custom validation
        assert response.status_code in [400, 422]
    
    def test_query_with_filters(self):
        """Test query with jurisdiction and document type filters"""
        def mock_flow_run(shared):
            # Verify filters are passed through
            assert shared["jurisdiction_filter"] == ["federal"]
            assert shared["doc_type_filter"] == ["statute"]
            assert shared["authority_level_min"] == 8
            assert shared["max_results"] == 10
            
            shared.update({
                "query_results": [],
                "generated_answer": "No relevant documents found.",
                "confidence_score": 0.0,
                "citations": []
            })
            
        app.state.online_flow.run = Mock(side_effect=mock_flow_run)
        
        response = self.client.post("/query", json={
            "question": "What are marriage requirements?",
            "jurisdiction_filter": ["federal"],
            "doc_type_filter": ["statute"],
            "authority_level_min": 8,
            "max_results": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["generated_answer"] == "No relevant documents found."
    
    def test_query_flow_failure(self):
        """Test handling of flow execution failures"""
        app.state.online_flow.run = Mock(side_effect=Exception("Flow failed"))
        
        response = self.client.post("/query", json={
            "question": "What are marriage requirements?"
        })
        
        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]
    
    def test_query_no_results(self):
        """Test query when no results are found"""
        def mock_flow_run(shared):
            shared.update({
                "query_results": [],
                "generated_answer": "I couldn't find relevant information to answer your question.",
                "confidence_score": 0.0,
                "citations": []
            })
            
        app.state.online_flow.run = Mock(side_effect=mock_flow_run)
        
        response = self.client.post("/query", json={
            "question": "What about alien marriage laws?"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 0
        assert data["confidence_score"] == 0.0
        assert "couldn't find relevant information" in data["generated_answer"]


def test_run_query_endpoint():
    """Simple function to run a basic test"""
    client = TestClient(app)
    
    # Mock the application state
    app.state.vector_store = Mock()
    app.state.online_flow = Mock()
    
    # Mock successful flow execution
    def mock_flow_run(shared):
        shared.update({
            "query_results": [{
                "chunk_id": "test_chunk",
                "document_id": "test_doc",
                "text": "Test legal content",
                "similarity_score": 0.8,
                "document_metadata": {
                    "document_id": "test_doc",
                    "title": "Test Document",
                    "doc_type": "statute",
                    "jurisdiction": "federal",
                    "authority": "Test Authority",
                    "authority_level": 8,
                    "date_issued": "2023-01-01",
                    "file_path": "/test/path"
                },
                "legal_context": {
                    "section": "Section 1",
                    "legal_concepts": ["test"],
                    "citations": ["Test Case"],
                    "case_number": "123",
                    "court": "Test Court"
                }
            }],
            "generated_answer": "Test answer",
            "confidence_score": 0.8,
            "citations": []
        })
    
    app.state.online_flow.run = Mock(side_effect=mock_flow_run)
    
    # Test the endpoint
    response = client.post("/query", json={
        "question": "Test question about marriage"
    })
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200


if __name__ == "__main__":
    # Run a simple test
    test_run_query_endpoint()
    print("✅ Basic query endpoint test passed!")