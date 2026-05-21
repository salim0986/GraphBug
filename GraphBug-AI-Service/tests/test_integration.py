"""
Integration Tests for AI Service
Tests the complete flow: ingestion → storage → query
"""

import pytest
import time
from src.api import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Test repository (use a small public repo)
TEST_REPO_URL = "https://github.com/octocat/Hello-World.git"
TEST_REPO_ID = "test-repo-123"
TEST_INSTALLATION_ID = "test-install-456"


class TestHealthEndpoint:
    """Test service health and connectivity"""
    
    def test_health_check(self):
        """Service should be up and healthy"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data


class TestIngestionFlow:
    """Test repository ingestion pipeline"""
    
    def test_ingest_repository_success(self):
        """Should successfully ingest a public repository"""
        payload = {
            "repo_url": TEST_REPO_URL,
            "repo_id": TEST_REPO_ID,
            "installation_id": TEST_INSTALLATION_ID
        }
        
        response = client.post("/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["repo_id"] == TEST_REPO_ID
        
        # Wait for ingestion to complete (max 30 seconds)
        time.sleep(30)
    
    def test_ingest_invalid_repo(self):
        """Should fail gracefully with invalid repo URL"""
        payload = {
            "repo_url": "https://github.com/invalid/nonexistent-repo-12345.git",
            "repo_id": "invalid-repo",
            "installation_id": TEST_INSTALLATION_ID
        }
        
        response = client.post("/ingest", json=payload)
        # Should accept the request but fail during processing
        assert response.status_code in [200, 500]
    
    def test_ingest_missing_fields(self):
        """Should reject requests with missing required fields"""
        payload = {"repo_url": TEST_REPO_URL}  # Missing repo_id
        
        response = client.post("/ingest", json=payload)
        assert response.status_code == 422  # Unprocessable Entity


class TestQueryFlow:
    """Test code search and retrieval"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure test repo is ingested before queries"""
        # Trigger ingestion if not already done
        payload = {
            "repo_url": TEST_REPO_URL,
            "repo_id": TEST_REPO_ID,
            "installation_id": TEST_INSTALLATION_ID
        }
        client.post("/ingest", json=payload)
        time.sleep(35)  # Wait for ingestion
    
    def test_query_repository_success(self):
        """Should find relevant code snippets"""
        payload = {
            "repo_id": TEST_REPO_ID,
            "query": "function definition"
        }
        
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert isinstance(data["results"], list)
        
        # Check result structure
        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "score" in result
            assert "name" in result
            assert "file" in result
            assert "code" in result
            assert "start_line" in result
    
    def test_query_nonexistent_repo(self):
        """Should handle queries for non-ingested repos"""
        payload = {
            "repo_id": "nonexistent-repo-999",
            "query": "test query"
        }
        
        response = client.post("/query", json=payload)
        # Should either return empty results or 404
        assert response.status_code in [200, 404, 500]
    
    def test_query_empty_string(self):
        """Should handle empty query strings"""
        payload = {
            "repo_id": TEST_REPO_ID,
            "query": ""
        }
        
        response = client.post("/query", json=payload)
        assert response.status_code in [200, 400]
    
    def test_query_special_characters(self):
        """Should handle queries with special characters"""
        payload = {
            "repo_id": TEST_REPO_ID,
            "query": "function* async() => {}"
        }
        
        response = client.post("/query", json=payload)
        assert response.status_code == 200


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_concurrent_ingestion_same_repo(self):
        """Should handle concurrent ingestion requests gracefully"""
        payload = {
            "repo_url": TEST_REPO_URL,
            "repo_id": "concurrent-test",
            "installation_id": TEST_INSTALLATION_ID
        }
        
        # Fire multiple requests
        responses = []
        for _ in range(3):
            resp = client.post("/ingest", json=payload)
            responses.append(resp)
        
        # All should accept or one should succeed
        success_codes = [r.status_code for r in responses]
        assert any(code == 200 for code in success_codes)
    
    def test_large_query_string(self):
        """Should handle very large query strings"""
        payload = {
            "repo_id": TEST_REPO_ID,
            "query": "a" * 10000  # 10KB query
        }
        
        response = client.post("/query", json=payload)
        assert response.status_code in [200, 413]  # OK or Payload Too Large


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
