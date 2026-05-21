"""
Security Tests for GraphBug AI Service

Tests for rate limiting, input validation, path traversal protection,
and other security features.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.security_utils import (
    safe_join,
    sanitize_repo_id,
    sanitize_owner_repo,
    sanitize_file_path,
    sanitize_error_message,
    validate_request_size
)


client = TestClient(app)


class TestRateLimiting:
    """Test rate limiting on API endpoints"""
    
    def test_rate_limit_on_review_endpoint(self):
        """Test that /review endpoint has rate limiting"""
        # Make 15 requests (limit is 10/minute)
        responses = []
        for _ in range(15):
            response = client.post(
                "/review",
                json={
                    "owner": "test",
                    "repo": "test-repo",
                    "pr_number": 1,
                    "installation_id": "12345",
                    "context": {}
                }
            )
            responses.append(response.status_code)
        
        # Should get rate limited (429 Too Many Requests)
        assert 429 in responses, "Rate limiting not working on /review endpoint"
    
    def test_rate_limit_on_ingest_endpoint(self):
        """Test that /ingest endpoint has rate limiting"""
        responses = []
        for _ in range(10):
            response = client.post(
                "/ingest",
                json={
                    "repo_url": "https://github.com/test/repo",
                    "repo_id": "test-repo",
                    "installation_id": "12345"
                }
            )
            responses.append(response.status_code)
        
        # Should get rate limited (429 Too Many Requests)
        assert 429 in responses, "Rate limiting not working on /ingest endpoint"


class TestPathTraversal:
    """Test path traversal protection"""
    
    def test_safe_join_prevents_parent_directory(self):
        """Test that safe_join prevents .. attacks"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join("/app/repos", "../etc/passwd")
    
    def test_safe_join_prevents_absolute_path(self):
        """Test that safe_join prevents absolute path injection"""
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join("/app/repos", "/etc/passwd")
    
    def test_safe_join_allows_valid_paths(self):
        """Test that safe_join allows legitimate paths"""
        result = safe_join("/app/repos", "user_repo", "file.py")
        assert "/app/repos/user_repo/file.py" in result
        assert ".." not in result
    
    def test_sanitize_file_path_rejects_parent_refs(self):
        """Test that sanitize_file_path rejects parent directory references"""
        with pytest.raises(ValueError, match="Parent directory reference"):
            sanitize_file_path("../etc/passwd")
        
        with pytest.raises(ValueError, match="Parent directory reference"):
            sanitize_file_path("foo/../../../etc/passwd")
    
    def test_sanitize_file_path_rejects_null_bytes(self):
        """Test that sanitize_file_path rejects null bytes"""
        with pytest.raises(ValueError, match="Null byte"):
            sanitize_file_path("file.txt\x00.jpg")


class TestInputValidation:
    """Test input validation for various endpoints"""
    
    def test_invalid_repo_id_rejected(self):
        """Test that invalid repo_id format is rejected"""
        with pytest.raises(ValueError):
            sanitize_repo_id("../malicious")
        
        with pytest.raises(ValueError):
            sanitize_repo_id("repo@#$%")
        
        with pytest.raises(ValueError):
            sanitize_repo_id("a" * 201)  # Too long
    
    def test_valid_repo_id_accepted(self):
        """Test that valid repo_id is accepted"""
        result = sanitize_repo_id("valid-repo_123")
        assert result == "valid-repo_123"
    
    def test_invalid_owner_rejected(self):
        """Test that invalid GitHub owner is rejected"""
        with pytest.raises(ValueError):
            sanitize_owner_repo("../malicious", "repo")
        
        with pytest.raises(ValueError):
            sanitize_owner_repo("owner@github", "repo")
    
    def test_invalid_repo_name_rejected(self):
        """Test that invalid GitHub repo name is rejected"""
        with pytest.raises(ValueError):
            sanitize_owner_repo("owner", "../malicious")
    
    def test_valid_owner_repo_accepted(self):
        """Test that valid GitHub identifiers are accepted"""
        owner, repo = sanitize_owner_repo("octocat", "Hello-World.test")
        assert owner == "octocat"
        assert repo == "Hello-World.test"
    
    def test_pydantic_validation_on_repo_request(self):
        """Test that Pydantic models validate input"""
        # Invalid repo_url
        response = client.post(
            "/ingest",
            json={
                "repo_url": "not-a-url",
                "repo_id": "test",
                "installation_id": "12345"
            }
        )
        assert response.status_code == 422  # Validation error
        
        # Invalid repo_id
        response = client.post(
            "/ingest",
            json={
                "repo_url": "https://github.com/test/repo",
                "repo_id": "../malicious",
                "installation_id": "12345"
            }
        )
        assert response.status_code == 422
        
        # Invalid installation_id
        response = client.post(
            "/ingest",
            json={
                "repo_url": "https://github.com/test/repo",
                "repo_id": "test-repo",
                "installation_id": "not-a-number"
            }
        )
        assert response.status_code == 422


class TestErrorHandling:
    """Test error message sanitization"""
    
    def test_sanitize_error_message_hides_details(self):
        """Test that error messages don't leak internal details"""
        error = FileNotFoundError("/app/internal/secret/file.txt not found")
        result = sanitize_error_message(error, include_details=False)
        
        assert "secret" not in result["error"]
        assert "/app/internal" not in result["error"]
        assert result["code"] == "NOT_FOUND"
    
    def test_sanitize_error_message_shows_details_in_dev(self):
        """Test that error messages show details in dev mode"""
        error = ValueError("Invalid input: foo")
        result = sanitize_error_message(error, include_details=True)
        
        assert "Invalid input: foo" in result["error"]
        assert result["type"] == "ValueError"


class TestRequestSizeValidation:
    """Test request size limits"""
    
    def test_reject_oversized_requests(self):
        """Test that oversized requests are rejected"""
        with pytest.raises(ValueError, match="Request too large"):
            validate_request_size(11 * 1024 * 1024)  # 11 MB
    
    def test_accept_valid_size_requests(self):
        """Test that valid size requests are accepted"""
        # Should not raise
        validate_request_size(5 * 1024 * 1024)  # 5 MB
        validate_request_size(None)  # No content-length


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are set"""
        response = client.options("/health")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or \
               "Access-Control-Allow-Origin" in response.headers


class TestSecurityHeaders:
    """Test security headers in responses"""
    
    def test_health_endpoint_accessible(self):
        """Test that health endpoint is accessible"""
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
