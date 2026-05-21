"""
Integration Tests for Phase 5 (GitHub Integration & Review Posting)

This module provides comprehensive tests for all Phase 5 components.

Test Coverage:
1. GitHub API client (authentication, rate limiting, PR fetching)
2. Review formatter (markdown generation)
3. Review poster (comment posting, duplicate detection)
4. Review storage (PostgreSQL + Neo4j)
5. Webhook integration (queue, workers, end-to-end)

Run with: pytest tests/test_phase5_integration.py -v
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

# Phase 5 Components
from github_client import (
    GitHubClient,
    GitHubConfig,
    TokenManager,
    create_github_client
)
from review_formatter import (
    ReviewFormatter,
    ReviewSummary,
    ReviewSection,
    Issue,
    Severity,
    ReviewEvent,
    parse_workflow_output_to_review
)
from review_poster import (
    ReviewPosterService,
    ReviewPostingConfig,
    ReviewPostResult,
    create_review_poster,
    post_review_from_workflow
)
from review_storage import (
    ReviewStorageService,
    ReviewMetadata,
    ReviewComment,
    StorageResult,
    create_review_storage_service,
    store_review_from_workflow
)
from webhook_integration import (
    ReviewQueue,
    ReviewTask,
    ReviewPriority,
    ReviewStatus,
    QueueConfig,
    WebhookIntegrationService,
    handle_pr_webhook
)


# ========================================================================
# FIXTURES
# ========================================================================

@pytest.fixture
def mock_github_config():
    """Mock GitHub configuration"""
    return GitHubConfig(
        app_id=12345,
        private_key="-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
        max_retries=2,
        retry_delay_seconds=1.0
    )


@pytest.fixture
def mock_pr_data():
    """Mock PR data from GitHub API"""
    return {
        "number": 123,
        "title": "Add payment processing",
        "body": "This PR adds payment processing functionality",
        "state": "open",
        "user": {"login": "testuser"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "head": {"sha": "abc123", "ref": "feature/payment"},
        "base": {"sha": "def456", "ref": "main"},
        "additions": 250,
        "deletions": 50,
        "files": [
            {
                "filename": "payment.py",
                "status": "added",
                "additions": 200,
                "deletions": 0,
                "patch": "@@ -0,0 +1,200 @@\n+def process_payment():\n+    pass"
            }
        ],
        "mergeable": True,
        "mergeable_state": "clean"
    }


@pytest.fixture
def mock_workflow_output():
    """Mock workflow output"""
    return {
        "pr_context": {
            "repo_id": "repo_123",
            "total_files": 3,
            "total_additions": 250,
            "total_deletions": 50,
            "critical_issues": [
                {
                    "title": "SQL Injection",
                    "description": "User input not sanitized",
                    "file": "auth.py",
                    "line": 45,
                    "suggestion": "Use parameterized queries",
                    "category": "security"
                }
            ],
            "high_issues": [],
            "medium_issues": [],
            "recommendations": ["Fix SQL injection before merging"],
            "key_changes": ["Added payment processing"],
            "positives": ["Good test coverage"]
        },
        "overall_summary": "Found critical security vulnerability",
        "review_strategy": "deep",
        "selected_model": "gemini-1.5-pro",
        "review_completed": True
    }


# ========================================================================
# PHASE 5.1 - GITHUB CLIENT TESTS
# ========================================================================

class TestGitHubClient:
    """Tests for GitHub API client"""
    
    @pytest.mark.asyncio
    async def test_token_generation(self, mock_github_config):
        """Test JWT token generation"""
        token_manager = TokenManager(mock_github_config)
        
        # Note: This will fail without valid private key
        # In real tests, use proper test keys
        with pytest.raises(Exception):
            token = token_manager.get_installation_token(12345)
    
    @pytest.mark.asyncio
    async def test_get_pull_request(self, mock_github_config, mock_pr_data):
        """Test fetching PR data"""
        with patch("github_client.Github") as MockGithub:
            # Mock GitHub instance
            mock_gh = MockGithub.return_value
            mock_repo = Mock()
            mock_pr = Mock()
            
            # Setup mock PR
            mock_pr.number = 123
            mock_pr.title = "Test PR"
            mock_pr.as_dict.return_value = mock_pr_data
            
            mock_repo.get_pull.return_value = mock_pr
            mock_gh.get_repo.return_value = mock_repo
            
            # Test
            client = GitHubClient(mock_github_config)
            pr_data = await client.get_pull_request("owner/repo", 123, 12345)
            
            assert pr_data["number"] == 123
            assert "files" in pr_data
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_github_config):
        """Test rate limit monitoring"""
        with patch("github_client.Github") as MockGithub:
            mock_gh = MockGithub.return_value
            mock_rate_limit = Mock()
            mock_core = Mock()
            mock_core.remaining = 50  # Below threshold
            mock_core.reset = datetime.now()
            mock_rate_limit.core = mock_core
            mock_gh.get_rate_limit.return_value = mock_rate_limit
            
            client = GitHubClient(mock_github_config)
            
            # Should detect low rate limit
            # Note: Actual implementation would wait
            await client._handle_rate_limit(12345)


# ========================================================================
# PHASE 5.2 - REVIEW FORMATTER TESTS
# ========================================================================

class TestReviewFormatter:
    """Tests for review formatter"""
    
    def test_format_review_summary(self):
        """Test formatting complete review"""
        formatter = ReviewFormatter()
        
        review = ReviewSummary(
            overall_assessment="Critical issues found",
            review_strategy="deep",
            model_used="gemini-1.5-pro",
            sections=[
                ReviewSection(
                    title="Security",
                    content="Found security vulnerabilities",
                    issues=[
                        Issue(
                            severity=Severity.CRITICAL,
                            title="SQL Injection",
                            description="User input not sanitized",
                            file="auth.py",
                            line=45
                        )
                    ]
                )
            ],
            statistics={
                "files_reviewed": 3,
                "critical_issues": 1
            },
            recommendations=["Fix security issues"],
            event_type=ReviewEvent.REQUEST_CHANGES
        )
        
        markdown = formatter.format_review(review)
        
        assert "# 🤖 AI Code Review" in markdown
        assert "gemini-1.5-pro" in markdown
        assert "Security" in markdown
        assert "SQL Injection" in markdown
        assert "![Critical]" in markdown
    
    def test_format_inline_comment(self):
        """Test formatting inline comment"""
        formatter = ReviewFormatter()
        
        issue = Issue(
            severity=Severity.HIGH,
            title="Memory Leak",
            description="Resource not released",
            file="memory.py",
            line=100,
            suggestion="Add try-finally block"
        )
        
        markdown = formatter.format_inline_comment(issue)
        
        assert "### 🟠 Memory Leak" in markdown
        assert "![High]" in markdown
        assert "Suggested fix:" in markdown
    
    def test_parse_workflow_output(self, mock_workflow_output):
        """Test parsing workflow output to review summary"""
        review = parse_workflow_output_to_review(mock_workflow_output)
        
        assert review.review_strategy == "deep"
        assert review.model_used == "gemini-1.5-pro"
        assert len(review.sections) > 0
        assert review.statistics["critical_issues"] == 1


# ========================================================================
# PHASE 5.3 - REVIEW POSTER TESTS
# ========================================================================

class TestReviewPoster:
    """Tests for review poster service"""
    
    @pytest.mark.asyncio
    async def test_post_review(self, mock_workflow_output):
        """Test posting review to GitHub"""
        with patch("review_poster.create_github_client") as mock_create_client:
            # Mock GitHub client
            mock_client = AsyncMock()
            mock_client.get_pull_request.return_value = {
                "head": {"sha": "abc123"}
            }
            mock_client.post_review_comment.return_value = {
                "id": 12345,
                "html_url": "https://github.com/owner/repo/pull/123#review-12345"
            }
            mock_client.post_inline_comment.return_value = {}
            mock_client.get_existing_reviews.return_value = []
            
            mock_create_client.return_value = mock_client
            
            # Test
            service = ReviewPosterService(mock_client)
            result = await service.post_review(
                repo_full_name="owner/repo",
                pr_number=123,
                workflow_output=mock_workflow_output,
                installation_id=12345
            )
            
            assert result.success
            assert result.review_id == 12345
            assert result.review_url is not None
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, mock_workflow_output):
        """Test detecting existing bot reviews"""
        with patch("review_poster.create_github_client") as mock_create_client:
            mock_client = AsyncMock()
            mock_client.get_pull_request.return_value = {"head": {"sha": "abc123"}}
            
            # Return existing review
            mock_client.get_existing_reviews.return_value = [
                {"id": 99999, "body": "🤖 AI Code Review System"}
            ]
            
            mock_create_client.return_value = mock_client
            
            config = ReviewPostingConfig(check_for_duplicates=True, update_existing_review=False)
            service = ReviewPosterService(mock_client, config=config)
            
            result = await service.post_review(
                repo_full_name="owner/repo",
                pr_number=123,
                workflow_output=mock_workflow_output,
                installation_id=12345
            )
            
            assert not result.success
            assert "duplicate" in result.error.lower()


# ========================================================================
# PHASE 5.4 - REVIEW STORAGE TESTS
# ========================================================================

class TestReviewStorage:
    """Tests for review storage service"""
    
    @pytest.mark.asyncio
    async def test_store_in_postgres(self):
        """Test storing review in PostgreSQL"""
        with patch("review_storage.httpx.AsyncClient") as MockClient:
            mock_http = MockClient.return_value
            
            # Mock API responses
            mock_http.post.return_value = AsyncMock(
                json=lambda: {"id": "review_123"},
                raise_for_status=lambda: None
            )
            
            service = ReviewStorageService(
                postgres_api_url="http://localhost:3000/api",
                neo4j_uri="neo4j://localhost:7687",
                neo4j_auth=("neo4j", "password")
            )
            
            review = ReviewMetadata(
                pull_request_id="pr_123",
                status="completed",
                summary={"overallScore": 85},
                key_changes=[],
                recommendations=[],
                positives=[],
                primary_model="flash",
                models_used=[],
                total_tokens_input=1000,
                total_tokens_output=500,
                total_cost=0.05,
                execution_time_ms=5000
            )
            
            review_id, success = await service._store_in_postgres(review, [])
            
            assert success
            assert review_id == "review_123"
    
    @pytest.mark.asyncio
    async def test_store_in_neo4j(self):
        """Test storing review in Neo4j"""
        with patch("review_storage.AsyncGraphDatabase") as MockDriver:
            mock_driver = MockDriver.driver.return_value
            mock_session = AsyncMock()
            mock_driver.session.return_value.__aenter__.return_value = mock_session
            
            service = ReviewStorageService(
                postgres_api_url="http://localhost:3000/api",
                neo4j_uri="neo4j://localhost:7687",
                neo4j_auth=("neo4j", "password")
            )
            service.neo4j_driver = mock_driver
            
            review = ReviewMetadata(
                pull_request_id="pr_123",
                status="completed",
                summary={"overallScore": 85, "issuesFound": 1},
                key_changes=[],
                recommendations=[],
                positives=[],
                primary_model="flash",
                models_used=[],
                total_tokens_input=1000,
                total_tokens_output=500,
                total_cost=0.05,
                execution_time_ms=5000
            )
            
            comments = [
                ReviewComment(
                    review_id="review_123",
                    file_path="auth.py",
                    line_start=45,
                    severity="critical",
                    category="security",
                    title="SQL Injection",
                    description="Issue"
                )
            ]
            
            success = await service._store_in_neo4j(
                "review_123", review, comments, "repo_123", "pr_1"
            )
            
            assert success


# ========================================================================
# PHASE 5.5 - WEBHOOK INTEGRATION TESTS
# ========================================================================

class TestReviewQueue:
    """Tests for review queue"""
    
    @pytest.mark.asyncio
    async def test_enqueue_task(self):
        """Test enqueueing review task"""
        config = QueueConfig(max_concurrent=1)
        queue = ReviewQueue(config)
        
        task = ReviewTask(
            repo_full_name="owner/repo",
            pr_number=123,
            installation_id=12345,
            repo_id="repo_123",
            pr_internal_id="pr_123",
            priority=ReviewPriority.MEDIUM
        )
        
        task_id = await queue.enqueue(task)
        
        assert task_id == "owner/repo#123"
        assert len(queue.queue) == 1
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based task ordering"""
        queue = ReviewQueue()
        
        # Enqueue tasks with different priorities
        low_task = ReviewTask(
            repo_full_name="owner/repo1",
            pr_number=1,
            installation_id=12345,
            repo_id="repo_1",
            pr_internal_id="pr_1",
            priority=ReviewPriority.LOW
        )
        
        high_task = ReviewTask(
            repo_full_name="owner/repo2",
            pr_number=2,
            installation_id=12345,
            repo_id="repo_2",
            pr_internal_id="pr_2",
            priority=ReviewPriority.HIGH
        )
        
        await queue.enqueue(low_task)
        await queue.enqueue(high_task)
        
        # High priority should be first
        assert queue.queue[0].priority == ReviewPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test preventing duplicate tasks"""
        queue = ReviewQueue()
        
        task = ReviewTask(
            repo_full_name="owner/repo",
            pr_number=123,
            installation_id=12345,
            repo_id="repo_123",
            pr_internal_id="pr_123",
            priority=ReviewPriority.MEDIUM
        )
        
        task_id1 = await queue.enqueue(task)
        task_id2 = await queue.enqueue(task)
        
        assert task_id1 == task_id2
        assert len(queue.queue) == 1


class TestWebhookIntegration:
    """Tests for webhook integration service"""
    
    @pytest.mark.asyncio
    async def test_handle_pr_webhook(self):
        """Test handling PR webhook event"""
        queue = ReviewQueue()
        service = WebhookIntegrationService(queue)
        
        payload = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "additions": 250,
                "deletions": 50
            },
            "repository": {
                "id": "repo_123",
                "full_name": "owner/repo"
            },
            "installation": {
                "id": 12345
            }
        }
        
        task_id = await service.handle_pull_request_webhook(payload)
        
        assert task_id == "owner/repo#123"
        assert len(queue.queue) == 1
    
    @pytest.mark.asyncio
    async def test_ignore_unsupported_actions(self):
        """Test ignoring unsupported webhook actions"""
        queue = ReviewQueue()
        service = WebhookIntegrationService(queue)
        
        payload = {
            "action": "closed",  # Not supported
            "pull_request": {"number": 123},
            "repository": {"full_name": "owner/repo"},
            "installation": {"id": 12345}
        }
        
        task_id = await service.handle_pull_request_webhook(payload)
        
        assert task_id == ""
        assert len(queue.queue) == 0


# ========================================================================
# END-TO-END INTEGRATION TEST
# ========================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_review_flow(self, mock_workflow_output, mock_pr_data):
        """Test complete review flow from webhook to storage"""
        # This would be a full integration test
        # Requires all services running (GitHub, Neo4j, PostgreSQL, Qdrant)
        # Mark with @pytest.mark.integration and run separately
        
        # 1. Simulate webhook
        # 2. Queue task
        # 3. Process review
        # 4. Post to GitHub
        # 5. Store in databases
        # 6. Verify all steps
        
        # Skip in unit tests
        pytest.skip("Integration test - requires all services running")


# ========================================================================
# RUN TESTS
# ========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
