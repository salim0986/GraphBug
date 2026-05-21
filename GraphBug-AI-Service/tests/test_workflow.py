"""
Workflow Integration Tests (Phase 4.6)

Tests for LangGraph code review workflow with different PR scenarios:
- Small PRs (quick review strategy)
- Medium PRs (standard review strategy)
- Large/complex PRs (deep review strategy)
- Error handling and retry logic
- State transitions and validation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from src.workflow import (
    ReviewState,
    WorkflowConfig,
    CodeReviewWorkflow,
    create_review_workflow
)
from src.context_builder import ContextBuilder
from src.gemini_client import GeminiClient


# ========================================================================
# FIXTURES
# ========================================================================

@pytest.fixture
def workflow_config():
    """Create test workflow configuration"""
    return WorkflowConfig(
        quick_review_max_files=3,
        quick_review_max_additions=100,
        standard_review_max_files=10,
        standard_review_max_additions=500,
        high_priority_complexity=70,
        high_priority_issues=3,
        max_files_per_batch=5,
        flash_lite_model="gemini-2.0-flash-exp",
        flash_model="gemini-2.0-flash-exp",
        pro_model="gemini-1.5-pro",
        max_retries=3,
        retry_delay_seconds=2.0
    )


@pytest.fixture
def mock_context_builder():
    """Mock ContextBuilder for testing"""
    mock = AsyncMock(spec=ContextBuilder)
    
    # Default PR context response
    mock.build_pr_context.return_value = {
        "repo_id": "test-repo",
        "pr_number": 123,
        "total_files": 2,
        "total_additions": 50,
        "total_deletions": 10,
        "languages": ["python"],
        "risk_level": "low",
        "requires_deep_review": False,
        "files": [
            {
                "filename": "test.py",
                "language": "python",
                "additions": 30,
                "deletions": 5,
                "complexity_score": 45,
                "issues_summary": {"critical": 0, "high": 0, "medium": 1},
                "dependencies": [],
                "similar_files": []
            },
            {
                "filename": "utils.py",
                "language": "python",
                "additions": 20,
                "deletions": 5,
                "complexity_score": 30,
                "issues_summary": {"critical": 0, "high": 0, "medium": 0},
                "dependencies": [],
                "similar_files": []
            }
        ],
        "critical_issues": [],
        "high_issues": [],
        "medium_issues": [
            {"file": "test.py", "line": 10, "description": "Complex function"}
        ],
        "recommendations": ["Consider adding tests"]
    }
    
    return mock


@pytest.fixture
def mock_gemini_client():
    """Mock GeminiClient for testing"""
    mock = AsyncMock(spec=GeminiClient)
    
    # Mock model selection
    mock.select_model.return_value = "gemini-2.0-flash-exp"
    
    # Mock review generation
    mock.generate_review.return_value = """# Code Review

## Overview
This is a test review for a small PR.

## Issues Found
- Medium: Complex function in test.py

## Recommendations
- Consider adding tests
- Review complexity

## Overall Assessment
✅ LGTM with minor suggestions
"""
    
    # Mock templates
    mock.templates = Mock()
    mock.templates.QUICK_REVIEW_PROMPT = Mock()
    mock.templates.QUICK_REVIEW_PROMPT.format = Mock(return_value="Quick review prompt")
    mock.templates.STANDARD_REVIEW_PROMPT = Mock()
    mock.templates.STANDARD_REVIEW_PROMPT.format = Mock(return_value="Standard review prompt")
    mock.templates.DEEP_REVIEW_PROMPT = Mock()
    mock.templates.DEEP_REVIEW_PROMPT.format = Mock(return_value="Deep review prompt")
    mock.templates.FILE_REVIEW_PROMPT = Mock()
    mock.templates.FILE_REVIEW_PROMPT.format = Mock(return_value="File review prompt")
    mock.templates.AGGREGATION_PROMPT = Mock()
    mock.templates.AGGREGATION_PROMPT.format = Mock(return_value="Aggregation prompt")
    
    return mock


@pytest.fixture
def workflow(workflow_config, mock_context_builder, mock_gemini_client):
    """Create workflow instance with mocked dependencies"""
    return CodeReviewWorkflow(
        config=workflow_config,
        context_builder=mock_context_builder,
        gemini_client=mock_gemini_client
    )


# ========================================================================
# QUICK REVIEW TESTS (Small PRs)
# ========================================================================

@pytest.mark.asyncio
async def test_quick_review_small_pr(workflow, mock_context_builder, mock_gemini_client):
    """Test quick review strategy for small PR (<3 files, <100 additions, low risk)"""
    
    # Create initial state
    state: ReviewState = {
        "pr_number": 123,
        "repo_id": "test-repo",
        "pr_title": "Fix bug in utils",
        "pr_description": "Small bug fix",
        "files": [
            {"filename": "test.py", "additions": 30, "deletions": 5, "patch": "..."},
            {"filename": "utils.py", "additions": 20, "deletions": 5, "patch": "..."}
        ],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    # Run workflow
    graph = workflow.build()
    result = await graph.ainvoke(state)
    
    # Assertions
    assert result["status"] == "completed"
    assert result["review_strategy"] == "quick"
    assert result["selected_model"] == "gemini-2.0-flash-exp"
    assert "overall_summary" in result
    assert len(result["overall_summary"]) > 0
    assert mock_context_builder.build_pr_context.called
    assert mock_gemini_client.generate_review.called


@pytest.mark.asyncio
async def test_quick_review_prioritization(workflow, mock_context_builder):
    """Test file prioritization in quick review"""
    
    state: ReviewState = {
        "pr_number": 124,
        "repo_id": "test-repo",
        "pr_title": "Quick fix",
        "pr_description": "Minor change",
        "files": [{"filename": "test.py", "additions": 10, "deletions": 2, "patch": "..."}],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    # Run analyze node
    result = await workflow._analyze_node(state)
    
    # Check prioritization fields exist
    assert "high_priority_files" in result
    assert "medium_priority_files" in result
    assert "low_priority_files" in result
    assert "pr_context" in result


# ========================================================================
# STANDARD REVIEW TESTS (Medium PRs)
# ========================================================================

@pytest.mark.asyncio
async def test_standard_review_medium_pr(workflow, mock_context_builder, mock_gemini_client):
    """Test standard review strategy for medium PR (4-10 files, 100-500 additions)"""
    
    # Update mock to return medium PR context
    mock_context_builder.build_pr_context.return_value = {
        "repo_id": "test-repo",
        "pr_number": 125,
        "total_files": 5,
        "total_additions": 250,
        "total_deletions": 50,
        "languages": ["python", "typescript"],
        "risk_level": "medium",
        "requires_deep_review": False,
        "files": [
            {
                "filename": f"file{i}.py",
                "language": "python",
                "additions": 50,
                "deletions": 10,
                "complexity_score": 55,
                "issues_summary": {"critical": 0, "high": 1, "medium": 2},
                "dependencies": [],
                "similar_files": []
            }
            for i in range(5)
        ],
        "critical_issues": [],
        "high_issues": [{"file": "file0.py", "description": "Potential bug"}],
        "medium_issues": [],
        "recommendations": ["Review error handling"]
    }
    
    state: ReviewState = {
        "pr_number": 125,
        "repo_id": "test-repo",
        "pr_title": "Feature: Add new API endpoints",
        "pr_description": "Adds 5 new endpoints",
        "files": [
            {"filename": f"file{i}.py", "additions": 50, "deletions": 10, "patch": "..."}
            for i in range(5)
        ],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    graph = workflow.build()
    result = await graph.ainvoke(state)
    
    assert result["status"] == "completed"
    assert result["review_strategy"] == "standard"
    assert "file_reviews" in result or "overall_summary" in result


# ========================================================================
# DEEP REVIEW TESTS (Large/Complex PRs)
# ========================================================================

@pytest.mark.asyncio
async def test_deep_review_large_pr(workflow, mock_context_builder, mock_gemini_client):
    """Test deep review strategy for large PR (>10 files OR >500 additions OR high risk)"""
    
    # Update mock to return large PR context
    mock_context_builder.build_pr_context.return_value = {
        "repo_id": "test-repo",
        "pr_number": 126,
        "total_files": 15,
        "total_additions": 800,
        "total_deletions": 200,
        "languages": ["python", "typescript", "sql"],
        "risk_level": "high",
        "requires_deep_review": True,
        "files": [
            {
                "filename": f"module{i}.py",
                "language": "python",
                "additions": 53,
                "deletions": 13,
                "complexity_score": 75,
                "issues_summary": {"critical": 1, "high": 2, "medium": 3},
                "dependencies": [],
                "similar_files": []
            }
            for i in range(15)
        ],
        "critical_issues": [
            {"file": "module0.py", "line": 45, "description": "SQL injection risk"}
        ],
        "high_issues": [
            {"file": "module1.py", "description": "Memory leak potential"}
        ],
        "medium_issues": [],
        "complexity_hotspots": [
            {"file": "module0.py", "function": "process_data", "call_count": 120}
        ],
        "high_coupling_files": [
            {"source": "module0.py", "target": "module1.py", "connections": 15}
        ],
        "recommendations": ["Security review required", "Performance testing needed"]
    }
    
    state: ReviewState = {
        "pr_number": 126,
        "repo_id": "test-repo",
        "pr_title": "Major refactor: Database layer rewrite",
        "pr_description": "Complete database abstraction rewrite",
        "files": [
            {"filename": f"module{i}.py", "additions": 53, "deletions": 13, "patch": "..."}
            for i in range(15)
        ],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    graph = workflow.build()
    result = await graph.ainvoke(state)
    
    assert result["status"] == "completed"
    assert result["review_strategy"] == "deep"
    assert result["selected_model"] == "gemini-1.5-pro"  # Should use pro model
    assert "overall_summary" in result


@pytest.mark.asyncio
async def test_deep_review_critical_issues(workflow, mock_context_builder):
    """Test that critical issues trigger deep review"""
    
    # PR with critical security issues
    mock_context_builder.build_pr_context.return_value = {
        "repo_id": "test-repo",
        "pr_number": 127,
        "total_files": 3,
        "total_additions": 100,
        "total_deletions": 20,
        "languages": ["python"],
        "risk_level": "critical",  # Critical risk level
        "requires_deep_review": True,
        "files": [],
        "critical_issues": [
            {"file": "auth.py", "line": 30, "description": "Authentication bypass"},
            {"file": "auth.py", "line": 55, "description": "SQL injection"}
        ],
        "high_issues": [],
        "medium_issues": [],
        "recommendations": ["Security audit required"]
    }
    
    state: ReviewState = {
        "pr_number": 127,
        "repo_id": "test-repo",
        "pr_title": "Update auth logic",
        "pr_description": "Changes to authentication",
        "files": [{"filename": "auth.py", "additions": 100, "deletions": 20, "patch": "..."}],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    # Run routing
    analyzed = await workflow._analyze_node(state)
    routed = await workflow._route_node(analyzed)
    
    # Should select deep review strategy due to critical risk
    assert routed["review_strategy"] == "deep"


# ========================================================================
# ERROR HANDLING TESTS
# ========================================================================

@pytest.mark.asyncio
async def test_error_handling_rate_limit(workflow, mock_gemini_client):
    """Test retry logic for rate limit errors"""
    
    # Simulate rate limit error
    mock_gemini_client.generate_review.side_effect = [
        Exception("Rate limit exceeded"),
        "# Review after retry\n\nSuccessful review"
    ]
    
    state: ReviewState = {
        "pr_number": 128,
        "repo_id": "test-repo",
        "pr_title": "Test PR",
        "pr_description": "Test",
        "files": [{"filename": "test.py", "additions": 10, "deletions": 0, "patch": "..."}],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0,
        "pr_context": {"total_files": 1, "total_additions": 10, "files": []},
        "review_strategy": "quick",
        "selected_model": "gemini-2.0-flash-exp"
    }
    
    # Run quick review node
    result = await workflow._review_quick_node(state)
    
    # Should have error recorded
    assert len(result["errors"]) > 0
    assert "rate limit" in result["errors"][0]["message"].lower()


@pytest.mark.asyncio
async def test_error_handling_fallback_review(workflow, mock_context_builder):
    """Test fallback review generation when Gemini fails"""
    
    state: ReviewState = {
        "pr_number": 129,
        "repo_id": "test-repo",
        "pr_title": "Test PR",
        "pr_description": "Test",
        "files": [],
        "status": "pending",
        "errors": [
            {"message": "Gemini API unavailable", "step": "review_quick", "timestamp": datetime.utcnow().isoformat()}
        ],
        "messages": [],
        "retry_count": 3,  # Max retries
        "pr_context": {
            "total_files": 2,
            "total_additions": 50,
            "total_deletions": 10,
            "risk_level": "low",
            "critical_issues": [
                {"file": "test.py", "line": 10, "description": "Security issue"}
            ],
            "high_issues": [],
            "medium_issues": [],
            "recommendations": ["Add tests", "Review security"]
        }
    }
    
    # Generate fallback review
    result = await workflow._generate_fallback_review(state)
    
    assert "overall_summary" in result
    assert "automated summary" in result["overall_summary"].lower()
    assert result["status"] in ["completed_with_fallback", "failed"]


@pytest.mark.asyncio
async def test_error_decision_retry(workflow):
    """Test error decision logic for retries"""
    
    state: ReviewState = {
        "pr_number": 130,
        "repo_id": "test-repo",
        "pr_title": "Test",
        "pr_description": "Test",
        "files": [],
        "status": "error_recovery",
        "errors": [{"message": "Temporary error", "step": "review", "timestamp": datetime.utcnow().isoformat()}],
        "messages": [],
        "retry_count": 1  # Below max
    }
    
    decision = workflow._error_decision(state)
    
    assert decision == "retry"


@pytest.mark.asyncio
async def test_error_decision_max_retries(workflow):
    """Test error decision when max retries exceeded"""
    
    state: ReviewState = {
        "pr_number": 131,
        "repo_id": "test-repo",
        "pr_title": "Test",
        "pr_description": "Test",
        "files": [],
        "status": "error_recovery",
        "errors": [{"message": "Persistent error", "step": "review", "timestamp": datetime.utcnow().isoformat()}],
        "messages": [],
        "retry_count": 3  # Max retries
    }
    
    decision = workflow._error_decision(state)
    
    assert decision == "end"


# ========================================================================
# STATE TRANSITION TESTS
# ========================================================================

@pytest.mark.asyncio
async def test_state_transitions_quick_path(workflow, mock_context_builder, mock_gemini_client):
    """Test complete state transitions for quick review path"""
    
    state: ReviewState = {
        "pr_number": 132,
        "repo_id": "test-repo",
        "pr_title": "Small fix",
        "pr_description": "Minor bug fix",
        "files": [{"filename": "test.py", "additions": 10, "deletions": 2, "patch": "..."}],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    # Track state changes through workflow
    graph = workflow.build()
    result = await graph.ainvoke(state)
    
    # Verify state progression
    assert result["status"] == "completed"
    assert "completed_at" in result
    
    # Verify all required fields populated
    assert "pr_context" in result
    assert "review_strategy" in result
    assert "selected_model" in result
    assert "overall_summary" in result


@pytest.mark.asyncio
async def test_routing_logic(workflow, mock_context_builder):
    """Test routing decision logic based on PR characteristics"""
    
    # Test cases: (files, additions, risk_level, expected_strategy)
    test_cases = [
        (2, 50, "low", "quick"),
        (5, 250, "medium", "standard"),
        (12, 800, "high", "deep"),
        (8, 600, "medium", "deep"),  # >500 additions
        (3, 100, "critical", "deep"),  # critical risk
    ]
    
    for files, additions, risk, expected_strategy in test_cases:
        mock_context_builder.build_pr_context.return_value = {
            "repo_id": "test",
            "pr_number": 1,
            "total_files": files,
            "total_additions": additions,
            "total_deletions": 10,
            "languages": ["python"],
            "risk_level": risk,
            "requires_deep_review": risk in ["high", "critical"],
            "files": [],
            "critical_issues": [],
            "high_issues": [],
            "medium_issues": [],
            "recommendations": []
        }
        
        state: ReviewState = {
            "pr_number": 1,
            "repo_id": "test",
            "pr_title": "Test",
            "pr_description": "Test",
            "files": [{"filename": f"file{i}.py", "additions": 10, "deletions": 1, "patch": "..."} for i in range(files)],
            "status": "pending",
            "errors": [],
            "messages": [],
            "retry_count": 0
        }
        
        analyzed = await workflow._analyze_node(state)
        routed = await workflow._route_node(analyzed)
        
        assert routed["review_strategy"] == expected_strategy, \
            f"Expected {expected_strategy} for {files} files, {additions} adds, {risk} risk, got {routed['review_strategy']}"


# ========================================================================
# HELPER METHODS TESTS
# ========================================================================

def test_format_issues_summary(workflow):
    """Test issues summary formatting"""
    
    pr_context = {
        "critical_issues": [{"file": "test.py", "description": "Critical"}],
        "high_issues": [{"file": "test.py", "description": "High1"}, {"file": "test.py", "description": "High2"}],
        "medium_issues": []
    }
    
    result = workflow._format_issues_summary(pr_context)
    
    assert "Critical: 1" in result
    assert "High: 2" in result
    assert "Medium: 0" in result


def test_format_files_summary(workflow):
    """Test files summary formatting"""
    
    pr_context = {
        "files": [
            {"filename": "test1.py", "language": "python", "additions": 10, "deletions": 2},
            {"filename": "test2.py", "language": "python", "additions": 20, "deletions": 5}
        ]
    }
    
    result = workflow._format_files_summary(pr_context, max_files=10)
    
    assert "test1.py" in result
    assert "test2.py" in result
    assert "+10 -2" in result


def test_find_file_context(workflow):
    """Test file context lookup"""
    
    files = [
        {"filename": "test1.py", "additions": 10},
        {"filename": "test2.py", "additions": 20}
    ]
    
    result = workflow._find_file_context(files, "test1.py")
    
    assert result is not None
    assert result["filename"] == "test1.py"
    assert result["additions"] == 10
    
    # Test not found
    result = workflow._find_file_context(files, "nonexistent.py")
    assert result is None


# ========================================================================
# INTEGRATION TESTS
# ========================================================================

@pytest.mark.asyncio
async def test_workflow_factory(mock_context_builder, mock_gemini_client):
    """Test workflow creation via factory function"""
    
    workflow = create_review_workflow(
        context_builder=mock_context_builder,
        gemini_client=mock_gemini_client
    )
    
    assert isinstance(workflow, CodeReviewWorkflow)
    assert workflow.context_builder == mock_context_builder
    assert workflow.gemini_client == mock_gemini_client


@pytest.mark.asyncio
async def test_full_workflow_execution(mock_context_builder, mock_gemini_client):
    """Integration test: Full workflow execution end-to-end"""
    
    workflow = create_review_workflow(
        context_builder=mock_context_builder,
        gemini_client=mock_gemini_client
    )
    
    state: ReviewState = {
        "pr_number": 999,
        "repo_id": "integration-test",
        "pr_title": "Integration Test PR",
        "pr_description": "Full workflow test",
        "files": [
            {"filename": "test.py", "additions": 50, "deletions": 10, "patch": "test patch"},
            {"filename": "utils.py", "additions": 30, "deletions": 5, "patch": "utils patch"}
        ],
        "status": "pending",
        "errors": [],
        "messages": [],
        "retry_count": 0
    }
    
    graph = workflow.build()
    result = await graph.ainvoke(state)
    
    # Verify complete execution
    assert result["status"] in ["completed", "completed_with_fallback"]
    assert result["pr_number"] == 999
    assert "pr_context" in result
    assert "review_strategy" in result
    assert "selected_model" in result
    assert "overall_summary" in result or "file_reviews" in result
    
    # Verify workflow was called in correct order
    assert mock_context_builder.build_pr_context.called
    assert mock_gemini_client.select_model.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
