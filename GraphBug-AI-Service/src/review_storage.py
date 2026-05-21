"""
Review Storage Service (Phase 5.4)

This module handles storing review data in:
1. PostgreSQL (Turso) - Review metadata, comments, analytics
2. Neo4j - Graph relationships between reviews and code entities

Design Decisions:
- Store review results in both databases for different use cases:
  * PostgreSQL: Historical data, analytics, user-facing queries
  * Neo4j: Graph relationships, impact analysis, code evolution tracking
- Create review-to-code entity relationships in Neo4j
- Support review comparison (before/after analysis)
- Track review lineage (multiple reviews for same PR)

Key Features:
- Dual database storage (PostgreSQL + Neo4j)
- Review comparison and diff
- Review history tracking
- Analytics and aggregation
- Cost tracking
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json

# Database clients
from neo4j import AsyncGraphDatabase
import httpx

logger = logging.getLogger(__name__)


# ========================================================================
# DATA CLASSES
# ========================================================================

@dataclass
class ReviewMetadata:
    """Review metadata for storage"""
    pull_request_id: str  # Foreign key to pull_requests table
    status: str  # pending, in_progress, completed, failed
    
    # Review results summary
    summary: Dict[str, Any]  # {overallScore, filesChanged, issuesFound, etc.}
    key_changes: List[str]
    recommendations: List[str]
    positives: List[str]
    
    # Model usage
    primary_model: str  # flash, pro, thinking
    models_used: List[Dict[str, Any]]
    total_tokens_input: int
    total_tokens_output: int
    total_cost: float
    execution_time_ms: int
    
    # Workflow tracking
    workflow_state: Optional[Dict] = None
    error_message: Optional[str] = None
    
    # GitHub tracking
    summary_comment_id: Optional[int] = None
    summary_comment_url: Optional[str] = None
    inline_comments_posted: int = 0
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ReviewComment:
    """Individual review comment for storage"""
    review_id: str  # Foreign key to code_reviews table
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    
    severity: str  # critical, high, medium, low, info
    category: str  # security, performance, bug, etc.
    
    title: str
    description: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    
    # GitHub tracking
    github_comment_id: Optional[int] = None
    github_comment_url: Optional[str] = None


@dataclass
class StorageResult:
    """Result of storage operation"""
    success: bool
    review_id: Optional[str] = None
    postgres_stored: bool = False
    neo4j_stored: bool = False
    comments_stored: int = 0
    error: Optional[str] = None


# ========================================================================
# NEO4J SCHEMA EXTENSIONS
# ========================================================================

# Cypher queries for review storage in Neo4j

CREATE_REVIEW_NODE = """
CREATE (r:Review {
    id: $id,
    pr_id: $pr_id,
    repo_id: $repo_id,
    status: $status,
    overall_score: $overall_score,
    files_changed: $files_changed,
    issues_found: $issues_found,
    critical_issues: $critical_issues,
    high_issues: $high_issues,
    medium_issues: $medium_issues,
    model_used: $model_used,
    total_cost: $total_cost,
    created_at: datetime($created_at),
    completed_at: datetime($completed_at)
})
RETURN r.id as id
"""

LINK_REVIEW_TO_PR = """
MATCH (r:Review {id: $review_id})
MATCH (pr:PullRequest {id: $pr_id})
CREATE (r)-[:REVIEWS]->(pr)
"""

LINK_REVIEW_TO_FILE = """
MATCH (r:Review {id: $review_id})
MATCH (f:File {path: $file_path, repo_id: $repo_id})
CREATE (r)-[:REVIEWED_FILE {
    issues_found: $issues_found,
    severity_max: $severity_max,
    categories: $categories
}]->(f)
"""

LINK_REVIEW_TO_FUNCTION = """
MATCH (r:Review {id: $review_id})
MATCH (f:Function {name: $function_name, file: $file_path, repo_id: $repo_id})
CREATE (r)-[:FOUND_ISSUE {
    severity: $severity,
    category: $category,
    title: $title,
    line: $line
}]->(f)
"""

GET_REVIEW_HISTORY = """
MATCH (r:Review)-[:REVIEWS]->(pr:PullRequest {id: $pr_id})
RETURN r
ORDER BY r.created_at DESC
LIMIT $limit
"""

GET_FILE_REVIEW_HISTORY = """
MATCH (r:Review)-[:REVIEWED_FILE]->(f:File {path: $file_path, repo_id: $repo_id})
RETURN r, f
ORDER BY r.created_at DESC
LIMIT $limit
"""

GET_FUNCTION_ISSUES = """
MATCH (r:Review)-[issue:FOUND_ISSUE]->(f:Function {name: $function_name, repo_id: $repo_id})
RETURN r, issue, f
ORDER BY r.created_at DESC
"""


# ========================================================================
# REVIEW STORAGE SERVICE
# ========================================================================

class ReviewStorageService:
    """
    Service for storing reviews in PostgreSQL and Neo4j
    
    Workflow:
    1. Store review metadata in PostgreSQL
    2. Store review comments in PostgreSQL
    3. Create review node in Neo4j
    4. Create relationships in Neo4j (review -> files, functions)
    5. Update review insights (analytics)
    
    Usage:
        service = ReviewStorageService(postgres_api_url, neo4j_config)
        result = await service.store_review(review_data)
    """
    
    def __init__(
        self,
        postgres_api_url: str,  # Frontend API URL for Turso operations
        neo4j_uri: str,
        neo4j_auth: tuple[str, str]
    ):
        """
        Initialize storage service
        
        Args:
            postgres_api_url: Frontend API URL (e.g., http://localhost:3000/api)
            neo4j_uri: Neo4j connection URI
            neo4j_auth: (username, password) tuple
        """
        self.postgres_api_url = postgres_api_url
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.http_client = httpx.AsyncClient()
        self.logger = logging.getLogger(f"{__name__}.ReviewStorageService")
    
    async def close(self):
        """Close connections"""
        await self.neo4j_driver.close()
        await self.http_client.aclose()
    
    async def store_review(
        self,
        review_metadata: ReviewMetadata,
        comments: List[ReviewComment],
        repo_id: str,
        pr_id: str
    ) -> StorageResult:
        """
        Store complete review in both databases
        
        Args:
            review_metadata: Review metadata
            comments: List of review comments
            repo_id: Repository ID
            pr_id: Pull request ID
            
        Returns:
            StorageResult with storage status
        """
        try:
            self.logger.info(f"Storing review for PR {pr_id}")
            
            # Step 1: Store in PostgreSQL
            review_id, postgres_success = await self._store_in_postgres(
                review_metadata, comments
            )
            
            if not postgres_success or not review_id:
                return StorageResult(
                    success=False,
                    error="Failed to store review in PostgreSQL"
                )
            
            # Step 2: Store in Neo4j
            neo4j_success = await self._store_in_neo4j(
                review_id, review_metadata, comments, repo_id, pr_id
            )
            
            # Step 3: Update insights (analytics)
            await self._update_insights(review_metadata, repo_id, pr_id)
            
            return StorageResult(
                success=True,
                review_id=review_id,
                postgres_stored=postgres_success,
                neo4j_stored=neo4j_success,
                comments_stored=len(comments)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store review: {e}", exc_info=True)
            return StorageResult(
                success=False,
                error=str(e)
            )
    
    async def _store_in_postgres(
        self,
        review: ReviewMetadata,
        comments: List[ReviewComment]
    ) -> tuple[Optional[str], bool]:
        """
        Store review in PostgreSQL via frontend API
        
        Args:
            review: Review metadata
            comments: Review comments
            
        Returns:
            (review_id, success)
        """
        try:
            # Create review
            review_data = asdict(review)
            review_data["started_at"] = review.started_at.isoformat() if review.started_at else None
            review_data["completed_at"] = review.completed_at.isoformat() if review.completed_at else None
            
            response = await self.http_client.post(
                f"{self.postgres_api_url}/reviews",
                json=review_data
            )
            response.raise_for_status()
            
            review_result = response.json()
            review_id = review_result.get("id")
            
            self.logger.info(f"Stored review in PostgreSQL: {review_id}")
            
            # Create comments
            if comments:
                for comment in comments:
                    comment_data = asdict(comment)
                    comment_data["review_id"] = review_id
                    
                    response = await self.http_client.post(
                        f"{self.postgres_api_url}/review_comments",
                        json=comment_data
                    )
                    response.raise_for_status()
                
                self.logger.info(f"Stored {len(comments)} comments in PostgreSQL")
            
            return review_id, True
            
        except Exception as e:
            self.logger.error(f"Failed to store in PostgreSQL: {e}")
            return None, False
    
    async def _store_in_neo4j(
        self,
        review_id: str,
        review: ReviewMetadata,
        comments: List[ReviewComment],
        repo_id: str,
        pr_id: str
    ) -> bool:
        """
        Store review in Neo4j with relationships
        
        Args:
            review_id: Review ID from PostgreSQL
            review: Review metadata
            comments: Review comments
            repo_id: Repository ID
            pr_id: Pull request ID
            
        Returns:
            bool: Success status
        """
        try:
            async with self.neo4j_driver.session() as session:
                # Create review node
                summary = review.summary or {}
                await session.run(
                    CREATE_REVIEW_NODE,
                    id=review_id,
                    pr_id=pr_id,
                    repo_id=repo_id,
                    status=review.status,
                    overall_score=summary.get("overallScore", 0),
                    files_changed=summary.get("filesChanged", 0),
                    issues_found=summary.get("issuesFound", 0),
                    critical_issues=summary.get("critical", 0),
                    high_issues=summary.get("high", 0),
                    medium_issues=summary.get("medium", 0),
                    model_used=review.primary_model,
                    total_cost=review.total_cost,
                    created_at=review.started_at.isoformat() if review.started_at else datetime.now().isoformat(),
                    completed_at=review.completed_at.isoformat() if review.completed_at else None
                )
                
                # Link to PR
                await session.run(
                    LINK_REVIEW_TO_PR,
                    review_id=review_id,
                    pr_id=pr_id
                )
                
                # Group comments by file
                files_map = {}
                for comment in comments:
                    if comment.file_path not in files_map:
                        files_map[comment.file_path] = []
                    files_map[comment.file_path].append(comment)
                
                # Link to files
                for file_path, file_comments in files_map.items():
                    severity_values = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
                    max_severity = min(
                        (severity_values.get(c.severity, 4) for c in file_comments),
                        default=4
                    )
                    severity_names = ["critical", "high", "medium", "low", "info"]
                    max_severity_name = severity_names[max_severity]
                    
                    categories = list(set(c.category for c in file_comments))
                    
                    await session.run(
                        LINK_REVIEW_TO_FILE,
                        review_id=review_id,
                        file_path=file_path,
                        repo_id=repo_id,
                        issues_found=len(file_comments),
                        severity_max=max_severity_name,
                        categories=categories
                    )
                
                # Link to functions (if possible - requires function matching)
                # TODO: Enhance this with AST matching to find exact functions
                
                self.logger.info(f"Stored review in Neo4j with {len(files_map)} file relationships")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store in Neo4j: {e}")
            return False
    
    async def _update_insights(
        self,
        review: ReviewMetadata,
        repo_id: str,
        pr_id: str
    ) -> None:
        """
        Update review insights (analytics aggregation)
        
        This is called after storing a review to update aggregated statistics
        
        Args:
            review: Review metadata
            repo_id: Repository ID
            pr_id: Pull request ID
        """
        try:
            # Trigger insight recalculation via API
            await self.http_client.post(
                f"{self.postgres_api_url}/review_insights/recalculate",
                json={
                    "scope": "repository",
                    "scope_id": repo_id,
                    "period_type": "week"
                }
            )
            
            self.logger.debug(f"Triggered insight recalculation for repo {repo_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update insights: {e}")
            # Don't fail the whole operation if insights fail
    
    # ====================================================================
    # REVIEW HISTORY & COMPARISON
    # ====================================================================
    
    async def get_review_history(
        self,
        pr_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get review history for a PR (from Neo4j)
        
        Args:
            pr_id: Pull request ID
            limit: Max number of reviews to return
            
        Returns:
            List of review dictionaries
        """
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    GET_REVIEW_HISTORY,
                    pr_id=pr_id,
                    limit=limit
                )
                
                reviews = []
                async for record in result:
                    reviews.append(dict(record["r"]))
                
                return reviews
                
        except Exception as e:
            self.logger.error(f"Failed to get review history: {e}")
            return []
    
    async def get_file_review_history(
        self,
        file_path: str,
        repo_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get review history for a specific file
        
        Useful for tracking how a file's review status changes over time
        
        Args:
            file_path: File path in repository
            repo_id: Repository ID
            limit: Max number of reviews to return
            
        Returns:
            List of review dictionaries
        """
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    GET_FILE_REVIEW_HISTORY,
                    file_path=file_path,
                    repo_id=repo_id,
                    limit=limit
                )
                
                reviews = []
                async for record in result:
                    reviews.append({
                        "review": dict(record["r"]),
                        "file": dict(record["f"])
                    })
                
                return reviews
                
        except Exception as e:
            self.logger.error(f"Failed to get file review history: {e}")
            return []
    
    async def compare_reviews(
        self,
        review_id_1: str,
        review_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two reviews (e.g., before/after refactoring)
        
        Args:
            review_id_1: First review ID
            review_id_2: Second review ID
            
        Returns:
            Comparison results with differences
        """
        try:
            # Fetch both reviews from PostgreSQL
            response1 = await self.http_client.get(
                f"{self.postgres_api_url}/reviews/{review_id_1}"
            )
            response2 = await self.http_client.get(
                f"{self.postgres_api_url}/reviews/{review_id_2}"
            )
            
            review1 = response1.json()
            review2 = response2.json()
            
            # Calculate differences
            summary1 = review1.get("summary", {})
            summary2 = review2.get("summary", {})
            
            comparison = {
                "review_1": review_id_1,
                "review_2": review_id_2,
                "score_change": summary2.get("overallScore", 0) - summary1.get("overallScore", 0),
                "issues_change": summary2.get("issuesFound", 0) - summary1.get("issuesFound", 0),
                "critical_change": summary2.get("critical", 0) - summary1.get("critical", 0),
                "high_change": summary2.get("high", 0) - summary1.get("high", 0),
                "medium_change": summary2.get("medium", 0) - summary1.get("medium", 0),
                "cost_change": review2.get("totalCost", 0) - review1.get("totalCost", 0),
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare reviews: {e}")
            return {}


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def create_review_storage_service(
    postgres_api_url: str = "http://localhost:3000/api",
    neo4j_uri: str = "neo4j://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "graphbug123"
) -> ReviewStorageService:
    """
    Factory function to create ReviewStorageService
    
    Args:
        postgres_api_url: Frontend API URL
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        
    Returns:
        ReviewStorageService ready to use
    """
    return ReviewStorageService(
        postgres_api_url=postgres_api_url,
        neo4j_uri=neo4j_uri,
        neo4j_auth=(neo4j_user, neo4j_password)
    )


async def store_review_from_workflow(
    workflow_output: Dict[str, Any],
    review_post_result: Dict[str, Any],
    repo_id: str,
    pr_id: str,
    pr_internal_id: str  # Database ID from pull_requests table
) -> StorageResult:
    """
    Convenience function to store review from workflow and posting results
    
    Args:
        workflow_output: Final state from LangGraph workflow
        review_post_result: Result from ReviewPosterService.post_review()
        repo_id: Repository ID
        pr_id: Pull request ID (GitHub PR number)
        pr_internal_id: Database ID from pull_requests table
        
    Returns:
        StorageResult with storage status
    """
    # Parse workflow output to review metadata
    pr_context = workflow_output.get("pr_context", {})
    
    summary = {
        "overallScore": pr_context.get("overall_score", 0),
        "filesChanged": pr_context.get("total_files", 0),
        "issuesFound": len(pr_context.get("critical_issues", [])) + 
                      len(pr_context.get("high_issues", [])) + 
                      len(pr_context.get("medium_issues", [])),
        "critical": len(pr_context.get("critical_issues", [])),
        "high": len(pr_context.get("high_issues", [])),
        "medium": len(pr_context.get("medium_issues", [])),
        "low": 0,
        "info": 0,
    }
    
    review_metadata = ReviewMetadata(
        pull_request_id=pr_internal_id,
        status="completed",
        summary=summary,
        key_changes=pr_context.get("key_changes", []),
        recommendations=pr_context.get("recommendations", []),
        positives=pr_context.get("positives", []),
        primary_model=workflow_output.get("selected_model", "flash"),
        models_used=[],  # TODO: Track from workflow
        total_tokens_input=0,  # TODO: Track from workflow
        total_tokens_output=0,  # TODO: Track from workflow
        total_cost=0.0,  # TODO: Calculate from workflow
        execution_time_ms=0,  # TODO: Track from workflow
        workflow_state=workflow_output,
        summary_comment_id=review_post_result.get("review_id"),
        summary_comment_url=review_post_result.get("review_url"),
        inline_comments_posted=review_post_result.get("inline_comment_count", 0),
        started_at=datetime.now(),
        completed_at=datetime.now()
    )
    
    # Parse comments from workflow
    comments = []
    for issue_list, severity in [
        (pr_context.get("critical_issues", []), "critical"),
        (pr_context.get("high_issues", []), "high"),
        (pr_context.get("medium_issues", []), "medium")
    ]:
        for issue in issue_list:
            comments.append(ReviewComment(
                review_id="",  # Will be filled by storage service
                file_path=issue.get("file", "unknown"),
                line_start=issue.get("line", 0),
                severity=severity,
                category=issue.get("category", "code_quality"),
                title=issue.get("title", "Issue found"),
                description=issue.get("description", ""),
                suggestion=issue.get("suggestion"),
                code_snippet=issue.get("code_snippet")
            ))
    
    # Store review
    service = create_review_storage_service()
    
    try:
        result = await service.store_review(
            review_metadata=review_metadata,
            comments=comments,
            repo_id=repo_id,
            pr_id=pr_id
        )
        return result
    finally:
        await service.close()


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    # Example: Store review
    
    async def example():
        service = create_review_storage_service()
        
        try:
            review = ReviewMetadata(
                pull_request_id="pr_123",
                status="completed",
                summary={
                    "overallScore": 75,
                    "filesChanged": 5,
                    "issuesFound": 3,
                    "critical": 1,
                    "high": 1,
                    "medium": 1,
                    "low": 0,
                    "info": 0
                },
                key_changes=["Added payment processing", "Updated database schema"],
                recommendations=["Fix SQL injection", "Add input validation"],
                positives=["Good test coverage", "Clear documentation"],
                primary_model="pro",
                models_used=[],
                total_tokens_input=5000,
                total_tokens_output=1500,
                total_cost=0.05,
                execution_time_ms=8000,
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
            
            comments = [
                ReviewComment(
                    review_id="",
                    file_path="auth.py",
                    line_start=45,
                    severity="critical",
                    category="security",
                    title="SQL Injection",
                    description="User input not sanitized",
                    suggestion="Use parameterized queries"
                )
            ]
            
            result = await service.store_review(
                review_metadata=review,
                comments=comments,
                repo_id="repo_123",
                pr_id="1"
            )
            
            if result.success:
                print(f"✅ Stored review: {result.review_id}")
                print(f"   PostgreSQL: {result.postgres_stored}")
                print(f"   Neo4j: {result.neo4j_stored}")
                print(f"   Comments: {result.comments_stored}")
            else:
                print(f"❌ Failed: {result.error}")
        finally:
            await service.close()
    
    asyncio.run(example())
