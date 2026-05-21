"""
Review Poster Service (Phase 5.3)

This module handles posting AI-generated reviews to GitHub PRs.

Design Decisions:
- Use single review comment for overall assessment
- Add inline comments for critical/high severity issues
- Check for existing reviews to avoid duplicates
- Determine review event type (APPROVE/COMMENT/REQUEST_CHANGES) based on severity
- Support review updates (edit existing review if bot already commented)
- Handle rate limiting and retries via GitHubClient

Key Features:
- Smart review posting (summary + inline comments)
- Duplicate detection
- Review status management (APPROVE/COMMENT/REQUEST_CHANGES)
- Error handling and logging
- Review metadata tracking
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from github_client import GitHubClient, create_github_client
from review_formatter import (
    ReviewFormatter,
    ReviewSummary,
    Issue,
    ReviewEvent,
    Severity,
    parse_workflow_output_to_review
)

logger = logging.getLogger(__name__)


# ========================================================================
# DATA CLASSES
# ========================================================================

@dataclass
class ReviewPostingConfig:
    """Configuration for review posting behavior"""
    
    # Inline comments settings
    post_inline_comments: bool = True
    inline_comment_severity_threshold: Severity = Severity.MEDIUM  # Post inline for >= MEDIUM
    max_inline_comments: int = 20  # Prevent comment spam
    
    # Review settings
    auto_approve_threshold: int = 0  # Auto-approve if <= X issues
    request_changes_threshold: int = 1  # Request changes if >= X critical issues
    
    # Duplicate handling
    check_for_duplicates: bool = True
    update_existing_review: bool = True  # Update bot's review if exists
    
    # Bot identification
    bot_signature: str = "🤖 AI Code Review System"
    
    @classmethod
    def from_review_strategy(cls, strategy: str) -> "ReviewPostingConfig":
        """
        Create config based on review strategy
        
        Args:
            strategy: quick, standard, or deep
            
        Returns:
            ReviewPostingConfig optimized for strategy
        """
        if strategy == "quick":
            return cls(
                post_inline_comments=False,
                max_inline_comments=5,
                auto_approve_threshold=2,
            )
        elif strategy == "standard":
            return cls(
                post_inline_comments=True,
                inline_comment_severity_threshold=Severity.HIGH,
                max_inline_comments=15,
                auto_approve_threshold=0,
            )
        elif strategy == "deep":
            return cls(
                post_inline_comments=True,
                inline_comment_severity_threshold=Severity.MEDIUM,
                max_inline_comments=30,
                auto_approve_threshold=0,
                request_changes_threshold=1,
            )
        else:
            return cls()


@dataclass
class ReviewPostResult:
    """Result of posting review to GitHub"""
    success: bool
    review_id: Optional[int] = None
    review_url: Optional[str] = None
    inline_comment_count: int = 0
    error: Optional[str] = None
    is_update: bool = False  # True if updated existing review


# ========================================================================
# REVIEW POSTER SERVICE
# ========================================================================

class ReviewPosterService:
    """
    Service for posting AI-generated reviews to GitHub PRs
    
    Workflow:
    1. Check for existing bot reviews (avoid duplicates)
    2. Determine review event type (APPROVE/COMMENT/REQUEST_CHANGES)
    3. Format review as markdown
    4. Post main review comment
    5. Post inline comments for critical issues
    6. Return result with metadata
    
    Usage:
        service = ReviewPosterService(github_client)
        result = await service.post_review(
            repo_full_name="owner/repo",
            pr_number=123,
            workflow_output=workflow_state,
            installation_id=12345
        )
    """
    
    def __init__(
        self,
        github_client: GitHubClient,
        formatter: Optional[ReviewFormatter] = None,
        config: Optional[ReviewPostingConfig] = None
    ):
        """
        Initialize review poster
        
        Args:
            github_client: GitHub API client
            formatter: Review formatter (creates default if None)
            config: Posting configuration (creates default if None)
        """
        self.github_client = github_client
        self.formatter = formatter or ReviewFormatter()
        self.config = config or ReviewPostingConfig()
        self.logger = logging.getLogger(f"{__name__}.ReviewPosterService")
    
    async def post_review(
        self,
        repo_full_name: str,
        pr_number: int,
        workflow_output: Dict[str, Any],
        installation_id: int,
        config_override: Optional[ReviewPostingConfig] = None
    ) -> ReviewPostResult:
        """
        Post AI-generated review to GitHub PR
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            pr_number: Pull request number
            workflow_output: Final state from LangGraph workflow
            installation_id: GitHub App installation ID
            config_override: Override default posting config
            
        Returns:
            ReviewPostResult with posting status
        """
        config = config_override or self.config
        
        try:
            self.logger.info(f"Posting review for {repo_full_name} PR #{pr_number}")
            
            # Step 1: Get PR data (need commit SHA)
            pr_data = await self.github_client.get_pull_request(
                repo_full_name, pr_number, installation_id
            )
            commit_id = pr_data["head"]["sha"]
            
            # Step 2: Check for existing reviews
            is_update = False
            existing_review_id = None
            
            if config.check_for_duplicates:
                existing_review_id = await self._find_existing_bot_review(
                    repo_full_name, pr_number, installation_id
                )
                
                if existing_review_id:
                    if config.update_existing_review:
                        self.logger.info(f"Found existing review {existing_review_id}, will update")
                        is_update = True
                    else:
                        self.logger.warning(f"Bot already reviewed PR #{pr_number}, skipping")
                        return ReviewPostResult(
                            success=False,
                            error="Bot already reviewed this PR (duplicate prevention)"
                        )
            
            # Step 3: Parse workflow output to review summary
            review_summary = parse_workflow_output_to_review(workflow_output)
            
            # Step 4: Determine review event type
            event_type = self._determine_event_type(review_summary, config)
            review_summary.event_type = event_type
            
            # Step 5: Format review as markdown
            review_body = self.formatter.format_review(review_summary)
            
            # Step 6: Post main review
            if is_update:
                # Update existing review (GitHub doesn't support editing reviews directly)
                # We'll dismiss the old one and post new one
                self.logger.info(f"Dismissing old review {existing_review_id}")
                await self._dismiss_review(
                    repo_full_name, pr_number, existing_review_id, installation_id
                )
            
            review_result = await self.github_client.post_review_comment(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                body=review_body,
                commit_id=commit_id,
                event=event_type.value,
                installation_id=installation_id
            )
            
            review_id = review_result.get("id")
            review_url = review_result.get("html_url")
            
            self.logger.info(f"Posted review {review_id}: {review_url}")
            
            # Step 7: Post inline comments for critical issues
            inline_count = 0
            
            if config.post_inline_comments:
                inline_count = await self._post_inline_comments(
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                    review_summary=review_summary,
                    commit_id=commit_id,
                    installation_id=installation_id,
                    config=config
                )
            
            return ReviewPostResult(
                success=True,
                review_id=review_id,
                review_url=review_url,
                inline_comment_count=inline_count,
                is_update=is_update
            )
            
        except Exception as e:
            self.logger.error(f"Failed to post review: {e}", exc_info=True)
            return ReviewPostResult(
                success=False,
                error=str(e)
            )
    
    async def _find_existing_bot_review(
        self,
        repo_full_name: str,
        pr_number: int,
        installation_id: int
    ) -> Optional[int]:
        """
        Find existing review by this bot
        
        Args:
            repo_full_name: Repository full name
            pr_number: Pull request number
            installation_id: GitHub App installation ID
            
        Returns:
            Review ID if found, None otherwise
        """
        try:
            reviews = await self.github_client.get_existing_reviews(
                repo_full_name, pr_number, installation_id
            )
            
            for review in reviews:
                # Check if review is from bot (contains signature)
                body = review.get("body", "")
                if self.config.bot_signature in body:
                    return review.get("id")
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to check existing reviews: {e}")
            return None
    
    async def _dismiss_review(
        self,
        repo_full_name: str,
        pr_number: int,
        review_id: int,
        installation_id: int
    ) -> None:
        """
        Dismiss (delete) existing review
        
        Note: GitHub API doesn't support deleting reviews directly.
        We can dismiss reviews, but they remain visible with "dismissed" status.
        
        Args:
            repo_full_name: Repository full name
            pr_number: Pull request number
            review_id: Review ID to dismiss
            installation_id: GitHub App installation ID
        """
        try:
            # GitHub doesn't expose dismiss review in PyGithub yet
            # For now, we'll just log that we'd dismiss it
            # In production, we'd use GitHub REST API directly:
            # PUT /repos/{owner}/{repo}/pulls/{pull_number}/reviews/{review_id}/dismissals
            self.logger.info(f"Would dismiss review {review_id} (not implemented)")
            
            # TODO: Implement via direct REST API call
            # github_instance = self.github_client._get_github_instance(installation_id)
            # repo = github_instance.get_repo(repo_full_name)
            # pull = repo.get_pull(pr_number)
            # review = pull.get_review(review_id)
            # review.dismiss(message="Updating with new review")
            
        except Exception as e:
            self.logger.warning(f"Failed to dismiss review: {e}")
    
    def _determine_event_type(
        self,
        review_summary: ReviewSummary,
        config: ReviewPostingConfig
    ) -> ReviewEvent:
        """
        Determine GitHub review event type based on issues found
        
        Logic:
        - REQUEST_CHANGES: If critical issues >= threshold
        - APPROVE: If total issues <= auto_approve_threshold
        - COMMENT: Otherwise
        
        Args:
            review_summary: Review summary with issues
            config: Posting configuration
            
        Returns:
            ReviewEvent type
        """
        # Count issues by severity
        critical_count = 0
        total_issues = 0
        
        for section in review_summary.sections:
            for issue in section.issues:
                total_issues += 1
                if issue.severity == Severity.CRITICAL:
                    critical_count += 1
        
        # Determine event type
        if critical_count >= config.request_changes_threshold:
            return ReviewEvent.REQUEST_CHANGES
        elif total_issues <= config.auto_approve_threshold:
            return ReviewEvent.APPROVE
        else:
            return ReviewEvent.COMMENT
    
    async def _post_inline_comments(
        self,
        repo_full_name: str,
        pr_number: int,
        review_summary: ReviewSummary,
        commit_id: str,
        installation_id: int,
        config: ReviewPostingConfig
    ) -> int:
        """
        Post inline comments for critical issues
        
        Args:
            repo_full_name: Repository full name
            pr_number: Pull request number
            review_summary: Review summary with issues
            commit_id: Commit SHA to comment on
            installation_id: GitHub App installation ID
            config: Posting configuration
            
        Returns:
            Number of inline comments posted
        """
        try:
            # Collect issues eligible for inline comments
            eligible_issues = []
            
            for section in review_summary.sections:
                for issue in section.issues:
                    # Check severity threshold
                    severity_values = {
                        Severity.CRITICAL: 0,
                        Severity.HIGH: 1,
                        Severity.MEDIUM: 2,
                        Severity.LOW: 3,
                        Severity.INFO: 4,
                    }
                    
                    issue_severity_value = severity_values[issue.severity]
                    threshold_value = severity_values[config.inline_comment_severity_threshold]
                    
                    # Only post if issue severity >= threshold AND has line number
                    if issue_severity_value <= threshold_value and issue.line:
                        eligible_issues.append(issue)
            
            # Limit to max inline comments
            if len(eligible_issues) > config.max_inline_comments:
                self.logger.warning(
                    f"Too many eligible issues ({len(eligible_issues)}), "
                    f"limiting to {config.max_inline_comments}"
                )
                eligible_issues = eligible_issues[:config.max_inline_comments]
            
            # Post inline comments
            posted_count = 0
            
            for issue in eligible_issues:
                try:
                    comment_body = self.formatter.format_inline_comment(issue)
                    
                    await self.github_client.post_inline_comment(
                        repo_full_name=repo_full_name,
                        pr_number=pr_number,
                        body=comment_body,
                        path=issue.file,
                        line=issue.line,
                        commit_id=commit_id,
                        installation_id=installation_id,
                        side="RIGHT"  # Comment on new code (right side of diff)
                    )
                    
                    posted_count += 1
                    self.logger.debug(f"Posted inline comment for {issue.file}:{issue.line}")
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Failed to post inline comment: {e}")
                    # Continue with other comments
            
            self.logger.info(f"Posted {posted_count} inline comments")
            return posted_count
            
        except Exception as e:
            self.logger.error(f"Failed to post inline comments: {e}")
            return 0


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def create_review_poster(
    strategy: Optional[str] = None,
    config_override: Optional[ReviewPostingConfig] = None
) -> ReviewPosterService:
    """
    Factory function to create ReviewPosterService
    
    Args:
        strategy: Review strategy (quick/standard/deep) - auto-configures
        config_override: Override auto-configuration
        
    Returns:
        ReviewPosterService ready to use
    """
    github_client = create_github_client()
    formatter = ReviewFormatter()
    
    # Auto-configure based on strategy
    if strategy and not config_override:
        config = ReviewPostingConfig.from_review_strategy(strategy)
    else:
        config = config_override or ReviewPostingConfig()
    
    return ReviewPosterService(
        github_client=github_client,
        formatter=formatter,
        config=config
    )


async def post_review_from_workflow(
    repo_full_name: str,
    pr_number: int,
    workflow_output: Dict[str, Any],
    installation_id: int,
    strategy: Optional[str] = None
) -> ReviewPostResult:
    """
    Convenience function to post review from workflow output
    
    Args:
        repo_full_name: Repository full name (owner/repo)
        pr_number: Pull request number
        workflow_output: Final state from LangGraph workflow
        installation_id: GitHub App installation ID
        strategy: Review strategy (auto-configures posting behavior)
        
    Returns:
        ReviewPostResult with posting status
    """
    service = create_review_poster(strategy=strategy)
    
    return await service.post_review(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        workflow_output=workflow_output,
        installation_id=installation_id
    )


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    # Example: Post review from workflow
    
    async def example():
        # Simulate workflow output
        workflow_output = {
            "pr_context": {
                "total_files": 3,
                "total_additions": 150,
                "total_deletions": 20,
                "critical_issues": [
                    {
                        "title": "SQL Injection",
                        "description": "User input not sanitized",
                        "file": "auth.py",
                        "line": 45,
                        "suggestion": "Use parameterized queries"
                    }
                ],
                "high_issues": [],
                "medium_issues": [],
                "recommendations": [
                    "Fix SQL injection before merging",
                    "Add input validation"
                ]
            },
            "overall_summary": "Found critical security vulnerability that must be fixed.",
            "review_strategy": "deep",
            "selected_model": "gemini-1.5-pro"
        }
        
        # Post review
        result = await post_review_from_workflow(
            repo_full_name="owner/repo",
            pr_number=123,
            workflow_output=workflow_output,
            installation_id=12345,
            strategy="deep"
        )
        
        if result.success:
            print(f"✅ Review posted: {result.review_url}")
            print(f"   Inline comments: {result.inline_comment_count}")
        else:
            print(f"❌ Failed: {result.error}")
    
    # Run example
    asyncio.run(example())
