"""
GitHub API Client for Code Review System (Phase 5.1)

This module provides a comprehensive GitHub API client for:
- App installation authentication
- Token management and refresh
- Repository and PR data fetching
- Rate limit handling
- Comment posting (Phase 5.3)

Design Decisions:
- Use PyGithub library for GitHub API v3
- Implement token caching to minimize API calls
- Handle rate limiting with exponential backoff
- Support both App authentication (for webhooks) and OAuth (for user actions)
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import jwt
from github import Github, GithubIntegration, Auth
from github.GithubException import GithubException, RateLimitExceededException

from .logger import setup_logger

logger = setup_logger(__name__)


# ========================================================================
# CONFIGURATION
# ========================================================================

@dataclass
class GitHubConfig:
    """GitHub API configuration"""
    
    # App credentials
    app_id: str
    private_key: str  # PEM format
    
    # Rate limiting
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    
    # Token caching
    token_expiry_buffer_minutes: int = 5  # Refresh token 5 min before expiry
    
    @classmethod
    def from_env(cls) -> "GitHubConfig":
        """Load configuration from environment variables"""
        app_id = os.getenv("GITHUB_APP_ID")
        private_key_path = os.getenv("GITHUB_PRIVATE_KEY_PATH")
        private_key_content = os.getenv("GITHUB_PRIVATE_KEY")
        
        if not app_id:
            raise ValueError("GITHUB_APP_ID environment variable is required")
        
        # Load private key from file or environment variable
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, 'r') as f:
                private_key = f.read()
        elif private_key_content:
            # Handle escaped newlines (\n -> actual newlines)
            private_key = private_key_content.replace('\\n', '\n')
        else:
            raise ValueError("Either GITHUB_PRIVATE_KEY_PATH or GITHUB_PRIVATE_KEY must be set")
        
        return cls(
            app_id=app_id,
            private_key=private_key
        )


# ========================================================================
# TOKEN MANAGER
# ========================================================================

class TokenManager:
    """
    Manages GitHub App installation tokens
    
    Why this exists:
    - Installation tokens expire after 1 hour
    - Need to refresh tokens before they expire
    - Cache tokens to minimize API calls
    - Handle multiple installations (different repos)
    """
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.integration = GithubIntegration(
            integration_id=config.app_id,
            private_key=config.private_key
        )
        
        # Token cache: {installation_id: (token, expiry_time)}
        self._token_cache: Dict[int, tuple[str, datetime]] = {}
        
        logger.info("TokenManager initialized")
    
    def get_installation_token(self, installation_id: int) -> str:
        """
        Get installation token (from cache or generate new)
        
        Args:
            installation_id: GitHub App installation ID
            
        Returns:
            str: Installation access token
        """
        # Check cache
        if installation_id in self._token_cache:
            token, expiry = self._token_cache[installation_id]
            
            # Check if token is still valid (with buffer)
            buffer = timedelta(minutes=self.config.token_expiry_buffer_minutes)
            if datetime.utcnow() + buffer < expiry:
                logger.debug(f"Using cached token for installation {installation_id}")
                return token
        
        # Generate new token
        logger.info(f"Generating new token for installation {installation_id}")
        auth = self.integration.get_access_token(installation_id)
        
        # Cache token
        expiry = datetime.utcnow() + timedelta(hours=1)  # Tokens valid for 1 hour
        self._token_cache[installation_id] = (auth.token, expiry)
        
        return auth.token
    
    def invalidate_token(self, installation_id: int):
        """Invalidate cached token (e.g., on 401 error)"""
        if installation_id in self._token_cache:
            logger.info(f"Invalidating cached token for installation {installation_id}")
            del self._token_cache[installation_id]


# ========================================================================
# GITHUB API CLIENT
# ========================================================================

class GitHubClient:
    """
    GitHub API client with App authentication
    
    Features:
    - App installation authentication
    - Automatic token refresh
    - Rate limit handling
    - Retry logic with exponential backoff
    - Repository and PR data fetching
    - Comment posting (Phase 5.3)
    
    Usage:
        client = GitHubClient(config)
        pr_data = await client.get_pull_request("owner/repo", 123, installation_id)
        await client.post_review_comment(...)
    """
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.token_manager = TokenManager(config)
        
        logger.info("GitHubClient initialized")
    
    def get_installation_token(self, installation_id: int) -> str:
        """
        Get installation access token (public method for repo cloning)
        
        Args:
            installation_id: GitHub App installation ID
            
        Returns:
            str: Installation access token for git operations
        """
        return self.token_manager.get_installation_token(installation_id)
    
    def _get_github_instance(self, installation_id: int) -> Github:
        """
        Get authenticated GitHub instance for installation
        
        Args:
            installation_id: GitHub App installation ID
            
        Returns:
            Github: Authenticated PyGithub instance
        """
        token = self.token_manager.get_installation_token(installation_id)
        auth = Auth.Token(token)
        return Github(auth=auth)
    
    async def _handle_rate_limit(self, github: Github):
        """
        Check rate limit and wait if necessary
        
        GitHub Rate Limits:
        - 5000 requests/hour for authenticated requests
        - Rate limit resets at a specific time
        """
        rate_limit = github.get_rate_limit()
        core = rate_limit.core
        
        if core.remaining < 100:  # Less than 100 requests left
            reset_time = core.reset.timestamp()
            wait_time = max(0, reset_time - time.time())
            
            logger.warning(
                f"Rate limit low: {core.remaining}/{core.limit}. "
                f"Waiting {wait_time:.0f}s until reset"
            )
            
            await asyncio.sleep(wait_time + 1)  # Wait until reset + 1s buffer
    
    async def _retry_on_error(self, func, *args, **kwargs):
        """
        Execute function with retry logic
        
        Handles:
        - Rate limit errors (wait and retry)
        - Transient network errors (exponential backoff)
        - Authentication errors (invalidate token and retry once)
        """
        for attempt in range(self.config.max_retries):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            
            except RateLimitExceededException as e:
                logger.warning(f"Rate limit exceeded: {e}")
                # GitHub will provide reset time
                await asyncio.sleep(60)  # Wait 1 minute
            
            except GithubException as e:
                if e.status == 401:  # Unauthorized
                    logger.warning("Authentication failed, invalidating token")
                    installation_id = kwargs.get('installation_id')
                    if installation_id:
                        self.token_manager.invalidate_token(installation_id)
                    
                    if attempt < self.config.max_retries - 1:
                        continue  # Retry with new token
                    raise
                
                elif e.status >= 500:  # Server error
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(f"GitHub server error {e.status}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise  # Client error, don't retry
            
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                logger.error(f"Error: {e}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Failed after {self.config.max_retries} retries")
    
    # ====================================================================
    # REPOSITORY OPERATIONS
    # ====================================================================
    
    async def get_repository(
        self,
        repo_full_name: str,
        installation_id: int
    ) -> Dict[str, Any]:
        """
        Get repository information
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            installation_id: GitHub App installation ID
            
        Returns:
            Dict with repository data
        """
        logger.info(f"Fetching repository: {repo_full_name}")
        
        def _fetch():
            github = self._get_github_instance(installation_id)
            repo = github.get_repo(repo_full_name)
            
            return {
                "id": repo.id,
                "name": repo.name,
                "full_name": repo.full_name,
                "owner": repo.owner.login,
                "description": repo.description,
                "language": repo.language,
                "default_branch": repo.default_branch,
                "private": repo.private,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
            }
        
        return await self._retry_on_error(_fetch)
    
    # ====================================================================
    # PULL REQUEST OPERATIONS
    # ====================================================================
    
    async def get_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
        installation_id: int
    ) -> Dict[str, Any]:
        """
        Get pull request information
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            pr_number: Pull request number
            installation_id: GitHub App installation ID
            
        Returns:
            Dict with PR data including files and diff
        """
        logger.info(f"Fetching PR: {repo_full_name}#{pr_number}")
        
        def _fetch():
            github = self._get_github_instance(installation_id)
            repo = github.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            # Get PR files with diffs
            files = []
            for file in pr.get_files():
                files.append({
                    "filename": file.filename,
                    "status": file.status,  # added, removed, modified, renamed
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch if file.patch else "",
                    "blob_url": file.blob_url,
                    "raw_url": file.raw_url,
                })
            
            return {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body or "",
                "state": pr.state,  # open, closed
                "user": {
                    "login": pr.user.login,
                    "id": pr.user.id,
                    "avatar_url": pr.user.avatar_url,
                },
                "created_at": pr.created_at.isoformat(),
                "updated_at": pr.updated_at.isoformat(),
                "base": {
                    "ref": pr.base.ref,
                    "sha": pr.base.sha,
                },
                "head": {
                    "ref": pr.head.ref,
                    "sha": pr.head.sha,
                },
                "files": files,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changed_files": pr.changed_files,
                "mergeable": pr.mergeable,
                "mergeable_state": pr.mergeable_state,
            }
        
        return await self._retry_on_error(_fetch)
    
    async def list_pull_requests(
        self,
        repo_full_name: str,
        installation_id: int,
        state: str = "open",
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        List pull requests in repository
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            installation_id: GitHub App installation ID
            state: PR state (open, closed, all)
            limit: Maximum number of PRs to return
            
        Returns:
            List of PR data dictionaries
        """
        logger.info(f"Listing PRs for {repo_full_name} (state={state}, limit={limit})")
        
        def _fetch():
            github = self._get_github_instance(installation_id)
            repo = github.get_repo(repo_full_name)
            pulls = repo.get_pulls(state=state)
            
            results = []
            for pr in pulls[:limit]:
                results.append({
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "user": pr.user.login,
                    "created_at": pr.created_at.isoformat(),
                    "updated_at": pr.updated_at.isoformat(),
                })
            
            return results
        
        return await self._retry_on_error(_fetch)
    
    # ====================================================================
    # REVIEW OPERATIONS (Phase 5.3)
    # ====================================================================
    
    async def post_review_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
        installation_id: int,
        commit_id: Optional[str] = None,
        event: str = "COMMENT"
    ) -> Dict[str, Any]:
        """
        Post a review comment on a pull request
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            pr_number: Pull request number
            body: Review comment body (markdown)
            installation_id: GitHub App installation ID
            commit_id: Specific commit to review (optional, uses latest if not provided)
            event: Review event type:
                - COMMENT: General comment (no approval/rejection)
                - APPROVE: Approve PR
                - REQUEST_CHANGES: Request changes
        
        Returns:
            Dict with review data
            
        Implementation in Phase 5.3
        """
        logger.info(f"Posting review to {repo_full_name}#{pr_number}")
        
        def _post():
            github = self._get_github_instance(installation_id)
            repo = github.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            # Use latest commit if not specified
            target_commit_id = commit_id if commit_id else pr.head.sha
            
            # Create review
            review = pr.create_review(
                body=body,
                commit=repo.get_commit(target_commit_id),
                event=event
            )
            
            return {
                "id": review.id,
                "user": review.user.login,
                "body": review.body,
                "state": review.state,
                "html_url": review.html_url,
                "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
            }
        
        return await self._retry_on_error(_post)
    
    async def post_inline_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
        path: str,
        line: int,
        installation_id: int,
        commit_id: Optional[str] = None,
        side: str = "RIGHT"
    ) -> Dict[str, Any]:
        """
        Post an inline comment on a specific line in PR diff
        
        Args:
            repo_full_name: Repository full name
            pr_number: Pull request number
            body: Comment body
            path: File path in the PR
            line: Line number in the diff
            installation_id: GitHub App installation ID
            commit_id: Specific commit (optional)
            side: Which side of diff (LEFT for old, RIGHT for new)
        
        Returns:
            Dict with comment data
            
        Implementation in Phase 5.3
        """
        logger.info(f"Posting inline comment to {repo_full_name}#{pr_number} at {path}:{line}")
        
        def _post():
            github = self._get_github_instance(installation_id)
            repo = github.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            # Use latest commit if not specified
            target_commit_id = commit_id if commit_id else pr.head.sha
            
            # Create review comment (inline)
            comment = pr.create_review_comment(
                body=body,
                commit=repo.get_commit(target_commit_id),
                path=path,
                line=line,
                side=side
            )
            
            return {
                "id": comment.id,
                "path": comment.path,
                "line": comment.line,
                "body": comment.body,
                "user": comment.user.login,
                "html_url": comment.html_url,
                "created_at": comment.created_at.isoformat(),
            }
        
        return await self._retry_on_error(_post)
    
    async def get_existing_reviews(
        self,
        repo_full_name: str,
        pr_number: int,
        installation_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get existing reviews on a PR (to avoid duplicate reviews)
        
        Args:
            repo_full_name: Repository full name
            pr_number: Pull request number
            installation_id: GitHub App installation ID
        
        Returns:
            List of existing reviews
        """
        logger.info(f"Fetching existing reviews for {repo_full_name}#{pr_number}")
        
        def _fetch():
            github = self._get_github_instance(installation_id)
            repo = github.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            reviews = []
            for review in pr.get_reviews():
                reviews.append({
                    "id": review.id,
                    "user": review.user.login,
                    "body": review.body,
                    "state": review.state,
                    "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
                })
            
            return reviews
        
        return await self._retry_on_error(_fetch)
    
    # ====================================================================
    # RATE LIMIT UTILITIES
    # ====================================================================
    
    async def get_rate_limit_status(self, installation_id: int) -> Dict[str, Any]:
        """
        Get current rate limit status
        
        Returns:
            Dict with rate limit info
        """
        github = self._get_github_instance(installation_id)
        rate_limit = github.get_rate_limit()
        
        return {
            "core": {
                "limit": rate_limit.core.limit,
                "remaining": rate_limit.core.remaining,
                "reset": rate_limit.core.reset.isoformat(),
            },
            "search": {
                "limit": rate_limit.search.limit,
                "remaining": rate_limit.search.remaining,
                "reset": rate_limit.search.reset.isoformat(),
            }
        }


# ========================================================================
# FACTORY FUNCTION
# ========================================================================

def create_github_client() -> GitHubClient:
    """Create GitHub client from environment configuration"""
    config = GitHubConfig.from_env()
    return GitHubClient(config)


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    async def main():
        # Create client
        client = create_github_client()
        
        # Example: Get PR data
        installation_id = 12345  # From webhook
        pr_data = await client.get_pull_request(
            repo_full_name="owner/repo",
            pr_number=123,
            installation_id=installation_id
        )
        
        print(f"PR #{pr_data['number']}: {pr_data['title']}")
        print(f"Files changed: {len(pr_data['files'])}")
        
        # Example: Post review
        await client.post_review_comment(
            repo_full_name="owner/repo",
            pr_number=123,
            body="# AI Code Review\n\nLooks good! ✅",
            installation_id=installation_id,
            event="APPROVE"
        )
    
    asyncio.run(main())
