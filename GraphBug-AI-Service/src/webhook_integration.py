"""
Webhook Integration Service (Phase 5.5)

This module integrates all Phase 5 components to provide end-to-end PR review automation.

Complete Flow:
1. GitHub webhook → Frontend API → AI Service
2. AI Service fetches PR data via GitHub API
3. AI Service runs LangGraph workflow (review generation)
4. AI Service formats review as markdown
5. AI Service posts review to GitHub
6. AI Service stores review in databases

Design Decisions:
- Use async queue for concurrent reviews (prevent overwhelming services)
- Rate limiting: Max 3 concurrent reviews
- Priority queue: Critical PRs first (based on PR size/complexity)
- Status tracking: pending → processing → completed/failed
- Error recovery: Retry failed reviews with exponential backoff
- Webhook idempotency: Prevent duplicate reviews

Key Features:
- End-to-end automation
- Concurrent review queue
- Priority-based scheduling
- Error recovery
- Status tracking
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from collections import deque
import time

# Phase 4 - Workflow
from workflow import create_review_workflow, ReviewState

# Phase 5 Components
from github_client import create_github_client, GitHubClient
from review_formatter import parse_workflow_output_to_review
from review_poster import create_review_poster, ReviewPostResult
from review_storage import create_review_storage_service, store_review_from_workflow

logger = logging.getLogger(__name__)


# ========================================================================
# ENUMS & DATA CLASSES
# ========================================================================

class ReviewPriority(Enum):
    """Review priority levels"""
    LOW = 3      # Small PRs (<100 lines)
    MEDIUM = 2   # Normal PRs (100-500 lines)
    HIGH = 1     # Large PRs (500-1000 lines)
    CRITICAL = 0 # Huge PRs (>1000 lines) or security-related


class ReviewStatus(Enum):
    """Review processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ReviewTask:
    """Represents a review task in the queue"""
    # PR identifiers
    repo_full_name: str
    pr_number: int
    installation_id: int
    
    # Repository context
    repo_id: str  # Internal repo ID
    pr_internal_id: str  # Database PR ID
    
    # Task metadata
    priority: ReviewPriority
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status
    status: ReviewStatus = ReviewStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    
    # Results
    error: Optional[str] = None
    review_id: Optional[str] = None
    
    def __lt__(self, other):
        """Compare by priority for heap queue"""
        return self.priority.value < other.priority.value


@dataclass
class QueueConfig:
    """Configuration for review queue"""
    max_concurrent: int = 3
    max_queue_size: int = 100
    retry_delay_seconds: int = 60
    task_timeout_seconds: int = 300  # 5 minutes


# ========================================================================
# REVIEW QUEUE
# ========================================================================

class ReviewQueue:
    """
    Priority queue for managing concurrent PR reviews
    
    Features:
    - Priority-based scheduling (critical PRs first)
    - Concurrent execution (max 3 reviews at once)
    - Error recovery with retries
    - Task timeout handling
    - Status tracking
    
    Usage:
        queue = ReviewQueue(config)
        await queue.start()
        task_id = await queue.enqueue(review_task)
        status = await queue.get_status(task_id)
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        """
        Initialize review queue
        
        Args:
            config: Queue configuration
        """
        self.config = config or QueueConfig()
        self.queue: deque[ReviewTask] = deque()
        self.tasks: Dict[str, ReviewTask] = {}  # task_id -> ReviewTask
        self.processing: set[str] = set()  # Currently processing task IDs
        self.running = False
        self.logger = logging.getLogger(f"{__name__}.ReviewQueue")
    
    def _generate_task_id(self, task: ReviewTask) -> str:
        """Generate unique task ID"""
        return f"{task.repo_full_name}#{task.pr_number}"
    
    async def enqueue(self, task: ReviewTask) -> str:
        """
        Add task to queue
        
        Args:
            task: Review task to enqueue
            
        Returns:
            str: Task ID for status tracking
        """
        task_id = self._generate_task_id(task)
        
        # Check if already in queue or processing
        if task_id in self.tasks:
            existing = self.tasks[task_id]
            if existing.status in [ReviewStatus.PENDING, ReviewStatus.PROCESSING]:
                self.logger.info(f"Task {task_id} already in queue, skipping")
                return task_id
        
        # Check queue size
        if len(self.queue) >= self.config.max_queue_size:
            raise RuntimeError(f"Queue full (max {self.config.max_queue_size})")
        
        # Add to queue (sorted by priority)
        self.queue.append(task)
        self.tasks[task_id] = task
        
        # Sort by priority (critical first)
        self.queue = deque(sorted(self.queue, key=lambda t: t.priority.value))
        
        self.logger.info(
            f"Enqueued {task_id} with priority {task.priority.name} "
            f"(queue size: {len(self.queue)})"
        )
        
        return task_id
    
    async def get_status(self, task_id: str) -> Optional[ReviewTask]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    async def start(self):
        """Start queue processor"""
        self.running = True
        self.logger.info("Review queue started")
        
        # Start worker tasks
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.config.max_concurrent)
        ]
        
        await asyncio.gather(*workers)
    
    async def stop(self):
        """Stop queue processor"""
        self.running = False
        self.logger.info("Review queue stopped")
    
    async def _worker(self, worker_id: int):
        """
        Worker task that processes reviews from queue
        
        Args:
            worker_id: Worker identifier for logging
        """
        self.logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get next task
                if not self.queue:
                    await asyncio.sleep(1)
                    continue
                
                task = self.queue.popleft()
                task_id = self._generate_task_id(task)
                
                # Mark as processing
                task.status = ReviewStatus.PROCESSING
                task.started_at = datetime.now()
                self.processing.add(task_id)
                
                self.logger.info(
                    f"Worker {worker_id} processing {task_id} "
                    f"(attempt {task.attempts + 1}/{task.max_attempts})"
                )
                
                # Process review with timeout
                try:
                    await asyncio.wait_for(
                        self._process_review(task),
                        timeout=self.config.task_timeout_seconds
                    )
                    
                    task.status = ReviewStatus.COMPLETED
                    task.completed_at = datetime.now()
                    self.logger.info(
                        f"Worker {worker_id} completed {task_id} in "
                        f"{(task.completed_at - task.started_at).total_seconds():.1f}s"
                    )
                    
                except asyncio.TimeoutError:
                    task.error = f"Task timeout ({self.config.task_timeout_seconds}s)"
                    task.status = ReviewStatus.FAILED
                    self.logger.error(f"Worker {worker_id} timeout on {task_id}")
                    
                except Exception as e:
                    task.error = str(e)
                    task.attempts += 1
                    
                    # Retry if not max attempts
                    if task.attempts < task.max_attempts:
                        task.status = ReviewStatus.PENDING
                        self.logger.warning(
                            f"Worker {worker_id} failed {task_id} (will retry): {e}"
                        )
                        
                        # Re-enqueue with delay
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        self.queue.append(task)
                    else:
                        task.status = ReviewStatus.FAILED
                        self.logger.error(
                            f"Worker {worker_id} failed {task_id} (max retries): {e}"
                        )
                
                finally:
                    self.processing.remove(task_id)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    async def _process_review(self, task: ReviewTask):
        """
        Process single review task
        
        This is the main integration point that connects all Phase 5 components
        
        Args:
            task: Review task to process
        """
        self.logger.info(f"Processing review for {task.repo_full_name} PR #{task.pr_number}")
        
        # Step 1: Fetch PR data from GitHub
        github_client = create_github_client()
        pr_data = await github_client.get_pull_request(
            task.repo_full_name,
            task.pr_number,
            task.installation_id
        )
        
        self.logger.info(
            f"Fetched PR data: {len(pr_data.get('files', []))} files, "
            f"+{pr_data.get('additions', 0)}/-{pr_data.get('deletions', 0)}"
        )
        
        # Step 2: Determine review strategy based on PR size
        total_changes = pr_data.get("additions", 0) + pr_data.get("deletions", 0)
        if total_changes < 100:
            strategy = "quick"
        elif total_changes < 500:
            strategy = "standard"
        else:
            strategy = "deep"
        
        # Step 3: Run LangGraph workflow
        from context_builder import ContextBuilder
        from code_analyzer import CodeAnalyzer
        from parser import UniversalParser
        from graph_builder import GraphBuilder
        from vector_builder import VectorBuilder
        from sentence_transformers import SentenceTransformer
        
        # Initialize components (TODO: Use singleton pattern)
        parser = UniversalParser()
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        graph_db = GraphBuilder("neo4j://localhost:7687", ("neo4j", "graphbug123"))
        vector_db = VectorBuilder("http://localhost:6333", embed_model)
        code_analyzer = CodeAnalyzer(graph_db, vector_db, parser)
        context_builder = ContextBuilder(code_analyzer, graph_db, vector_db)
        
        workflow = create_review_workflow(context_builder=context_builder)
        
        # Create initial state
        initial_state: ReviewState = {
            "pr_context": {
                "repo_id": task.repo_id,
                "pr_number": task.pr_number,
                "files": pr_data.get("files", []),
                "total_files": len(pr_data.get("files", [])),
                "total_additions": pr_data.get("additions", 0),
                "total_deletions": pr_data.get("deletions", 0),
            },
            "review_strategy": strategy,
            "selected_model": "flash",  # Will be determined by workflow
            "overall_summary": "",
            "review_completed": False
        }
        
        # Run workflow
        final_state = await workflow.ainvoke(initial_state)
        
        self.logger.info(f"Workflow completed for {task.repo_full_name} PR #{task.pr_number}")
        
        # Check if workflow failed (don't post partial/error reviews)
        workflow_status = final_state.get("status")
        if workflow_status == "failed":
            error_msg = "Workflow failed to generate review"
            errors = final_state.get("errors", [])
            if errors:
                last_error = errors[-1]
                error_msg = f"{error_msg}: {last_error.get('message', 'Unknown error')}"
                
            self.logger.error(error_msg)
            # Raise error to trigger task retry/failure logic in worker
            raise RuntimeError(error_msg)
        
        # Step 4: Post review to GitHub
        poster = create_review_poster(strategy=strategy)
        post_result = await poster.post_review(
            repo_full_name=task.repo_full_name,
            pr_number=task.pr_number,
            workflow_output=final_state,
            installation_id=task.installation_id
        )
        
        if not post_result.success:
            raise RuntimeError(f"Failed to post review: {post_result.error}")
        
        self.logger.info(
            f"Posted review to GitHub: {post_result.review_url} "
            f"({post_result.inline_comment_count} inline comments)"
        )
        
        # Step 5: Store review in databases
        storage_result = await store_review_from_workflow(
            workflow_output=final_state,
            review_post_result=asdict(post_result) if hasattr(post_result, '__dict__') else post_result,
            repo_id=task.repo_id,
            pr_id=str(task.pr_number),
            pr_internal_id=task.pr_internal_id
        )
        
        if not storage_result.success:
            self.logger.warning(f"Failed to store review: {storage_result.error}")
            # Don't fail the whole task if storage fails
        
        task.review_id = storage_result.review_id
        
        self.logger.info(f"Review completed for {task.repo_full_name} PR #{task.pr_number}")


# ========================================================================
# WEBHOOK INTEGRATION SERVICE
# ========================================================================

class WebhookIntegrationService:
    """
    Service for handling GitHub webhook events and triggering reviews
    
    Usage:
        service = WebhookIntegrationService(queue)
        await service.handle_pull_request_webhook(payload)
    """
    
    def __init__(self, queue: ReviewQueue):
        """
        Initialize webhook integration service
        
        Args:
            queue: Review queue for task scheduling
        """
        self.queue = queue
        self.logger = logging.getLogger(f"{__name__}.WebhookIntegrationService")
    
    async def handle_pull_request_webhook(
        self,
        payload: Dict[str, Any]
    ) -> str:
        """
        Handle pull request webhook event
        
        Args:
            payload: Webhook payload from GitHub
            
        Returns:
            str: Task ID for tracking
        """
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})
        installation = payload.get("installation", {})
        
        # Only process opened and synchronized events
        if action not in ["opened", "synchronize"]:
            self.logger.info(f"Ignoring PR action: {action}")
            return ""
        
        # Determine priority based on PR size
        additions = pr.get("additions", 0)
        deletions = pr.get("deletions", 0)
        total_changes = additions + deletions
        
        if total_changes < 100:
            priority = ReviewPriority.LOW
        elif total_changes < 500:
            priority = ReviewPriority.MEDIUM
        elif total_changes < 1000:
            priority = ReviewPriority.HIGH
        else:
            priority = ReviewPriority.CRITICAL
        
        # Create review task
        task = ReviewTask(
            repo_full_name=repo.get("full_name"),
            pr_number=pr.get("number"),
            installation_id=installation.get("id"),
            repo_id=repo.get("id"),  # TODO: Map to internal repo ID
            pr_internal_id="",  # TODO: Create PR in database first
            priority=priority
        )
        
        # Enqueue task
        task_id = await self.queue.enqueue(task)
        
        self.logger.info(
            f"Enqueued review for {task.repo_full_name} PR #{task.pr_number} "
            f"with priority {priority.name}"
        )
        
        return task_id


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

# Global queue instance
_review_queue: Optional[ReviewQueue] = None


def get_review_queue(config: Optional[QueueConfig] = None) -> ReviewQueue:
    """
    Get singleton review queue instance
    
    Args:
        config: Queue configuration (only used on first call)
        
    Returns:
        ReviewQueue instance
    """
    global _review_queue
    
    if _review_queue is None:
        _review_queue = ReviewQueue(config)
    
    return _review_queue


async def start_review_queue(config: Optional[QueueConfig] = None):
    """
    Start review queue processor
    
    Call this from application startup (e.g., FastAPI lifespan)
    
    Args:
        config: Queue configuration
    """
    queue = get_review_queue(config)
    await queue.start()


async def stop_review_queue():
    """
    Stop review queue processor
    
    Call this from application shutdown
    """
    if _review_queue:
        await _review_queue.stop()


async def handle_pr_webhook(payload: Dict[str, Any]) -> str:
    """
    Convenience function to handle PR webhook
    
    Args:
        payload: Webhook payload from GitHub
        
    Returns:
        str: Task ID for tracking
    """
    queue = get_review_queue()
    service = WebhookIntegrationService(queue)
    return await service.handle_pull_request_webhook(payload)


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    # Example: Start queue and handle webhook
    
    async def example():
        # Start queue
        config = QueueConfig(
            max_concurrent=3,
            max_queue_size=50,
            task_timeout_seconds=300
        )
        
        queue_task = asyncio.create_task(start_review_queue(config))
        
        # Simulate webhook
        payload = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "additions": 250,
                "deletions": 50,
            },
            "repository": {
                "id": "repo_123",
                "full_name": "owner/repo",
            },
            "installation": {
                "id": 12345,
            }
        }
        
        task_id = await handle_pr_webhook(payload)
        print(f"Enqueued task: {task_id}")
        
        # Wait for task
        queue = get_review_queue()
        while True:
            status = await queue.get_status(task_id)
            if status and status.status == ReviewStatus.COMPLETED:
                print(f"Review completed: {status.review_id}")
                break
            elif status and status.status == ReviewStatus.FAILED:
                print(f"Review failed: {status.error}")
                break
            
            await asyncio.sleep(2)
        
        # Cleanup
        await stop_review_queue()
    
    asyncio.run(example())
