"""
API Endpoint for Webhook Integration (Phase 5.5)

This module provides FastAPI endpoint for triggering reviews from webhooks.

Endpoint: POST /review/trigger
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from webhook_integration import (
    handle_pr_webhook,
    get_review_queue,
    ReviewStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["review"])


# ========================================================================
# REQUEST/RESPONSE MODELS
# ========================================================================

class TriggerReviewRequest(BaseModel):
    """Request to trigger PR review"""
    repo_full_name: str
    pr_number: int
    installation_id: int
    repo_id: str
    pr_internal_id: str
    priority: Optional[str] = "medium"  # low, medium, high, critical


class TriggerReviewResponse(BaseModel):
    """Response from trigger review"""
    task_id: str
    status: str
    message: str


class ReviewStatusResponse(BaseModel):
    """Response with review status"""
    task_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int
    error: Optional[str] = None
    review_id: Optional[str] = None


# ========================================================================
# ENDPOINTS
# ========================================================================

@router.post("/trigger", response_model=TriggerReviewResponse)
async def trigger_review(
    request: TriggerReviewRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger AI code review for a pull request
    
    This endpoint is called by the frontend webhook handler when a PR is opened
    or synchronized (new commits pushed).
    
    Args:
        request: Review trigger request with PR details
        background_tasks: FastAPI background tasks
        
    Returns:
        TriggerReviewResponse with task ID for tracking
    """
    try:
        logger.info(
            f"Triggering review for {request.repo_full_name} "
            f"PR #{request.pr_number}"
        )
        
        # Create webhook payload
        payload = {
            "action": "opened",  # TODO: Pass actual action
            "pull_request": {
                "number": request.pr_number,
                "additions": 0,  # TODO: Fetch from GitHub or pass in request
                "deletions": 0,
            },
            "repository": {
                "id": request.repo_id,
                "full_name": request.repo_full_name,
            },
            "installation": {
                "id": request.installation_id,
            }
        }
        
        # Enqueue review
        task_id = await handle_pr_webhook(payload)
        
        return TriggerReviewResponse(
            task_id=task_id,
            status="pending",
            message=f"Review enqueued for {request.repo_full_name} PR #{request.pr_number}"
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger review: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=ReviewStatusResponse)
async def get_review_status(task_id: str):
    """
    Get status of a review task
    
    Args:
        task_id: Task ID from trigger_review response
        
    Returns:
        ReviewStatusResponse with current status
    """
    try:
        queue = get_review_queue()
        task = await queue.get_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return ReviewStatusResponse(
            task_id=task_id,
            status=task.status.value,
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            attempts=task.attempts,
            error=task.error,
            review_id=task.review_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get review status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/github")
async def handle_github_webhook(payload: Dict[str, Any]):
    """
    Handle GitHub webhook event
    
    This is an alternative endpoint that can be called directly from GitHub
    instead of going through the frontend.
    
    Args:
        payload: GitHub webhook payload
        
    Returns:
        Success message with task ID
    """
    try:
        event_type = payload.get("action")
        
        # Only handle PR events
        if "pull_request" not in payload:
            return {"message": "Not a pull request event", "status": "ignored"}
        
        # Enqueue review
        task_id = await handle_pr_webhook(payload)
        
        if not task_id:
            return {"message": "Event ignored", "status": "ignored"}
        
        return {
            "message": "Review enqueued",
            "status": "success",
            "task_id": task_id
        }
        
    except Exception as e:
        logger.error(f"Failed to handle webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
