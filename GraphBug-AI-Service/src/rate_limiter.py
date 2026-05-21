"""
Rate Limiting Middleware
Protects API from abuse and ensures fair usage across installations
"""

import time
from typing import Dict, Optional
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from .logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with per-installation tracking
    
    Features:
    - Separate limits for different endpoint types
    - Installation-based tracking
    - Graceful degradation (warning logs instead of hard blocks for now)
    """
    
    def __init__(self):
        # Track: installation_id -> {endpoint_type: [(timestamp, tokens_used)]}
        self.buckets: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        
        # Rate limits: requests per minute
        self.limits = {
            "ingestion": 10,      # Heavy operations
            "search": 60,         # Medium operations
            "review": 30,         # Heavy AI operations  
            "webhook": 100,       # Lightweight webhooks
            "default": 120        # General API calls
        }
        
        # Cleanup old entries every 5 minutes
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _cleanup_old_entries(self):
        """Remove entries older than 1 minute"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 60  # 1 minute ago
        
        for installation_id in list(self.buckets.keys()):
            for endpoint_type in list(self.buckets[installation_id].keys()):
                self.buckets[installation_id][endpoint_type] = [
                    (ts, tokens) for ts, tokens in self.buckets[installation_id][endpoint_type]
                    if ts > cutoff_time
                ]
                
                # Remove empty endpoint types
                if not self.buckets[installation_id][endpoint_type]:
                    del self.buckets[installation_id][endpoint_type]
            
            # Remove empty installations
            if not self.buckets[installation_id]:
                del self.buckets[installation_id]
        
        self.last_cleanup = current_time
        logger.debug(f"Cleaned up rate limiter, {len(self.buckets)} installations tracked")
    
    def check_rate_limit(
        self, 
        installation_id: str, 
        endpoint_type: str = "default",
        tokens: int = 1
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limit
        
        Args:
            installation_id: GitHub installation ID (or "anonymous")
            endpoint_type: Type of endpoint (ingestion, search, review, webhook, default)
            tokens: Number of tokens to consume (default 1)
            
        Returns:
            (allowed: bool, error_message: Optional[str])
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute window
        
        # Get recent requests
        recent_requests = [
            (ts, t) for ts, t in self.buckets[installation_id][endpoint_type]
            if ts > cutoff_time
        ]
        
        # Count tokens used in last minute
        tokens_used = sum(t for _, t in recent_requests)
        limit = self.limits.get(endpoint_type, self.limits["default"])
        
        if tokens_used + tokens > limit:
            remaining_time = 60 - (current_time - recent_requests[0][0]) if recent_requests else 0
            error_msg = (
                f"Rate limit exceeded for {endpoint_type}. "
                f"Limit: {limit} requests/min. "
                f"Current: {tokens_used}. "
                f"Retry in {int(remaining_time)}s"
            )
            logger.warning(f"🚫 {error_msg} | Installation: {installation_id}")
            return False, error_msg
        
        # Record this request
        self.buckets[installation_id][endpoint_type].append((current_time, tokens))
        
        # Log if approaching limit (>80%)
        if tokens_used + tokens > limit * 0.8:
            logger.warning(
                f"⚠️ Installation {installation_id} approaching rate limit: "
                f"{tokens_used + tokens}/{limit} for {endpoint_type}"
            )
        
        return True, None
    
    def get_usage_stats(self, installation_id: str) -> Dict[str, int]:
        """Get current usage statistics for an installation"""
        current_time = time.time()
        cutoff_time = current_time - 60
        
        stats = {}
        for endpoint_type, requests in self.buckets[installation_id].items():
            recent = [t for ts, t in requests if ts > cutoff_time]
            stats[endpoint_type] = sum(recent)
        
        return stats


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting
    
    Applies rate limits based on endpoint path and installation ID
    """
    
    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    def _get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type from request path"""
        if "/ingest" in path or "/process" in path:
            return "ingestion"
        elif "/search" in path or "/context" in path:
            return "search"
        elif "/review" in path or "/generate" in path:
            return "review"
        elif "/webhook" in path:
            return "webhook"
        else:
            return "default"
    
    def _get_installation_id(self, request: Request) -> str:
        """Extract installation ID from request"""
        # Try query parameter
        installation_id = request.query_params.get("installation_id")
        if installation_id:
            return installation_id
        
        # Try request body (if POST/PUT)
        if hasattr(request.state, "body"):
            body = request.state.body
            if isinstance(body, dict):
                installation_id = body.get("installation_id")
                if installation_id:
                    return installation_id
        
        # Try headers
        installation_id = request.headers.get("X-Installation-ID")
        if installation_id:
            return installation_id
        
        # Fallback to IP address
        return request.client.host if request.client else "anonymous"
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter"""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        endpoint_type = self._get_endpoint_type(request.url.path)
        installation_id = self._get_installation_id(request)
        
        # Check rate limit
        allowed, error_msg = self.limiter.check_rate_limit(
            installation_id, 
            endpoint_type
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": error_msg,
                    "installation_id": installation_id,
                    "endpoint_type": endpoint_type
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.limiter.limits.get(endpoint_type, 120)),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        usage = self.limiter.get_usage_stats(installation_id)
        limit = self.limiter.limits.get(endpoint_type, 120)
        current_usage = usage.get(endpoint_type, 0)
        
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_usage))
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
