"""
Concurrent Ingestion Lock - Prevents multiple ingestions of same repository
"""

import asyncio
from typing import Dict, Optional
from datetime import datetime
from .logger import setup_logger

logger = setup_logger(__name__)


class IngestionLock:
    """
    Manages concurrent ingestion protection
    
    Prevents race conditions when multiple users/webhooks trigger ingestion
    of the same repository simultaneously.
    """
    
    def __init__(self):
        # Track: repo_id -> lock info
        self.locks: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def acquire(self, repo_id: str, installation_id: str) -> tuple[bool, Optional[str]]:
        """
        Try to acquire lock for repository ingestion
        
        Args:
            repo_id: Repository identifier
            installation_id: GitHub installation ID
            
        Returns:
            (acquired: bool, error_message: Optional[str])
        """
        async with self._lock:
            if repo_id in self.locks:
                lock_info = self.locks[repo_id]
                elapsed = (datetime.utcnow() - lock_info["started_at"]).total_seconds()
                
                # If lock is older than 30 minutes, assume stale and release
                if elapsed > 1800:  # 30 minutes
                    logger.warning(
                        f"Stale lock detected for {repo_id} "
                        f"(held for {int(elapsed)}s), releasing"
                    )
                    del self.locks[repo_id]
                else:
                    error_msg = (
                        f"Repository {repo_id} is already being processed. "
                        f"Started {int(elapsed)}s ago by installation {lock_info['installation_id']}. "
                        f"Please wait or try again later."
                    )
                    logger.info(f"Lock acquisition failed: {error_msg}")
                    return False, error_msg
            
            # Acquire lock
            self.locks[repo_id] = {
                "installation_id": installation_id,
                "started_at": datetime.utcnow()
            }
            logger.info(
                f"🔒 Lock acquired for {repo_id} "
                f"(installation {installation_id})"
            )
            return True, None
    
    async def release(self, repo_id: str):
        """Release lock for repository"""
        async with self._lock:
            if repo_id in self.locks:
                lock_info = self.locks[repo_id]
                elapsed = (datetime.utcnow() - lock_info["started_at"]).total_seconds()
                del self.locks[repo_id]
                logger.info(
                    f"🔓 Lock released for {repo_id} "
                    f"(held for {int(elapsed)}s)"
                )
            else:
                logger.warning(f"Attempted to release non-existent lock: {repo_id}")
    
    async def is_locked(self, repo_id: str) -> bool:
        """Check if repository is currently locked"""
        async with self._lock:
            return repo_id in self.locks
    
    async def get_lock_info(self, repo_id: str) -> Optional[Dict]:
        """Get information about current lock"""
        async with self._lock:
            if repo_id in self.locks:
                lock_info = self.locks[repo_id].copy()
                lock_info["elapsed_seconds"] = (
                    datetime.utcnow() - lock_info["started_at"]
                ).total_seconds()
                return lock_info
            return None
    
    async def cleanup_stale_locks(self, max_age_seconds: int = 1800):
        """Remove locks older than max_age_seconds (default 30min)"""
        async with self._lock:
            current_time = datetime.utcnow()
            stale_repos = []
            
            for repo_id, lock_info in self.locks.items():
                elapsed = (current_time - lock_info["started_at"]).total_seconds()
                if elapsed > max_age_seconds:
                    stale_repos.append(repo_id)
            
            for repo_id in stale_repos:
                elapsed = (current_time - self.locks[repo_id]["started_at"]).total_seconds()
                logger.warning(
                    f"Removing stale lock for {repo_id} "
                    f"(age: {int(elapsed)}s)"
                )
                del self.locks[repo_id]
            
            if stale_repos:
                logger.info(f"Cleaned up {len(stale_repos)} stale locks")
            
            return len(stale_repos)
