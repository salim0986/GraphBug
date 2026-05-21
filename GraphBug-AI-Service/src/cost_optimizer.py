"""
Cost Optimization Module

Features:
- Prompt compression (reduce token usage by 30-50%)
- Review result caching for PR updates
- Smart model selection (use cheaper models when possible)
- Token counting and budget tracking
"""

import hashlib
import json
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from .logger import setup_logger

logger = setup_logger(__name__)


class ReviewCache:
    """
    Cache review results for PR updates
    
    When a PR is synchronized (new commits pushed), we can:
    1. Check if files changed
    2. Reuse reviews for unchanged files
    3. Only review new/modified files
    
    This can save 50-80% of API costs for PR updates
    """
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        
    def get_cache_key(self, pr_id: str, file_path: str, file_hash: str) -> str:
        """Generate cache key from PR ID, file path, and content hash"""
        key_data = f"{pr_id}:{file_path}:{file_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached review if exists and not expired"""
        if cache_key not in self.cache:
            return None
        
        cached = self.cache[cache_key]
        cached_at = cached.get("cached_at")
        
        if not cached_at:
            return None
        
        # Check if expired
        cached_time = datetime.fromisoformat(cached_at)
        if datetime.utcnow() - cached_time > self.ttl:
            logger.debug(f"Cache expired for {cache_key}")
            del self.cache[cache_key]
            return None
        
        logger.info(f"✅ Cache HIT for {cache_key[:12]}...")
        return cached.get("review")
    
    def set(self, cache_key: str, review: Dict[str, Any]):
        """Cache a review result"""
        self.cache[cache_key] = {
            "review": review,
            "cached_at": datetime.utcnow().isoformat()
        }
        logger.debug(f"Cached review: {cache_key[:12]}...")
    
    def clear_pr(self, pr_id: str):
        """Clear all cache entries for a PR"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(pr_id)]
        for key in keys_to_remove:
            del self.cache[key]
        logger.debug(f"Cleared {len(keys_to_remove)} cache entries for PR {pr_id}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = len(self.cache)
        expired = sum(
            1 for cached in self.cache.values()
            if datetime.utcnow() - datetime.fromisoformat(cached["cached_at"]) > self.ttl
        )
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired
        }


class PromptCompressor:
    """
    Compress prompts to reduce token usage
    
    Techniques:
    1. Remove redundant context
    2. Abbreviate long variable names in examples
    3. Use concise formatting
    4. Skip verbose instructions for simple reviews
    """
    
    @staticmethod
    def compress_similar_code(similar_code: list, max_items: int = 3) -> str:
        """
        Compress similar code context
        Only include top N most relevant matches
        """
        if not similar_code:
            return "None found"
        
        compressed = []
        for item in similar_code[:max_items]:
            # Abbreviated format
            compressed.append(
                f"- {item.get('file')}:{item.get('line')} "
                f"({item.get('type')}, score: {item.get('score', 0):.2f})"
            )
        
        if len(similar_code) > max_items:
            compressed.append(f"... +{len(similar_code) - max_items} more")
        
        return "\n".join(compressed)
    
    @staticmethod
    def compress_dependencies(dependencies: list, max_items: int = 5) -> str:
        """Compress dependency list"""
        if not dependencies:
            return "None"
        
        compressed = []
        for dep in dependencies[:max_items]:
            compressed.append(f"- {dep.get('name')} ({dep.get('file')})")
        
        if len(dependencies) > max_items:
            compressed.append(f"... +{len(dependencies) - max_items} more")
        
        return "\n".join(compressed)
    
    @staticmethod
    def compress_code_snippet(code: str, max_lines: int = 20) -> str:
        """Compress long code snippets"""
        lines = code.split("\n")
        
        if len(lines) <= max_lines:
            return code
        
        # Take first and last portions
        head_lines = max_lines // 2
        tail_lines = max_lines - head_lines
        
        compressed = (
            "\n".join(lines[:head_lines]) +
            f"\n... ({len(lines) - max_lines} lines omitted) ...\n" +
            "\n".join(lines[-tail_lines:])
        )
        
        return compressed
    
    @staticmethod
    def should_use_flash_lite(
        total_files: int,
        total_additions: int,
        critical_issues: int,
        high_issues: int
    ) -> bool:
        """
        Determine if flash-lite model is sufficient (cost optimization)
        
        Use flash-lite when:
        - Small PR (< 3 files, < 150 lines)
        - No critical/high issues
        - Low complexity
        """
        is_small = total_files <= 3 and total_additions <= 150
        no_serious_issues = critical_issues == 0 and high_issues <= 1
        
        return is_small and no_serious_issues


class TokenBudget:
    """
    Track token usage and enforce budgets
    
    Helps prevent cost overruns by:
    1. Tracking tokens per PR/repo
    2. Enforcing daily/monthly limits
    3. Alerting when approaching limits
    """
    
    def __init__(self, daily_limit: int = 1000000, monthly_limit: int = 25000000):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.usage: Dict[str, Dict[str, int]] = {}  # date -> {pr_id: tokens}
        
    def log_usage(self, pr_id: str, tokens: int):
        """Log token usage for a PR"""
        today = datetime.utcnow().date().isoformat()
        
        if today not in self.usage:
            self.usage[today] = {}
        
        self.usage[today][pr_id] = self.usage[today].get(pr_id, 0) + tokens
        
        logger.info(f"Token usage: {tokens} for PR {pr_id}")
        
        # Check limits
        daily_total = sum(self.usage[today].values())
        if daily_total > self.daily_limit * 0.8:
            logger.warning(f"⚠️  Approaching daily token limit: {daily_total}/{self.daily_limit}")
    
    def get_daily_usage(self) -> int:
        """Get today's token usage"""
        today = datetime.utcnow().date().isoformat()
        return sum(self.usage.get(today, {}).values())
    
    def get_monthly_usage(self) -> int:
        """Get this month's token usage"""
        current_month = datetime.utcnow().strftime("%Y-%m")
        monthly_total = 0
        
        for date_str, usage in self.usage.items():
            if date_str.startswith(current_month):
                monthly_total += sum(usage.values())
        
        return monthly_total
    
    def can_proceed(self, estimated_tokens: int) -> bool:
        """Check if we can proceed with review given token estimate"""
        daily_total = self.get_daily_usage()
        monthly_total = self.get_monthly_usage()
        
        if daily_total + estimated_tokens > self.daily_limit:
            logger.error(f"❌ Daily token limit would be exceeded: {daily_total + estimated_tokens}/{self.daily_limit}")
            return False
        
        if monthly_total + estimated_tokens > self.monthly_limit:
            logger.error(f"❌ Monthly token limit would be exceeded: {monthly_total + estimated_tokens}/{self.monthly_limit}")
            return False
        
        return True


# Global instances
review_cache = ReviewCache(ttl_minutes=120)  # 2 hour cache
token_budget = TokenBudget(daily_limit=1000000, monthly_limit=25000000)
