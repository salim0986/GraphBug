"""
Data Cleanup Service - Handles orphaned data and rollback operations
Ensures data consistency across Neo4j, Qdrant, and PostgreSQL
"""

import asyncio
from typing import Optional, Dict, Any
from .logger import setup_logger
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder

logger = setup_logger(__name__)


class DataCleanup:
    """
    Manages cleanup operations for failed ingestions and deletions
    Ensures no orphaned data remains in any database
    """
    
    def __init__(self, graph_db: GraphBuilder, vector_db: VectorBuilder):
        self.graph_db = graph_db
        self.vector_db = vector_db
    
    async def cleanup_failed_ingestion(self, repo_id: str, reason: str = "unknown") -> Dict[str, Any]:
        """
        Rollback partial ingestion on failure
        
        This is called when ingestion fails mid-way to ensure no partial data remains.
        
        Args:
            repo_id: Repository identifier
            reason: Reason for failure (for logging)
            
        Returns:
            Dict with cleanup statistics
        """
        logger.warning(f"🧹 Starting cleanup for failed ingestion: {repo_id}")
        logger.warning(f"   Reason: {reason}")
        
        stats = {
            "repo_id": repo_id,
            "reason": reason,
            "vectors_deleted": 0,
            "graph_nodes_deleted": 0,
            "success": False
        }
        
        try:
            # 1. Delete vectors from Qdrant
            try:
                self.vector_db.delete_repo(repo_id)
                stats["vectors_deleted"] = "unknown"  # Qdrant doesn't return count
                logger.info(f"   ✓ Vectors cleaned up for {repo_id}")
            except Exception as e:
                logger.error(f"   ✗ Vector cleanup failed: {e}")
                raise
            
            # 2. Delete graph nodes from Neo4j
            try:
                deleted_count = self.graph_db.delete_repo(repo_id)
                stats["graph_nodes_deleted"] = deleted_count
                logger.info(f"   ✓ Graph nodes cleaned up: {deleted_count} nodes")
            except Exception as e:
                logger.error(f"   ✗ Graph cleanup failed: {e}")
                raise
            
            stats["success"] = True
            logger.info(f"✅ Cleanup complete for {repo_id}")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed for {repo_id}: {e}")
            stats["error"] = str(e)
        
        return stats
    
    async def verify_consistency(self, repo_id: str) -> Dict[str, Any]:
        """
        Verify data consistency across databases
        
        Checks that repository data exists in both Neo4j and Qdrant,
        or doesn't exist in either (no orphaned data).
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Dict with consistency status
        """
        logger.info(f"🔍 Verifying data consistency for {repo_id}")
        
        result = {
            "repo_id": repo_id,
            "consistent": False,
            "has_vectors": False,
            "has_graph_data": False,
            "issues": []
        }
        
        try:
            # Check vectors
            vector_stats = self.vector_db.get_repository_stats(repo_id)
            result["has_vectors"] = vector_stats.get("total_snippets", 0) > 0
            
            # Check graph
            graph_stats = self.graph_db.get_repo_node_count(repo_id)
            result["has_graph_data"] = graph_stats > 0
            
            # Consistency check
            if result["has_vectors"] == result["has_graph_data"]:
                result["consistent"] = True
                logger.info(f"   ✓ Data is consistent")
            else:
                result["consistent"] = False
                if result["has_vectors"] and not result["has_graph_data"]:
                    result["issues"].append("Orphaned vectors (no graph data)")
                elif result["has_graph_data"] and not result["has_vectors"]:
                    result["issues"].append("Orphaned graph data (no vectors)")
                logger.warning(f"   ⚠️ Inconsistency detected: {result['issues']}")
        
        except Exception as e:
            logger.error(f"❌ Consistency check failed: {e}")
            result["error"] = str(e)
        
        return result
    
    async def cleanup_orphaned_data(self) -> Dict[str, Any]:
        """
        Scan and cleanup any orphaned data across all repositories
        
        This is a maintenance operation that should be run periodically.
        
        Returns:
            Dict with cleanup statistics
        """
        logger.info("🔍 Scanning for orphaned data across all repositories")
        
        stats = {
            "repos_checked": 0,
            "inconsistencies_found": 0,
            "cleaned_up": 0,
            "errors": []
        }
        
        # This would need to iterate through all repos in PostgreSQL
        # and check consistency. Left as TODO for now.
        
        logger.info("⚠️ Global orphaned data cleanup not yet implemented")
        return stats


class IngestionCheckpoint:
    """
    Transaction-like behavior for ingestion operations
    Allows rollback to previous state on failure
    """
    
    def __init__(self, repo_id: str, cleanup: DataCleanup):
        self.repo_id = repo_id
        self.cleanup = cleanup
        self.created_at = None
        self.committed = False
    
    async def __aenter__(self):
        """Start checkpoint"""
        logger.info(f"📍 Creating ingestion checkpoint for {self.repo_id}")
        from datetime import datetime
        self.created_at = datetime.utcnow()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Rollback on exception, commit on success"""
        if exc_type is not None and not self.committed:
            # Exception occurred and not committed - rollback
            logger.error(f"❌ Ingestion failed, rolling back: {exc_val}")
            await self.cleanup.cleanup_failed_ingestion(
                self.repo_id,
                reason=f"{exc_type.__name__}: {exc_val}"
            )
            return False  # Re-raise exception
        elif self.committed:
            logger.info(f"✅ Checkpoint committed for {self.repo_id}")
        return False
    
    def commit(self):
        """Mark checkpoint as successful"""
        self.committed = True
        logger.info(f"✓ Committing checkpoint for {self.repo_id}")
