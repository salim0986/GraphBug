"""
Incremental Ingestion System with Performance Optimizations

Features:
- Git diff detection for changed files only
- Batch processing for embeddings (50+ at once)
- Parallel file parsing with asyncio
- Smart caching and upsert operations
- Memory-efficient processing
"""

import os
import git
import asyncio
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from .logger import setup_logger
from .parser import UniversalParser, EXTENSION_MAP
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder

logger = setup_logger(__name__)


class IncrementalIngester:
    """
    High-performance incremental ingestion engine
    
    Optimizations:
    1. Only processes changed files (git diff)
    2. Batches embeddings (50+ functions per API call)
    3. Parallel file parsing with asyncio
    4. Upserts instead of full deletion
    5. Memory-efficient streaming
    """
    
    def __init__(
        self,
        parser: UniversalParser,
        graph_db: GraphBuilder,
        vector_db: VectorBuilder,
        ignore_dirs: Set[str]
    ):
        self.parser = parser
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.ignore_dirs = ignore_dirs
        self.batch_size = 50  # Embed 50 functions at once
        
    async def ingest_incremental(
        self,
        repo_id: str,
        local_path: str,
        last_commit: Optional[str] = None,
        current_commit: str = "HEAD"
    ) -> Dict[str, int]:
        """
        Incrementally ingest only changed files
        
        Args:
            repo_id: Unique repository identifier
            local_path: Path to local git repository
            last_commit: Previous commit SHA (None for full ingestion)
            current_commit: Current commit SHA (default: HEAD)
            
        Returns:
            Dict with stats: files_processed, nodes_added, nodes_deleted, vectors_updated
        """
        logger.info("=" * 80)
        logger.info(f"🔄 INCREMENTAL INGESTION STARTING")
        logger.info(f"   Repo ID: {repo_id}")
        logger.info(f"   Local path: {local_path}")
        logger.info(f"   Range: {last_commit or 'initial'} → {current_commit}")
        logger.info("=" * 80)
        
        stats = {
            "files_processed": 0,
            "files_deleted": 0,
            "nodes_added": 0,
            "nodes_deleted": 0,
            "vectors_updated": 0
        }
        
        try:
            repo = git.Repo(local_path)
            
            # Get changed files
            if last_commit:
                changed_files, deleted_files = self._get_changed_files(
                    repo, last_commit, current_commit
                )
            else:
                # Full ingestion - process all files
                logger.info("📁 Full ingestion mode - scanning all files...")
                changed_files = self._get_all_files(local_path)
                deleted_files = set()
            
            logger.info(f"📊 Changed files: {len(changed_files)}, Deleted files: {len(deleted_files)}")
            
            # Process deleted files first
            if deleted_files:
                logger.info(f"🗑️ Processing {len(deleted_files)} deleted files...")
            for file_path in deleted_files:
                await self._delete_file_data(repo_id, file_path)
                stats["files_deleted"] += 1
                stats["nodes_deleted"] += 1  # Approximate
            
            # Process changed files in parallel batches
            if changed_files:
                logger.info(f"⚙️ Processing {len(changed_files)} changed files in parallel batches...")
            batch_size = 10  # Parse 10 files in parallel
            for i in range(0, len(changed_files), batch_size):
                batch = changed_files[i:i + batch_size]
                logger.info(f"   Batch {i//batch_size + 1}/{(len(changed_files) + batch_size - 1)//batch_size}: Processing {len(batch)} files...")
                batch_stats = await self._process_file_batch(
                    repo_id, local_path, batch
                )
                stats["files_processed"] += batch_stats["files_processed"]
                stats["nodes_added"] += batch_stats["nodes_added"]
                stats["vectors_updated"] += batch_stats["vectors_updated"]
            
            # Rebuild dependencies for changed files only
            logger.info("🔗 Rebuilding dependencies...")
            self.graph_db.build_dependencies(repo_id)
            
            logger.info("=" * 80)
            logger.info(f"✅ INCREMENTAL INGESTION COMPLETE")
            logger.info(f"   Files processed: {stats['files_processed']}")
            logger.info(f"   Files deleted: {stats['files_deleted']}")
            logger.info(f"   Nodes added: {stats['nodes_added']}")
            logger.info(f"   Nodes deleted: {stats['nodes_deleted']}")
            logger.info(f"   Vectors updated: {stats['vectors_updated']}")
            logger.info("=" * 80)
            return stats
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ INCREMENTAL INGESTION FAILED")
            logger.error(f"   Repo ID: {repo_id}")
            logger.error(f"   Error: {e}")
            logger.error("=" * 80)
            logger.error(f"Full traceback:", exc_info=True)
            raise
    
    def _get_changed_files(
        self,
        repo: git.Repo,
        from_commit: str,
        to_commit: str
    ) -> Tuple[List[str], Set[str]]:
        """
        Get list of changed and deleted files between two commits
        
        Returns:
            (changed_files, deleted_files)
        """
        try:
            # Get diff between commits
            diff = repo.commit(from_commit).diff(to_commit)
            
            changed_files = []
            deleted_files = set()
            
            for change in diff:
                # change.a_path is the file path
                file_path = change.a_path or change.b_path
                
                # Filter by extension
                if not self._is_valid_file(file_path):
                    continue
                
                if change.deleted_file:
                    deleted_files.add(file_path)
                else:
                    # Modified or added file
                    changed_files.append(file_path)
            
            return changed_files, deleted_files
            
        except Exception as e:
            logger.error(f"Error getting changed files: {e}")
            return [], set()
    
    def _get_all_files(self, local_path: str) -> List[str]:
        """Get all valid files for full ingestion"""
        all_files = []
        
        for root, dirs, files in os.walk(local_path):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_path)
                
                if self._is_valid_file(rel_path):
                    all_files.append(rel_path)
        
        return all_files
    
    def _is_valid_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        return filename in EXTENSION_MAP or ext in EXTENSION_MAP
    
    async def _delete_file_data(self, repo_id: str, file_path: str):
        """Delete all data for a file"""
        try:
            # Delete from graph
            file_uid = f"{repo_id}::{file_path}"
            self.graph_db.delete_file(file_uid)
            
            # Delete from vectors
            self.vector_db.delete_by_file(repo_id, file_path)
            
            logger.debug(f"Deleted data for {file_path}")
            
        except Exception as e:
            logger.error(f"Error deleting file data for {file_path}: {e}")
    
    async def _process_file_batch(
        self,
        repo_id: str,
        base_path: str,
        file_paths: List[str]
    ) -> Dict[str, int]:
        """
        Process a batch of files in parallel
        
        Returns stats for the batch
        """
        stats = {
            "files_processed": 0,
            "nodes_added": 0,
            "vectors_updated": 0
        }
        
        # Parse files in parallel
        parse_tasks = [
            self._parse_file(base_path, file_path)
            for file_path in file_paths
        ]
        
        parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)
        
        # Collect all nodes for batch embedding
        all_nodes_for_embedding = []
        
        for file_path, result in zip(file_paths, parse_results):
            if isinstance(result, Exception):
                logger.error(f"Error parsing {file_path}: {result}")
                continue
            
            if result is None:
                continue
            
            captures, code_bytes = result
            
            if not captures:
                continue
            
            stats["files_processed"] += 1
            
            # Update graph
            try:
                self.graph_db.process_file_nodes(
                    repo_id, file_path, captures, code_bytes
                )
                stats["nodes_added"] += len(captures)
            except Exception as e:
                logger.error(f"Error updating graph for {file_path}: {e}")
                continue
            
            # Collect nodes for batch embedding
            for node, capture_name in captures:
                # Apply filters (same as original)
                lines_of_code = node.end_point[0] - node.start_point[0]
                if lines_of_code < 3:
                    continue
                
                node_type = node.type
                ALLOWED_TYPES = [
                    "function_definition", "function_declaration", "function_item",
                    "method_definition", "method_declaration",
                    "class_definition", "class_declaration",
                    "interface_declaration", "impl_item",
                    "table", "block"
                ]
                
                _, ext = os.path.splitext(file_path)
                is_logic_node = node_type in ALLOWED_TYPES
                is_config_file = ext in [".yaml", ".yml", ".toml", ".tf", ".dockerfile"]
                
                if not (is_logic_node or (is_config_file and lines_of_code > 0)):
                    continue
                
                try:
                    func_code = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
                    first_line = func_code.splitlines()[0] if func_code.splitlines() else "unknown"
                    func_name = first_line[:100].strip()
                    
                    all_nodes_for_embedding.append({
                        "repo_id": repo_id,
                        "func_name": func_name,
                        "func_code": func_code,
                        "file_path": file_path,
                        "start_line": node.start_point[0]
                    })
                    
                except Exception as e:
                    logger.warning(f"Error preparing node for embedding: {e}")
        
        # Batch embed all collected nodes
        if all_nodes_for_embedding:
            embedded_count = await self._batch_embed_nodes(all_nodes_for_embedding)
            stats["vectors_updated"] = embedded_count
        
        return stats
    
    async def _parse_file(
        self,
        base_path: str,
        file_path: str
    ) -> Optional[Tuple]:
        """Parse a single file asynchronously"""
        try:
            full_path = os.path.join(base_path, file_path)
            
            # Run sync parsing in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.parser.parse_file,
                full_path
            )
            
        except Exception as e:
            logger.error(f"Parse error for {file_path}: {e}")
            return None
    
    async def _batch_embed_nodes(self, nodes: List[Dict]) -> int:
        """
        Embed nodes in batches for performance
        
        Args:
            nodes: List of node dictionaries with code and metadata
            
        Returns:
            Number of vectors created
        """
        embedded_count = 0
        
        try:
            # Process in batches of self.batch_size
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i + self.batch_size]
                
                # Extract code snippets for batch encoding
                code_snippets = [node["func_code"] for node in batch]
                
                # Batch embed (this is much faster than one-by-one)
                embeddings = self.vector_db.embed_model.encode(
                    code_snippets,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
                
                # Store vectors with metadata
                for node, embedding in zip(batch, embeddings):
                    try:
                        self.vector_db.upsert_function_vector(
                            repo_id=node["repo_id"],
                            func_name=node["func_name"],
                            embedding=embedding,
                            file_path=node["file_path"],
                            start_line=node["start_line"],
                            raw_code=node["func_code"]
                        )
                        embedded_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error storing vector: {e}")
                
                logger.debug(f"Batch embedded {len(batch)} nodes")
        
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
        
        return embedded_count


async def ingest_repo_incremental(
    repo_id: str,
    repo_url: str,
    local_path: str,
    parser: UniversalParser,
    graph_db: GraphBuilder,
    vector_db: VectorBuilder,
    ignore_dirs: Set[str],
    last_commit: Optional[str] = None
) -> Dict[str, int]:
    """
    High-level function to perform incremental ingestion
    
    Args:
        repo_id: Repository identifier
        repo_url: Git repository URL
        local_path: Local clone path
        parser: UniversalParser instance
        graph_db: GraphBuilder instance
        vector_db: VectorBuilder instance
        ignore_dirs: Set of directories to ignore
        last_commit: Previous commit SHA (None for full ingestion)
        
    Returns:
        Stats dictionary
    """
    ingester = IncrementalIngester(parser, graph_db, vector_db, ignore_dirs)
    
    # Ensure collection exists
    vector_db.ensure_collection()
    
    # Run incremental ingestion
    stats = await ingester.ingest_incremental(
        repo_id=repo_id,
        local_path=local_path,
        last_commit=last_commit
    )
    
    return stats
