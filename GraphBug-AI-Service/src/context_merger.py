"""
Context Merger
Combines temporary (PR-only) and permanent (main branch) GraphRAG contexts
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .logger import setup_logger
from .temporary_graph import TemporaryGraphBuilder, TemporaryVectorBuilder, TempNode

logger = setup_logger(__name__)


@dataclass
class MergedContext:
    """Unified context combining temporary and permanent data"""
    
    # Node-level context
    dependencies: List[Dict[str, Any]]  # Functions/classes this depends on
    dependents: List[Dict[str, Any]]    # Functions/classes that depend on this
    similar_code: List[Dict[str, Any]]  # Similar code snippets
    
    # File-level context
    file_imports: List[str]
    file_dependencies: List[str]
    
    # Metadata
    source: str  # "temporary", "permanent", or "merged"
    temp_nodes_count: int = 0
    permanent_nodes_count: int = 0


class ContextMerger:
    """
    Merges temporary and permanent GraphRAG contexts
    
    Strategy:
    1. Check permanent database first (Neo4j + Qdrant)
    2. Check temporary in-memory graph second
    3. Combine results with deduplication
    4. Prioritize temporary results (newer code)
    """
    
    def __init__(
        self,
        temp_graph: Optional[TemporaryGraphBuilder] = None,
        temp_vector: Optional[TemporaryVectorBuilder] = None
    ):
        self.temp_graph = temp_graph
        self.temp_vector = temp_vector
        
    def merge_file_context(
        self,
        filename: str,
        permanent_context: Dict[str, Any],
        code_content: Optional[str] = None
    ) -> MergedContext:
        """
        Merge context for a specific file
        
        Args:
            filename: File path
            permanent_context: Context from permanent databases (Neo4j + Qdrant)
            code_content: Optional raw file content for on-the-fly parsing
            
        Returns:
            MergedContext with combined data
        """
        logger.info(f"[ContextMerger] Merging context for {filename}")
        
        # Extract permanent data
        perm_deps = permanent_context.get("dependencies", [])
        perm_dependents = permanent_context.get("dependents", [])
        perm_similar = permanent_context.get("similar_code", [])
        perm_imports = permanent_context.get("imports", [])
        perm_file_deps = permanent_context.get("file_dependencies", [])
        
        # Initialize temporary data containers
        temp_deps = []
        temp_dependents = []
        temp_similar = []
        temp_imports = []
        temp_file_deps = []
        
        # Check if file exists in temporary graph
        has_temp_data = False
        if self.temp_graph and filename in self.temp_graph.files:
            has_temp_data = True
            file_node = self.temp_graph.files[filename]
            
            # Get temporary imports and dependencies
            temp_imports = list(file_node.imports)
            temp_file_deps = self.temp_graph.get_file_dependencies(filename)
            
            # Get node-level dependencies from temporary graph
            for node in file_node.nodes:
                # Dependencies (what this node calls)
                node_deps = self.temp_graph.get_node_dependencies(node.id)
                temp_deps.extend(node_deps)
                
                # Similar code from temporary vectors
                if self.temp_vector:
                    similar = self.temp_vector.find_similar_to_node(
                        node.id,
                        limit=5,
                        min_score=0.7
                    )
                    temp_similar.extend(similar)
        
        # Merge with deduplication
        merged_dependencies = self._merge_lists(perm_deps, temp_deps, key="name")
        merged_dependents = self._merge_lists(perm_dependents, temp_dependents, key="name")
        merged_similar = self._merge_lists(perm_similar, temp_similar, key="node_id")
        merged_imports = list(set(perm_imports + temp_imports))
        merged_file_deps = list(set(perm_file_deps + temp_file_deps))
        
        # Determine source
        if has_temp_data and perm_deps:
            source = "merged"
        elif has_temp_data:
            source = "temporary"
        else:
            source = "permanent"
        
        return MergedContext(
            dependencies=merged_dependencies,
            dependents=merged_dependents,
            similar_code=merged_similar,
            file_imports=merged_imports,
            file_dependencies=merged_file_deps,
            source=source,
            temp_nodes_count=len(temp_deps) + len(temp_similar),
            permanent_nodes_count=len(perm_deps) + len(perm_similar)
        )
    
    def merge_similar_code_search(
        self,
        query: str,
        permanent_results: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merge similar code search results from permanent and temporary
        
        Args:
            query: Search query
            permanent_results: Results from Qdrant permanent database
            limit: Maximum results to return
            
        Returns:
            Merged list of similar code snippets
        """
        temp_results = []
        
        # Search temporary vectors if available
        if self.temp_vector:
            temp_results = self.temp_vector.search_similar(
                query,
                limit=limit,
                min_score=0.7
            )
            # Mark as temporary source
            for result in temp_results:
                result["source"] = "temporary"
        
        # Mark permanent results
        for result in permanent_results:
            result["source"] = "permanent"
        
        # Combine and sort by score
        all_results = permanent_results + temp_results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Deduplicate by text similarity
        deduplicated = self._deduplicate_similar_code(all_results)
        
        return deduplicated[:limit]
    
    def _merge_lists(
        self,
        list1: List[Dict[str, Any]],
        list2: List[Dict[str, Any]],
        key: str
    ) -> List[Dict[str, Any]]:
        """
        Merge two lists of dicts with deduplication by key
        
        Strategy: Prioritize items from list2 (temporary) over list1 (permanent)
        """
        seen = {}
        result = []
        
        # Add list2 first (temporary has priority)
        for item in list2:
            item_key = item.get(key)
            if item_key and item_key not in seen:
                seen[item_key] = True
                item["source"] = "temporary"
                result.append(item)
        
        # Add list1 (permanent) for items not in list2
        for item in list1:
            item_key = item.get(key)
            if item_key and item_key not in seen:
                seen[item_key] = True
                item["source"] = "permanent"
                result.append(item)
        
        return result
    
    def _deduplicate_similar_code(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate similar code entries based on text similarity"""
        if not results:
            return []
        
        deduplicated = [results[0]]
        
        for result in results[1:]:
            # Check if this result is too similar to any existing result
            is_duplicate = False
            for existing in deduplicated:
                if self._are_similar_texts(
                    result.get("text", ""),
                    existing.get("text", "")
                ):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _are_similar_texts(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are very similar (simple Jaccard similarity)"""
        if not text1 or not text2:
            return False
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) >= threshold if union > 0 else False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about temporary and permanent contexts"""
        stats = {
            "has_temporary_graph": self.temp_graph is not None,
            "has_temporary_vectors": self.temp_vector is not None
        }
        
        if self.temp_graph:
            stats.update({
                "temp_files": len(self.temp_graph.files),
                "temp_nodes": len(self.temp_graph.nodes)
            })
        
        if self.temp_vector:
            stats.update({
                "temp_vectors": len(self.temp_vector.vectors)
            })
        
        return stats
