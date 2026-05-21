"""
Temporary GraphRAG Context Builder
Builds in-memory graph and vector context for PR files without permanent storage
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from .logger import setup_logger
from .parser import UniversalParser
import hashlib

logger = setup_logger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TempNode:
    """Temporary graph node (function, class, etc.)"""
    id: str
    name: str
    type: str  # function, class, method, etc.
    file: str
    line: int
    end_line: int
    code: str
    language: str
    
    # Relationships
    calls: Set[str] = field(default_factory=set)  # Node IDs this calls
    called_by: Set[str] = field(default_factory=set)  # Node IDs that call this
    imports: Set[str] = field(default_factory=set)  # Module/file imports
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for context"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "code": self.code[:500],  # Truncate for context
            "language": self.language
        }


@dataclass
class TempFileNode:
    """Temporary file-level node"""
    path: str
    language: str
    nodes: List[TempNode] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    depends_on: Set[str] = field(default_factory=set)  # File dependencies


@dataclass
class TempVectorEntry:
    """Temporary vector entry for similarity search"""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def similarity(self, other: "TempVectorEntry") -> float:
        """Calculate cosine similarity with another entry"""
        if self.embedding is None or other.embedding is None:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(self.embedding, other.embedding))
        magnitude_a = sum(a * a for a in self.embedding) ** 0.5
        magnitude_b = sum(b * b for b in other.embedding) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)


# ============================================================================
# TEMPORARY GRAPH BUILDER
# ============================================================================

class TemporaryGraphBuilder:
    """
    Builds in-memory graph structure from PR files
    
    Features:
    - Parse files with tree-sitter
    - Extract nodes (functions, classes, methods)
    - Build call graphs and dependencies
    - Find relationships with existing code
    """
    
    def __init__(self, parser: UniversalParser):
        self.parser = parser
        self.nodes: Dict[str, TempNode] = {}  # node_id -> node
        self.files: Dict[str, TempFileNode] = {}  # file_path -> file_node
        self.node_index: Dict[str, List[str]] = defaultdict(list)  # name -> node_ids
        
    def process_file(self, filename: str, content: str, language: str) -> TempFileNode:
        """
        Parse and process a single file
        
        Args:
            filename: Relative file path
            content: File content
            language: Programming language
            
        Returns:
            TempFileNode with extracted nodes and relationships
        """
        logger.info(f"[TempGraph] Processing file: {filename} ({language})")
        
        # Create file node
        file_node = TempFileNode(path=filename, language=language)
        
        # Skip invalid/unsupported languages
        if not language or language == "text":
            logger.warning(f"[TempGraph] Invalid language '{language}' for {filename}, skipping")
            return file_node
        
        try:
            # Parse with tree-sitter
            content_bytes = content.encode('utf-8')
            captures, _ = self.parser.parse_code(content_bytes, language)
            
            if not captures:
                logger.warning(f"[TempGraph] No captures found for {filename}")
                return file_node
            
            # Extract nodes from captures
            for node, capture_name in captures:
                temp_node = self._create_temp_node(node, capture_name, filename, language, content_bytes)
                if temp_node:
                    self.nodes[temp_node.id] = temp_node
                    file_node.nodes.append(temp_node)
                    self.node_index[temp_node.name].append(temp_node.id)
            
            # Extract imports
            file_node.imports = self._extract_imports(content, language)
            
            # Build intra-file call graph
            self._build_call_graph(file_node)
            
            self.files[filename] = file_node
            logger.info(f"[TempGraph] Processed {filename}: {len(file_node.nodes)} nodes, {len(file_node.imports)} imports")
            
        except Exception as e:
            logger.error(f"[TempGraph] Error processing {filename}: {e}", exc_info=True)
        
        return file_node
    
    def _create_temp_node(
        self,
        tree_node: Any,
        capture_name: str,
        filename: str,
        language: str,
        content_bytes: bytes
    ) -> Optional[TempNode]:
        """Create a TempNode from tree-sitter node"""
        try:
            # Filter by node type (only functions, classes, methods)
            node_type = tree_node.type
            if node_type not in [
                "function_definition", "function_declaration", "function_item",
                "method_definition", "method_declaration",
                "class_definition", "class_declaration"
            ]:
                return None
            
            # Extract code
            code = content_bytes[tree_node.start_byte:tree_node.end_byte].decode('utf-8', errors='ignore')
            
            # Extract name (first line, cleaned)
            first_line = code.split('\n')[0].strip()
            name = first_line[:100]
            
            # Generate unique ID
            node_id = hashlib.md5(f"{filename}:{tree_node.start_point[0]}:{name}".encode()).hexdigest()[:16]
            
            return TempNode(
                id=node_id,
                name=name,
                type=node_type,
                file=filename,
                line=tree_node.start_point[0],
                end_line=tree_node.end_point[0],
                code=code,
                language=language
            )
            
        except Exception as e:
            logger.error(f"[TempGraph] Error creating temp node: {e}")
            return None
    
    def _extract_imports(self, content: str, language: str) -> Set[str]:
        """Extract import statements from file content"""
        imports = set()
        
        try:
            lines = content.split('\n')
            
            # Language-specific import patterns
            if language in ['python', 'py']:
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Extract module name
                        if line.startswith('import '):
                            module = line.replace('import ', '').split()[0].strip()
                        else:
                            module = line.split('from ')[1].split('import')[0].strip()
                        imports.add(module)
            
            elif language in ['javascript', 'typescript', 'jsx', 'tsx']:
                for line in lines:
                    line = line.strip()
                    if 'import' in line and 'from' in line:
                        # Extract module path
                        parts = line.split('from')
                        if len(parts) > 1:
                            module = parts[1].strip().strip(';').strip('"').strip("'")
                            imports.add(module)
            
        except Exception as e:
            logger.error(f"[TempGraph] Error extracting imports: {e}")
        
        return imports
    
    def _build_call_graph(self, file_node: TempFileNode):
        """Build call relationships within a file"""
        # Simple heuristic: look for function calls in code
        for node in file_node.nodes:
            for other_node in file_node.nodes:
                if node.id != other_node.id:
                    # Check if node's code mentions other_node's name
                    if other_node.name in node.code:
                        node.calls.add(other_node.id)
                        other_node.called_by.add(node.id)
    
    def build_dependencies(self):
        """Build file-level dependency graph"""
        for file_path, file_node in self.files.items():
            for import_path in file_node.imports:
                # Check if import corresponds to another file in the temporary graph
                if import_path in self.files:
                    file_node.depends_on.add(import_path)
    
    def get_node(self, node_id: str) -> Optional[TempNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_nodes_by_name(self, name: str) -> List[TempNode]:
        """Get all nodes with a given name"""
        node_ids = self.node_index.get(name, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_file_dependencies(self, filename: str) -> List[str]:
        """Get files that the given file depends on"""
        file_node = self.files.get(filename)
        if not file_node:
            return []
        return list(file_node.depends_on)
    
    def get_node_dependencies(self, node_id: str) -> List[Dict[str, Any]]:
        """Get nodes that the given node calls"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [
            self.nodes[called_id].to_dict()
            for called_id in node.calls
            if called_id in self.nodes
        ]
    
    def to_context_dict(self) -> Dict[str, Any]:
        """Convert temporary graph to context dictionary"""
        return {
            "files": {path: {
                "nodes": [n.to_dict() for n in fn.nodes],
                "imports": list(fn.imports),
                "dependencies": list(fn.depends_on)
            } for path, fn in self.files.items()},
            "total_nodes": len(self.nodes),
            "total_files": len(self.files)
        }


# ============================================================================
# TEMPORARY VECTOR BUILDER
# ============================================================================

class TemporaryVectorBuilder:
    """
    Builds in-memory vector index for PR files
    
    Features:
    - Generate embeddings using sentence-transformers
    - Similarity search
    - Find similar code across temporary and permanent contexts
    """
    
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.vectors: List[TempVectorEntry] = []
        self.index: Dict[str, int] = {}  # node_id -> vector index
        
    def add_nodes(self, nodes: List[TempNode]):
        """
        Add multiple nodes with batch embedding (OPTIMIZED)
        This is 10-20x faster than adding nodes one-by-one
        """
        if not nodes:
            return
        
        try:
            # Prepare texts for batch encoding
            texts = []
            node_list = []
            
            for node in nodes:
                # Create text for embedding (name + code snippet)
                text = f"{node.name}\n{node.code[:500]}"
                texts.append(text)
                node_list.append(node)
            
            # Batch encode all texts at once (MAJOR PERFORMANCE BOOST)
            logger.debug(f"[TempVector] Batch encoding {len(texts)} nodes...")
            embeddings = self.embed_model.encode(
                texts,
                batch_size=50,  # Process 50 at once
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Create vector entries
            for node, text, embedding in zip(node_list, texts, embeddings):
                entry = TempVectorEntry(
                    id=node.id,
                    text=text,
                    embedding=embedding.tolist(),
                    metadata={
                        "name": node.name,
                        "type": node.type,
                        "file": node.file,
                        "line": node.line,
                        "language": node.language
                    }
                )
                
                self.index[node.id] = len(self.vectors)
                self.vectors.append(entry)
            
            logger.debug(f"[TempVector] Batch encoded {len(embeddings)} nodes successfully")
            
        except Exception as e:
            logger.error(f"[TempVector] Error in batch encoding: {e}")
    
    def add_node(self, node: TempNode):
        """Add a single node (fallback for individual additions)"""
        try:
            # Create text for embedding (name + code snippet)
            text = f"{node.name}\n{node.code[:500]}"
            
            # Generate embedding
            embedding = self.embed_model.encode(text).tolist()
            
            # Create vector entry
            entry = TempVectorEntry(
                id=node.id,
                text=text,
                embedding=embedding,
                metadata={
                    "name": node.name,
                    "type": node.type,
                    "file": node.file,
                    "line": node.line,
                    "language": node.language
                }
            )
            
            self.index[node.id] = len(self.vectors)
            self.vectors.append(entry)
            
        except Exception as e:
            logger.error(f"[TempVector] Error adding node {node.id}: {e}")
    
    def search_similar(self, query: str, limit: int = 5, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar code using text query"""
        try:
            # Generate query embedding
            query_embedding = self.embed_model.encode(query).tolist()
            
            # Create temporary entry for similarity calculation
            query_entry = TempVectorEntry(id="query", text=query, embedding=query_embedding)
            
            # Calculate similarities
            results = []
            for entry in self.vectors:
                score = query_entry.similarity(entry)
                if score >= min_score:
                    results.append({
                        "score": score,
                        "node_id": entry.id,
                        "metadata": entry.metadata,
                        "text": entry.text[:200]
                    })
            
            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"[TempVector] Error searching: {e}")
            return []
    
    def find_similar_to_node(self, node_id: str, limit: int = 5, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Find nodes similar to a given node"""
        if node_id not in self.index:
            return []
        
        idx = self.index[node_id]
        source_entry = self.vectors[idx]
        
        results = []
        for entry in self.vectors:
            if entry.id != node_id:
                score = source_entry.similarity(entry)
                if score >= min_score:
                    results.append({
                        "score": score,
                        "node_id": entry.id,
                        "metadata": entry.metadata,
                        "text": entry.text[:200]
                    })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def to_context_dict(self) -> Dict[str, Any]:
        """Convert temporary vectors to context dictionary"""
        return {
            "total_vectors": len(self.vectors),
            "indexed_nodes": list(self.index.keys())
        }
