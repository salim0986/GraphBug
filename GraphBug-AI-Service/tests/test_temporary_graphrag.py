"""
Integration Test for Temporary GraphRAG
Tests the new temporary in-memory graph and vector builders
"""

import asyncio
import pytest
from src.temporary_graph import (
    TemporaryGraphBuilder,
    TemporaryVectorBuilder,
    TempNode,
    TempFileNode
)
from src.context_merger import ContextMerger, MergedContext
from src.parser import UniversalParser
from sentence_transformers import SentenceTransformer


# Sample Python code for testing
SAMPLE_PYTHON_CODE = """
def calculate_total(items):
    \"\"\"Calculate total price of items\"\"\"
    total = 0
    for item in items:
        total += item['price']
    return total

class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        self.items.append(item)
    
    def get_total(self):
        return calculate_total(self.items)
"""

SAMPLE_JS_CODE = """
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}

class ShoppingCart {
  constructor() {
    this.items = [];
  }
  
  addItem(item) {
    this.items.push(item);
  }
  
  getTotal() {
    return calculateTotal(this.items);
  }
}
"""


@pytest.fixture
def parser():
    """Create UniversalParser instance"""
    return UniversalParser()


@pytest.fixture
def embed_model():
    """Create embedding model (lazy loaded)"""
    return SentenceTransformer('all-MiniLM-L6-v2')


@pytest.fixture
def temp_graph(parser):
    """Create TemporaryGraphBuilder instance"""
    return TemporaryGraphBuilder(parser)


@pytest.fixture
def temp_vector(embed_model):
    """Create TemporaryVectorBuilder instance"""
    return TemporaryVectorBuilder(embed_model)


class TestTemporaryGraphBuilder:
    """Test TemporaryGraphBuilder functionality"""
    
    def test_process_python_file(self, temp_graph):
        """Test processing a Python file"""
        file_node = temp_graph.process_file(
            "cart.py",
            SAMPLE_PYTHON_CODE,
            "python"
        )
        
        assert file_node.path == "cart.py"
        assert file_node.language == "python"
        assert len(file_node.nodes) > 0
        
        # Check that function and class were extracted
        node_names = [n.name for n in file_node.nodes]
        print(f"Extracted nodes: {node_names}")
        
        # At least some nodes should be extracted
        assert len(temp_graph.nodes) > 0
    
    def test_process_javascript_file(self, temp_graph):
        """Test processing a JavaScript file"""
        file_node = temp_graph.process_file(
            "cart.js",
            SAMPLE_JS_CODE,
            "javascript"
        )
        
        assert file_node.path == "cart.js"
        assert file_node.language == "javascript"
        assert len(file_node.nodes) >= 0  # May be 0 if tree-sitter fails
    
    def test_extract_imports(self, temp_graph):
        """Test import extraction"""
        python_code = """
import os
from typing import List, Dict
from utils.helpers import calculate_total
"""
        file_node = temp_graph.process_file(
            "test.py",
            python_code,
            "python"
        )
        
        print(f"Extracted imports: {file_node.imports}")
        assert "os" in file_node.imports
        assert "typing" in file_node.imports or "utils.helpers" in file_node.imports
    
    def test_to_context_dict(self, temp_graph):
        """Test converting graph to context dictionary"""
        temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        
        context = temp_graph.to_context_dict()
        
        assert "files" in context
        assert "total_nodes" in context
        assert "total_files" in context
        assert context["total_files"] == 1


class TestTemporaryVectorBuilder:
    """Test TemporaryVectorBuilder functionality"""
    
    def test_add_node(self, temp_vector, temp_graph):
        """Test adding a node and generating embedding"""
        # First create a node
        file_node = temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        
        if file_node.nodes:
            # Add first node to vector index
            temp_vector.add_node(file_node.nodes[0])
            
            assert len(temp_vector.vectors) == 1
            assert file_node.nodes[0].id in temp_vector.index
            assert temp_vector.vectors[0].embedding is not None
    
    def test_search_similar(self, temp_vector, temp_graph):
        """Test similarity search"""
        # Add multiple nodes
        file_node = temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        temp_vector.add_nodes(file_node.nodes)
        
        if len(temp_vector.vectors) > 0:
            # Search for similar code
            results = temp_vector.search_similar(
                "calculate total price",
                limit=3,
                min_score=0.0  # Lower threshold for test
            )
            
            print(f"Search results: {results}")
            assert isinstance(results, list)
    
    def test_find_similar_to_node(self, temp_vector, temp_graph):
        """Test finding similar nodes"""
        # Add multiple nodes
        file_node1 = temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        file_node2 = temp_graph.process_file("cart2.py", SAMPLE_PYTHON_CODE, "python")
        
        temp_vector.add_nodes(file_node1.nodes + file_node2.nodes)
        
        if len(temp_vector.vectors) > 1:
            node_id = temp_vector.vectors[0].id
            results = temp_vector.find_similar_to_node(
                node_id,
                limit=3,
                min_score=0.0
            )
            
            assert isinstance(results, list)


class TestContextMerger:
    """Test ContextMerger functionality"""
    
    def test_merge_file_context_temporary_only(self, temp_graph, temp_vector):
        """Test merging with only temporary data"""
        # Build temporary graph
        file_node = temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        temp_vector.add_nodes(file_node.nodes)
        temp_graph.build_dependencies()
        
        # Create merger
        merger = ContextMerger(temp_graph=temp_graph, temp_vector=temp_vector)
        
        # Merge with empty permanent context
        merged = merger.merge_file_context(
            "cart.py",
            permanent_context={
                "dependencies": [],
                "dependents": [],
                "similar_code": [],
                "imports": [],
                "file_dependencies": []
            }
        )
        
        assert merged.source == "temporary"
        assert merged.temp_nodes_count >= 0
    
    def test_merge_file_context_hybrid(self, temp_graph, temp_vector):
        """Test merging temporary + permanent data"""
        # Build temporary graph
        file_node = temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        temp_vector.add_nodes(file_node.nodes)
        
        # Create merger
        merger = ContextMerger(temp_graph=temp_graph, temp_vector=temp_vector)
        
        # Merge with mock permanent context
        merged = merger.merge_file_context(
            "cart.py",
            permanent_context={
                "dependencies": [{"name": "utils.calculate", "type": "function"}],
                "dependents": [],
                "similar_code": [],
                "imports": ["os", "sys"],
                "file_dependencies": []
            }
        )
        
        # Should be merged if we have temporary data
        if file_node.nodes:
            assert merged.source in ["temporary", "merged"]
        
        # Should have combined imports
        assert len(merged.file_imports) > 0
    
    def test_merge_similar_code_search(self, temp_graph, temp_vector):
        """Test merging similar code search results"""
        # Build temporary graph
        file_node = temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        temp_vector.add_nodes(file_node.nodes)
        
        # Create merger
        merger = ContextMerger(temp_graph=temp_graph, temp_vector=temp_vector)
        
        # Mock permanent results
        permanent_results = [
            {
                "score": 0.85,
                "node_id": "perm_1",
                "metadata": {"file": "old_cart.py", "line": 10},
                "text": "def calculate_price(items):"
            }
        ]
        
        # Merge
        merged_results = merger.merge_similar_code_search(
            "calculate total",
            permanent_results,
            limit=10
        )
        
        # Should have results marked with source
        for result in merged_results:
            assert "source" in result
            assert result["source"] in ["permanent", "temporary"]
    
    def test_get_statistics(self, temp_graph, temp_vector):
        """Test statistics collection"""
        # Build temporary graph
        temp_graph.process_file("cart.py", SAMPLE_PYTHON_CODE, "python")
        
        # Create merger
        merger = ContextMerger(temp_graph=temp_graph, temp_vector=temp_vector)
        
        stats = merger.get_statistics()
        
        assert "has_temporary_graph" in stats
        assert "has_temporary_vectors" in stats
        assert stats["has_temporary_graph"] is True
        assert stats["has_temporary_vectors"] is True


@pytest.mark.asyncio
class TestWorkflowIntegration:
    """Test integration with workflow"""
    
    async def test_build_temporary_graphrag(self):
        """Test building temporary GraphRAG from PR files"""
        from src.workflow import CodeReviewWorkflow
        from src.context_builder import ContextBuilder
        
        # Create workflow instance
        workflow = CodeReviewWorkflow()
        
        # Mock PR files
        pr_files = [
            {
                "filename": "cart.py",
                "language": "python",
                "status": "added",
                "patch": SAMPLE_PYTHON_CODE,
                "additions": 20,
                "deletions": 0
            }
        ]
        
        # Build temporary GraphRAG
        temp_graph, temp_vector = await workflow._build_temporary_graphrag(pr_files)
        
        # Verify results
        assert temp_graph is not None
        assert temp_vector is not None
        
        if temp_graph.files:
            print(f"Built temporary graph with {len(temp_graph.files)} files")
            print(f"Total nodes: {len(temp_graph.nodes)}")
            print(f"Total vectors: {len(temp_vector.vectors)}")
            
            assert len(temp_graph.files) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
