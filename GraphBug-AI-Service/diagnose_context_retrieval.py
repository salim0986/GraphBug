#!/usr/bin/env python3
"""
Comprehensive Diagnostic Script for Code Review Context Retrieval Issues

This script tests both pipelines:
1. Ingestion Pipeline - Verifies data is stored in Neo4j and Qdrant
2. Context Retrieval Pipeline - Verifies data can be retrieved during code review

Usage:
    python diagnose_context_retrieval.py <repo_id>

Example:
    python diagnose_context_retrieval.py my-test-repo
"""

import sys
import asyncio
from typing import Dict, Any, List
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, QDRANT_API_KEY
from src.graph_builder import GraphBuilder
from src.vector_builder import VectorBuilder
from src.logger import setup_logger
from src.parser import UniversalParser
from src.analyzer import CodeAnalyzer, PRAnalysisRequest, FileChange
from src.context_builder import ContextBuilder
from sentence_transformers import SentenceTransformer

logger = setup_logger(__name__)

class ContextRetrievalDiagnostics:
    """Comprehensive diagnostics for context retrieval issues"""
    
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.results = {
            "repo_id": repo_id,
            "tests": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Initialize services
        logger.info("Initializing services...")
        self.graph_db = GraphBuilder(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = VectorBuilder(QDRANT_URL, self.embed_model, api_key=QDRANT_API_KEY)
        self.parser = UniversalParser()
        self.code_analyzer = CodeAnalyzer(self.graph_db, self.vector_db, self.parser)
        self.context_builder = ContextBuilder(self.code_analyzer, self.graph_db, self.vector_db)
        
    def log_test(self, test_name: str, passed: bool, details: Dict[str, Any]):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        self.results["tests"][test_name] = {
            "passed": passed,
            "details": details
        }
        
    async def test_neo4j_connection(self) -> bool:
        """Test 1: Neo4j Connection"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Neo4j Connection")
        logger.info("="*80)
        
        try:
            with self.graph_db.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_val = result.single()["test"]
                
            self.log_test("neo4j_connection", True, {
                "uri": NEO4J_URI,
                "connected": True
            })
            return True
        except Exception as e:
            self.log_test("neo4j_connection", False, {
                "uri": NEO4J_URI,
                "error": str(e)
            })
            self.results["errors"].append(f"Neo4j connection failed: {e}")
            return False
    
    async def test_qdrant_connection(self) -> bool:
        """Test 2: Qdrant Connection"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Qdrant Connection")
        logger.info("="*80)
        
        try:
            collections = self.vector_db.client.get_collections()
            has_collection = any(c.name == "repo_code" for c in collections.collections)
            
            self.log_test("qdrant_connection", True, {
                "url": QDRANT_URL,
                "collections": [c.name for c in collections.collections],
                "has_repo_code_collection": has_collection
            })
            
            if not has_collection:
                self.results["warnings"].append("Collection 'repo_code' does not exist. Run ingestion first.")
            
            return True
        except Exception as e:
            self.log_test("qdrant_connection", False, {
                "url": QDRANT_URL,
                "error": str(e)
            })
            self.results["errors"].append(f"Qdrant connection failed: {e}")
            return False
    
    async def test_neo4j_data_exists(self) -> bool:
        """Test 3: Check if repo data exists in Neo4j"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Neo4j Data Verification")
        logger.info("="*80)
        
        try:
            with self.graph_db.driver.session() as session:
                # Count total nodes for this repo
                result = session.run(
                    "MATCH (n) WHERE n.repo_id = $repo_id RETURN count(n) as count",
                    repo_id=self.repo_id
                )
                total_nodes = result.single()["count"]
                
                # Count by type
                result = session.run(
                    """
                    MATCH (n) 
                    WHERE n.repo_id = $repo_id 
                    RETURN labels(n)[0] as type, count(n) as count
                    ORDER BY count DESC
                    """,
                    repo_id=self.repo_id
                )
                nodes_by_type = {record["type"]: record["count"] for record in result}
                
                # Count relationships
                result = session.run(
                    """
                    MATCH (a)-[r]->(b) 
                    WHERE a.repo_id = $repo_id 
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                    """,
                    repo_id=self.repo_id
                )
                relationships = {record["rel_type"]: record["count"] for record in result}
                
                # Get sample files
                result = session.run(
                    """
                    MATCH (n) 
                    WHERE n.repo_id = $repo_id AND n.file IS NOT NULL
                    RETURN DISTINCT n.file as file
                    LIMIT 10
                    """,
                    repo_id=self.repo_id
                )
                sample_files = [record["file"] for record in result]
                
            has_data = total_nodes > 0
            
            self.log_test("neo4j_data_exists", has_data, {
                "total_nodes": total_nodes,
                "nodes_by_type": nodes_by_type,
                "relationships": relationships,
                "sample_files": sample_files
            })
            
            if not has_data:
                self.results["errors"].append(
                    f"No data found in Neo4j for repo_id='{self.repo_id}'. "
                    "Run ingestion via POST /ingest endpoint."
                )
                self.results["recommendations"].append(
                    "Run: curl -X POST http://localhost:8000/ingest "
                    f"-H 'Content-Type: application/json' "
                    f"-d '{{\"repo_url\": \"YOUR_REPO_URL\", \"repo_id\": \"{self.repo_id}\", \"installation_id\": \"YOUR_INSTALLATION_ID\"}}'"
                )
            else:
                logger.info(f"   📊 Total nodes: {total_nodes}")
                logger.info(f"   📂 Files indexed: {len(sample_files)}")
                logger.info(f"   🔗 Relationships: {sum(relationships.values())}")
            
            return has_data
        except Exception as e:
            self.log_test("neo4j_data_exists", False, {"error": str(e)})
            self.results["errors"].append(f"Failed to query Neo4j: {e}")
            return False
    
    async def test_qdrant_data_exists(self) -> bool:
        """Test 4: Check if repo data exists in Qdrant"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Qdrant Vector Data Verification")
        logger.info("="*80)
        
        try:
            # Check collection exists
            if not self.vector_db.client.collection_exists("repo_code"):
                self.log_test("qdrant_data_exists", False, {
                    "error": "Collection 'repo_code' does not exist"
                })
                self.results["errors"].append("Qdrant collection 'repo_code' not found.")
                return False
            
            # Get collection info
            collection_info = self.vector_db.client.get_collection("repo_code")
            total_vectors = collection_info.points_count
            
            # Try to search for vectors with this repo_id
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Create a dummy query vector
            dummy_query = self.embed_model.encode("test query").tolist()
            
            result = self.vector_db.client.query_points(
                collection_name="repo_code",
                query=dummy_query,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=self.repo_id)
                        )
                    ]
                ),
                limit=10
            )
            
            repo_vectors = len(result.points)
            
            # Get sample payloads
            sample_payloads = []
            for point in result.points[:3]:
                sample_payloads.append({
                    "name": point.payload.get("name"),
                    "file": point.payload.get("file"),
                    "type": point.payload.get("type")
                })
            
            has_data = repo_vectors > 0
            
            self.log_test("qdrant_data_exists", has_data, {
                "total_vectors_in_collection": total_vectors,
                "vectors_for_repo": repo_vectors,
                "sample_vectors": sample_payloads
            })
            
            if not has_data:
                self.results["errors"].append(
                    f"No vectors found in Qdrant for repo_id='{self.repo_id}'. "
                    "Ingestion may have failed or completed partially."
                )
            else:
                logger.info(f"   📊 Total vectors in collection: {total_vectors}")
                logger.info(f"   🎯 Vectors for this repo: {repo_vectors}")
            
            return has_data
        except Exception as e:
            self.log_test("qdrant_data_exists", False, {"error": str(e)})
            self.results["errors"].append(f"Failed to query Qdrant: {e}")
            return False
    
    async def test_graph_queries(self) -> bool:
        """Test 5: Test graph query methods used during context retrieval"""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Graph Query Methods")
        logger.info("="*80)
        
        try:
            # Get a sample file from Neo4j
            with self.graph_db.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n) 
                    WHERE n.repo_id = $repo_id AND n.file IS NOT NULL
                    RETURN DISTINCT n.file as file
                    LIMIT 1
                    """,
                    repo_id=self.repo_id
                )
                record = result.single()
                if not record:
                    self.log_test("graph_queries", False, {
                        "error": "No files found in Neo4j"
                    })
                    return False
                
                test_file = record["file"]
            
            logger.info(f"   Testing with file: {test_file}")
            
            # Test find_related_by_file
            related_entities = self.graph_db.find_related_by_file(
                self.repo_id, test_file, limit=10
            )
            
            # Test find_file_dependencies
            dependencies = self.graph_db.find_file_dependencies(
                self.repo_id, test_file
            )
            
            # Test find_impacted_callers
            impacted = self.graph_db.find_impacted_callers(
                self.repo_id, test_file, limit=5
            )
            
            success = True
            details = {
                "test_file": test_file,
                "related_entities_count": len(related_entities),
                "dependencies_count": len(dependencies),
                "impacted_callers_count": len(impacted),
                "sample_entities": [
                    {
                        "name": e.get("name"),
                        "type": e.get("type"),
                        "line": e.get("line")
                    }
                    for e in related_entities[:3]
                ]
            }
            
            self.log_test("graph_queries", success, details)
            
            logger.info(f"   📊 Related entities: {len(related_entities)}")
            logger.info(f"   🔗 Dependencies: {len(dependencies)}")
            logger.info(f"   📢 Impacted callers: {len(impacted)}")
            
            if len(related_entities) == 0:
                self.results["warnings"].append(
                    f"No entities found for file {test_file}. "
                    "Graph relationships may be missing."
                )
            
            return success
        except Exception as e:
            self.log_test("graph_queries", False, {"error": str(e)})
            self.results["errors"].append(f"Graph query failed: {e}")
            return False
    
    async def test_vector_search(self) -> bool:
        """Test 6: Test vector search used during context retrieval"""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Vector Search Methods")
        logger.info("="*80)
        
        try:
            # Test search with a simple query
            test_query = "function implementation authentication"
            
            results = self.vector_db.search_similar(
                self.repo_id, test_query, limit=5
            )
            
            sample_results = []
            for point in results[:3]:
                sample_results.append({
                    "name": point.payload.get("name"),
                    "file": point.payload.get("file"),
                    "score": point.score
                })
            
            has_results = len(results) > 0
            
            self.log_test("vector_search", has_results, {
                "query": test_query,
                "results_count": len(results),
                "sample_results": sample_results
            })
            
            logger.info(f"   🔍 Query: '{test_query}'")
            logger.info(f"   📊 Results: {len(results)}")
            
            if not has_results:
                self.results["warnings"].append(
                    "Vector search returned no results. "
                    "Embeddings may not be indexed properly."
                )
            
            return has_results
        except Exception as e:
            self.log_test("vector_search", False, {"error": str(e)})
            self.results["errors"].append(f"Vector search failed: {e}")
            return False
    
    async def test_context_builder(self) -> bool:
        """Test 7: Test the actual context builder used in code reviews"""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: Context Builder (Full Pipeline)")
        logger.info("="*80)
        
        try:
            # Get a sample file from Neo4j to create a mock PR
            with self.graph_db.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n) 
                    WHERE n.repo_id = $repo_id AND n.file IS NOT NULL
                    RETURN DISTINCT n.file as file
                    LIMIT 1
                    """,
                    repo_id=self.repo_id
                )
                record = result.single()
                if not record:
                    self.log_test("context_builder", False, {
                        "error": "No files found to test with"
                    })
                    return False
                
                test_file = record["file"]
            
            logger.info(f"   Creating mock PR with file: {test_file}")
            
            # Create a mock file change
            mock_file_change = FileChange(
                filename=test_file,
                status="modified",
                additions=10,
                deletions=5,
                patch="@@ -1,5 +1,10 @@\n+# Mock change\n",
                language="python"
            )
            
            # Build PR context
            logger.info("   Building PR context...")
            pr_context = await self.context_builder.build_pr_context(
                pr_number=9999,
                repo_id=self.repo_id,
                title="Test PR for Diagnostics",
                description="Testing context retrieval",
                files=[mock_file_change],
                base_ref="main",
                head_ref="test"
            )
            
            # Analyze the context
            context_summary = {
                "total_files": pr_context.total_files,
                "total_additions": pr_context.total_additions,
                "total_deletions": pr_context.total_deletions,
                "languages": pr_context.languages,
                "risk_level": pr_context.risk_level,
                "critical_issues": len(pr_context.critical_issues),
                "high_issues": len(pr_context.high_issues),
                "file_contexts": []
            }
            
            # Check if file contexts have data
            for file_ctx in pr_context.files:
                context_summary["file_contexts"].append({
                    "filename": file_ctx.filename,
                    "entities_count": len(file_ctx.entities),
                    "dependencies_count": len(file_ctx.dependencies),
                    "issues_count": sum(file_ctx.issues_summary.values()),
                    "complexity_score": file_ctx.complexity_score
                })
            
            # Determine if context retrieval worked
            has_entities = any(fc["entities_count"] > 0 for fc in context_summary["file_contexts"])
            has_dependencies = any(fc["dependencies_count"] > 0 for fc in context_summary["file_contexts"])
            
            success = has_entities or has_dependencies
            
            self.log_test("context_builder", success, context_summary)
            
            logger.info(f"   📊 Files analyzed: {context_summary['total_files']}")
            logger.info(f"   🔍 Has entities: {has_entities}")
            logger.info(f"   🔗 Has dependencies: {has_dependencies}")
            logger.info(f"   ⚠️ Risk level: {context_summary['risk_level']}")
            
            if not success:
                self.results["errors"].append(
                    "Context builder ran but returned empty context. "
                    "This indicates data is not being retrieved properly from Neo4j/Qdrant."
                )
                self.results["recommendations"].append(
                    "Check context_builder.py _build_file_context() method. "
                    "Verify graph_db.find_related_by_file() is returning data."
                )
            
            return success
        except Exception as e:
            self.log_test("context_builder", False, {"error": str(e)})
            self.results["errors"].append(f"Context builder failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all diagnostic tests"""
        logger.info("\n" + "="*80)
        logger.info(f"🔍 CONTEXT RETRIEVAL DIAGNOSTICS: {self.repo_id}")
        logger.info("="*80)
        
        # Test 1: Connections
        neo4j_ok = await self.test_neo4j_connection()
        qdrant_ok = await self.test_qdrant_connection()
        
        if not neo4j_ok or not qdrant_ok:
            logger.error("\n❌ Database connections failed. Cannot proceed with other tests.")
            return self.results
        
        # Test 2: Data Existence
        neo4j_has_data = await self.test_neo4j_data_exists()
        qdrant_has_data = await self.test_qdrant_data_exists()
        
        if not neo4j_has_data or not qdrant_has_data:
            logger.error("\n❌ Ingestion pipeline has issues. Data not found in databases.")
            logger.error("   Run ingestion first before testing context retrieval.")
            return self.results
        
        # Test 3: Query Methods
        await self.test_graph_queries()
        await self.test_vector_search()
        
        # Test 4: Full Pipeline
        await self.test_context_builder()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate diagnostic summary"""
        logger.info("\n" + "="*80)
        logger.info("📋 DIAGNOSTIC SUMMARY")
        logger.info("="*80)
        
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for t in self.results["tests"].values() if t["passed"])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        
        if self.results["errors"]:
            logger.error("\n❌ ERRORS:")
            for i, error in enumerate(self.results["errors"], 1):
                logger.error(f"   {i}. {error}")
        
        if self.results["warnings"]:
            logger.warning("\n⚠️ WARNINGS:")
            for i, warning in enumerate(self.results["warnings"], 1):
                logger.warning(f"   {i}. {warning}")
        
        if self.results["recommendations"]:
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                logger.info(f"   {i}. {rec}")
        
        # Final verdict
        logger.info("\n" + "="*80)
        if passed_tests == total_tests:
            logger.info("✅ ALL TESTS PASSED - Context retrieval is working correctly!")
        elif passed_tests >= total_tests * 0.7:
            logger.warning("⚠️ PARTIAL SUCCESS - Some components working, others need attention")
        else:
            logger.error("❌ MULTIPLE FAILURES - Context retrieval has serious issues")
        logger.info("="*80)


async def main():
    """Main diagnostic function"""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_context_retrieval.py <repo_id>")
        print("\nExample:")
        print("  python diagnose_context_retrieval.py my-test-repo")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    
    diagnostics = ContextRetrievalDiagnostics(repo_id)
    results = await diagnostics.run_all_tests()
    
    # Print summary as JSON for programmatic access
    import json
    print("\n" + "="*80)
    print("JSON RESULTS (for parsing):")
    print("="*80)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
