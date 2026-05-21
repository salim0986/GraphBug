from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance
from qdrant_client.http import models
import uuid
from typing import Optional
from .logger import setup_logger

logger = setup_logger(__name__)

class VectorBuilder:
    def __init__(self, url, model, api_key: Optional[str] = None):
        """
        Initialize VectorBuilder with optional API key for Qdrant Cloud.
        
        Args:
            url: Qdrant server URL
            model: Embedding model
            api_key: Optional API key for Qdrant Cloud authentication
        """
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("Initialized Qdrant client with API key authentication")
        else:
            self.client = QdrantClient(url=url)
            logger.info("Initialized Qdrant client without authentication")
        self.model = model
        self.embed_model = model  # Expose for batch operations
        self.collection = "repo_code"

    def ensure_collection(self):
        """
        Creates the collection if it doesn't exist.
        Also creates payload index for repo_id filtering.
        """
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info(f"Created vector collection '{self.collection}'")
                
                # Create payload index for repo_id filtering
                # This is CRITICAL for delete_repo() and tenant isolation
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="repo_id",
                    field_schema="keyword"  # Exact match for string filtering
                )
                logger.info(f"✅ Created payload index for 'repo_id' field")
            else:
                # Collection exists, ensure index exists
                try:
                    collection_info = self.client.get_collection(self.collection)
                    # Check if index exists, if not create it
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name="repo_id",
                        field_schema="keyword"
                    )
                    logger.info(f"✅ Ensured payload index for 'repo_id' field")
                except Exception as index_error:
                    # Index might already exist, that's okay
                    logger.debug(f"Payload index check: {index_error}")
        except Exception as e:
            logger.error(f"Vector DB connection error: {e}")
            raise

    def ingest_function_chunk(self, repo_id, func_name, func_code, file_path, start_line):
        """
        Ingest a chunk of code with repo_id for tenant isolation.
        """
        try:
            vector = self.model.encode(func_code).tolist()

            payload = {
                "repo_id": repo_id,  # Tenant Badge
                "name": func_name,
                "file": file_path,
                "type": "function",
                "start_line": start_line,
                "raw_code": func_code 
            }

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                ]
            )
        except Exception as e:
            logger.warning(f"Vector ingest error for {func_name}: {e}")    
    def upsert_function_vector(self, repo_id, func_name, embedding, file_path, start_line, raw_code):
        """
        Upsert a vector with pre-computed embedding (for batch operations)
        
        Args:
            repo_id: Repository identifier
            func_name: Function name
            embedding: Pre-computed embedding vector (numpy array or list)
            file_path: File path
            start_line: Starting line number
            raw_code: Raw function code
        """
        try:
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                vector = embedding.tolist()
            else:
                vector = list(embedding)
            
            payload = {
                "repo_id": repo_id,
                "name": func_name,
                "file": file_path,
                "type": "function",
                "start_line": start_line,
                "raw_code": raw_code
            }
            
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                ]
            )
        except Exception as e:
            logger.error(f"Vector upsert error: {e}")

    def search_similar(self, repo_id, query_text, limit=5):
        """
        Retrieves code relevant to the query, strictly filtered by repo_id.
        """
        logger.debug(f"[VectorDB] search_similar: repo_id={repo_id}, query='{query_text[:50]}...', limit={limit}")
        
        try:
            query_vector = self.model.encode(query_text).tolist()
            
            result = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=repo_id)
                        )
                    ]
                ),
                limit=limit
            ).points
            
            logger.debug(f"[VectorDB] search_similar returned {len(result)} results")
            if len(result) == 0:
                logger.warning(f"[VectorDB] No vectors found for repo_id={repo_id}, query='{query_text[:50]}'")
            
            return result
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def delete_repo(self, repo_id: str):
        """Deletes all vectors associated with a specific repo_id."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="repo_id",
                                match=models.MatchValue(value=repo_id),
                            ),
                        ],
                    )
                ),
            )
            logger.info(f"🗑️  Deleted vectors for repo {repo_id}")
        except Exception as e:
            logger.error(f"⚠️ Vector delete failed: {e}")
    
    # ========================================================================
    # ADVANCED VECTOR SEARCH FOR CODE REVIEW (Phase 3.3)
    # ========================================================================
    
    def search_similar_code(self, repo_id: str, code_snippet: str, limit: int = 5, min_score: float = 0.7):
        """
        Find similar code snippets using semantic search
        Returns code with similarity scores above threshold
        """
        try:
            query_vector = self.model.encode(code_snippet).tolist()
            
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=repo_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                score_threshold=min_score
            ).points
            
            similar_code = []
            for point in results:
                similar_code.append({
                    "name": point.payload.get("name", "unknown"),
                    "file": point.payload.get("file", ""),
                    "line": point.payload.get("start_line", 0),
                    "code": point.payload.get("raw_code", ""),
                    "similarity": float(point.score),
                    "type": point.payload.get("type", "function")
                })
            
            logger.info(f"Found {len(similar_code)} similar code snippets (threshold: {min_score})")
            return similar_code
            
        except Exception as e:
            logger.error(f"Error searching similar code: {e}")
            return []
    
    def find_duplicate_code(self, repo_id: str, code_snippet: str, limit: int = 3, threshold: float = 0.9):
        """
        Find potential duplicate code (high similarity threshold)
        Useful for detecting copy-pasted code
        """
        try:
            duplicates = self.search_similar_code(repo_id, code_snippet, limit=limit, min_score=threshold)
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} potential duplicates (threshold: {threshold})")
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicate code: {e}")
            return []
    
    def semantic_code_search(self, repo_id: str, natural_language_query: str, limit: int = 10):
        """
        Search code using natural language description
        E.g., "functions that handle authentication" or "code that processes payments"
        """
        try:
            query_vector = self.model.encode(natural_language_query).tolist()
            
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="repo_id",
                            match=MatchValue(value=repo_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            ).points
            
            matches = []
            for point in results:
                matches.append({
                    "name": point.payload.get("name", "unknown"),
                    "file": point.payload.get("file", ""),
                    "line": point.payload.get("start_line", 0),
                    "code": point.payload.get("raw_code", ""),
                    "relevance": float(point.score),
                    "type": point.payload.get("type", "function")
                })
            
            logger.info(f"Semantic search found {len(matches)} matches for: '{natural_language_query}'")
            return matches
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def search_by_file(self, repo_id: str, file_path: str):
        """
        Get all code snippets from a specific file
        Useful for file-level analysis
        """
        try:
            # Use scroll to get all points from a file (no limit)
            results = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="repo_id", match=MatchValue(value=repo_id)),
                        FieldCondition(key="file", match=MatchValue(value=file_path))
                    ]
                ),
                limit=100,
                with_payload=True
            )[0]  # Returns tuple (points, next_page_offset)
            
            snippets = []
            for point in results:
                snippets.append({
                    "name": point.payload.get("name", "unknown"),
                    "file": point.payload.get("file", ""),
                    "line": point.payload.get("start_line", 0),
                    "code": point.payload.get("raw_code", ""),
                    "type": point.payload.get("type", "function")
                })
            
            # Sort by line number
            snippets.sort(key=lambda x: x["line"])
            
            logger.info(f"Found {len(snippets)} code snippets in {file_path}")
            return snippets
            
        except Exception as e:
            logger.error(f"Error searching by file: {e}")
            return []
    
    def search_by_type(self, repo_id: str, code_type: str, limit: int = 50):
        """
        Search for specific code types (function, class, method, etc.)
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="repo_id", match=MatchValue(value=repo_id)),
                        FieldCondition(key="type", match=MatchValue(value=code_type))
                    ]
                ),
                limit=limit,
                with_payload=True
            )[0]
            
            items = []
            for point in results:
                items.append({
                    "name": point.payload.get("name", "unknown"),
                    "file": point.payload.get("file", ""),
                    "line": point.payload.get("start_line", 0),
                    "code": point.payload.get("raw_code", ""),
                    "type": point.payload.get("type", "function")
                })
            
            logger.info(f"Found {len(items)} items of type '{code_type}'")
            return items
            
        except Exception as e:
            logger.error(f"Error searching by type: {e}")
            return []
    
    def get_repository_stats(self, repo_id: str):
        """
        Get statistics about indexed code in repository
        """
        try:
            # Count total snippets
            results = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="repo_id", match=MatchValue(value=repo_id))
                    ]
                ),
                limit=10000,  # High limit to get all
                with_payload=True
            )[0]
            
            total_snippets = len(results)
            
            # Aggregate by type and file
            by_type = {}
            by_file = {}
            
            for point in results:
                code_type = point.payload.get("type", "unknown")
                file_path = point.payload.get("file", "unknown")
                
                by_type[code_type] = by_type.get(code_type, 0) + 1
                by_file[file_path] = by_file.get(file_path, 0) + 1
            
            stats = {
                "total_snippets": total_snippets,
                "by_type": by_type,
                "total_files": len(by_file),
                "top_files": sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            logger.info(f"Repository stats: {total_snippets} snippets across {len(by_file)} files")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting repository stats: {e}")
            return {
                "total_snippets": 0,
                "by_type": {},
                "total_files": 0,
                "top_files": []
            }
    
    def delete_by_file(self, repo_id: str, file_path: str):
        """
        Delete all vectors for a specific file (for incremental updates)
        
        Args:
            repo_id: Repository identifier
            file_path: File path to delete vectors for
        """
        try:
            # Use delete with filter
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(key="repo_id", match=MatchValue(value=repo_id)),
                            FieldCondition(key="file", match=MatchValue(value=file_path))
                        ]
                    )
                )
            )
            logger.debug(f"Deleted vectors for {file_path}")
            
        except Exception as e:
            logger.error(f"Error deleting vectors for {file_path}: {e}")