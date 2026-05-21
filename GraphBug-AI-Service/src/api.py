import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from .parser import UniversalParser, EXTENSION_MAP
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder
from .analyzer import (
    CodeAnalyzer,
    PRAnalysisRequest,
    FileAnalysisRequest,
    DiffAnalysisRequest,
    PRAnalysisResult,
    FileAnalysisResult
)
from .context_builder import ContextBuilder, PRContext
from .workflow import CodeReviewWorkflow, WorkflowConfig, create_review_workflow
from .logger import setup_logger, LogContext
from .incremental_ingest import ingest_repo_incremental
from sentence_transformers import SentenceTransformer
from .github_client import GitHubClient, GitHubConfig
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .security_utils import (
    safe_join,
    sanitize_repo_id,
    sanitize_owner_repo,
    sanitize_file_path,
    sanitize_error_message,
    validate_request_size
)
from .rate_limiter import RateLimiter, RateLimitMiddleware
from .cleanup import DataCleanup, IngestionCheckpoint
import git
import shutil
import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid
import httpx
import jwt
from datetime import datetime, timezone

# Initialize rate limiters
limiter = Limiter(key_func=get_remote_address)
advanced_limiter = RateLimiter()

app = FastAPI(title="Graph Bug AI Service", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = setup_logger(__name__)

# Configure CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

# Add advanced rate limiting middleware
app.add_middleware(RateLimitMiddleware, limiter=advanced_limiter)

@app.on_event("startup")
async def startup_event():
    """Log service status on startup"""
    logger.info("=" * 80)
    logger.info("🚀 Graph Bug AI Service Starting...")
    logger.info(f"   Version: 2.0.0")
    logger.info(f"   GitHub Client: {'✅ Ready' if github_client else '❌ Not configured'}")
    logger.info(f"   Rate Limiting: ✅ Enabled")
    logger.info(f"   Data Cleanup: ✅ Enabled")
    logger.info("=" * 80)

# --- CONFIGURATION ---

IGNORE_DIRS = {
    ".git", ".github", ".vscode", ".idea",
    "node_modules", "venv", "env",
    "dist", "build", "out", "target", "bin", "obj",
    "__pycache__", "coverage", "tmp", "temp", "migrations"
}

# Initialize Singletons
parser = UniversalParser()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load config
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, QDRANT_API_KEY

# Initialize databases with config
graph_db = GraphBuilder(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
vector_db = VectorBuilder(QDRANT_URL, embed_model, api_key=QDRANT_API_KEY)

# Initialize cleanup service
cleanup_service = DataCleanup(graph_db, vector_db)

# Initialize Code Analyzer and Context Builder
code_analyzer = CodeAnalyzer(graph_db, vector_db, parser)
context_builder = ContextBuilder(code_analyzer, graph_db, vector_db)

# Initialize GitHub Client (Phase 5.3)
try:
    logger.info("=" * 80)
    logger.info("🔑 Initializing GitHub Client...")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    github_config = GitHubConfig.from_env()
    github_client = GitHubClient(github_config)
    
    # Test the token generation to ensure credentials are valid
    logger.info("Testing GitHub App credentials...")
    sys.stdout.flush()
    
    try:
        # Try to generate a JWT to verify private key is valid
        test_jwt = jwt.encode(
            {
                "iat": int(time.time()),
                "exp": int(time.time()) + 60,
                "iss": github_config.app_id
            },
            github_config.private_key,
            algorithm="RS256"
        )
        logger.info(f"✅ JWT generation successful ({len(test_jwt)} chars)")
        sys.stdout.flush()
    except Exception as jwt_error:
        logger.error(f"❌ JWT generation failed: {jwt_error}")
        logger.error("   This indicates the private key is invalid or malformed")
        sys.stdout.flush()
        raise
    
    logger.info("=" * 80)
    logger.info("✅ GitHub Client Initialized Successfully!")
    logger.info(f"   App ID: {github_config.app_id}")
    logger.info(f"   Private key loaded: {len(github_config.private_key)} bytes")
    logger.info("   PR comments will be posted automatically")
    logger.info("   Repository cloning will use GitHub App authentication")
    logger.info("=" * 80)
    sys.stdout.flush()
    
except Exception as e:
    logger.error("=" * 80)
    logger.error(f"❌ GITHUB CLIENT INITIALIZATION FAILED")
    logger.error(f"   Error: {e}")
    logger.error(f"   Error type: {type(e).__name__}")
    logger.error("=" * 80)
    logger.error("Consequences:")
    logger.error("  - PR comments will NOT be posted")
    logger.error("  - Private repositories CANNOT be cloned")
    logger.error("  - Only public repositories will work")
    logger.error("=" * 80)
    logger.error("Required environment variables:")
    logger.error("  - GITHUB_APP_ID: Your GitHub App ID")
    logger.error("  - GITHUB_PRIVATE_KEY: Your GitHub App private key (PEM format)")
    logger.error("  - Or GITHUB_PRIVATE_KEY_PATH: Path to private key file")
    logger.error("=" * 80)
    logger.error("Full traceback:", exc_info=True)
    sys.stdout.flush()
    github_client = None

# Initialize LangGraph Workflow (Phase 4) with dependencies
review_workflow = create_review_workflow(
    context_builder=context_builder
)

# --- MODELS ---

class RepoRequest(BaseModel):
    repo_url: str = Field(..., max_length=500, pattern=r'^https://github\.com/[\w-]+/[\w.-]+(.git)?$')
    repo_id: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_-]+$')
    installation_id: str = Field(..., pattern=r'^\d+$', max_length=20)
    incremental: Optional[bool] = False  # Enable incremental ingestion
    last_commit: Optional[str] = Field(None, min_length=7, max_length=40, pattern=r'^[a-fA-F0-9]+$')  # Previous commit SHA for incremental

class QueryRequest(BaseModel):
    repo_id: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_-]+$')
    query: str = Field(..., min_length=1, max_length=1000)

def process_repo_task(req: RepoRequest):
    """The Heavy Worker Function: Clones -> Parses -> Indexes"""
    logger.info("=" * 80)
    logger.info(f"🔄 STARTING FULL INGESTION")
    logger.info(f"   Repo URL: {req.repo_url}")
    logger.info(f"   Repo ID: {req.repo_id}")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    # Sanitize repo_id to prevent path traversal
    try:
        safe_repo_id = sanitize_repo_id(req.repo_id)
    except ValueError as e:
        logger.error(f"Invalid repo_id: {e}")
        return
    
    # Use safe_join to prevent directory traversal
    try:
        local_path = safe_join("./temp_repos", safe_repo_id)
    except ValueError as e:
        logger.error(f"Path traversal attempt detected: {e}")
        return
    
    # 1. Clean & Clone
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    
    try:
        logger.info(f"Cloning repository from {req.repo_url}")
        logger.info(f"GitHub client available: {github_client is not None}")
        sys.stdout.flush()
        
        # FIX: Use GitHub App installation token for private repo access
        if github_client:
            try:
                logger.info(f"Attempting to get installation token for installation {req.installation_id}")
                sys.stdout.flush()
                
                # Get installation access token
                token = github_client.get_installation_token(int(req.installation_id))
                
                if not token or len(token) < 10:
                    raise ValueError(f"Invalid token received: {len(token) if token else 0} chars")
                
                logger.info(f"✅ Successfully obtained installation token ({len(token)} chars)")
                sys.stdout.flush()
                
                # Build authenticated clone URL
                # Format: https://x-access-token:TOKEN@github.com/owner/repo.git
                auth_url = req.repo_url.replace(
                    "https://github.com/",
                    f"https://x-access-token:{token}@github.com/"
                )
                
                logger.info(f"Cloning with GitHub App authentication (installation {req.installation_id})")
                sys.stdout.flush()
                
                # Clone with timeout to prevent hanging
                git.Repo.clone_from(auth_url, local_path, depth=1)  # Shallow clone for speed
                
                logger.info(f"✅ Successfully cloned repository with authentication")
                sys.stdout.flush()
                
            except Exception as auth_error:
                logger.error("=" * 80)
                logger.error(f"❌ GITHUB APP AUTHENTICATION FAILED")
                logger.error(f"   Installation ID: {req.installation_id}")
                logger.error(f"   Error: {auth_error}")
                logger.error(f"   Error type: {type(auth_error).__name__}")
                logger.error("=" * 80)
                logger.error("Full traceback:", exc_info=True)
                sys.stdout.flush()
                
                # DO NOT fallback to public clone for private repos
                # This will fail with "could not read Username" error
                raise Exception(
                    f"Failed to clone repository with GitHub App authentication. "
                    f"This is likely a private repository. Error: {auth_error}"
                )
        else:
            # No GitHub client available - only works for public repos
            logger.error("=" * 80)
            logger.error(f"❌ GITHUB CLIENT NOT AVAILABLE")
            logger.error(f"   Cannot clone private repositories without GitHub App credentials")
            logger.error(f"   Set GITHUB_APP_ID and GITHUB_PRIVATE_KEY in environment")
            logger.error("=" * 80)
            sys.stdout.flush()
            
            raise Exception(
                "GitHub client not initialized. Cannot clone repository. "
                "Please set GITHUB_APP_ID and GITHUB_PRIVATE_KEY environment variables."
            )
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ GIT CLONE FAILED")
        logger.error(f"   Repo URL: {req.repo_url}")
        logger.error(f"   Installation ID: {req.installation_id}")
        logger.error(f"   Local path: {local_path}")
        logger.error(f"   Error: {e}")
        logger.error("=" * 80)
        logger.error("Common causes:")
        logger.error("  1. Private repo without valid GitHub App token")
        logger.error("  2. GitHub App not installed on repository")
        logger.error("  3. Invalid installation ID")
        logger.error("  4. Repository doesn't exist or was deleted")
        logger.error("  5. Network connectivity issues")
        logger.error("=" * 80)
        sys.stdout.flush()
        return

    start_time = time.time()

    # 2. Prepare Vector DB
    vector_db.ensure_collection()
    
    logger.info(f"🔍 Collecting files to process...")
    
    # Collect all valid files first
    files_to_process = []
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            file_path = os.path.join(root, file)
            filename = os.path.basename(file)
            _, ext = os.path.splitext(filename)
            is_valid = filename in EXTENSION_MAP or ext in EXTENSION_MAP
            
            if is_valid:
                files_to_process.append(file_path)
    
    total_files = len(files_to_process)
    logger.info(f"📊 Found {total_files} code files to process")
    
    if total_files == 0:
        logger.warning(f"No valid code files found in {req.repo_id}")
        shutil.rmtree(local_path)
        return
    
    # 3. Parallel Processing - Parse files in batches
    indexed_count = 0
    file_count = 0
    batch_size = 20  # Parallel workers for better throughput
    
    def parse_single_file(file_path):
        """Parse a single file and return results"""
        try:
            captures, code_bytes = parser.parse_file(file_path)
            if captures:
                rel_path = os.path.relpath(file_path, local_path)
                return (file_path, rel_path, captures, code_bytes, None)
            return None
        except Exception as e:
            return (file_path, None, None, None, str(e))
    
    logger.info(f"⚡ Starting parallel processing with {batch_size} workers...")
    
    # Process files in parallel batches
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in range(0, total_files, batch_size):
            batch = files_to_process[i:i + batch_size]
            batch_start = time.time()
            
            # Submit batch with timeout protection
            futures = {}
            for fp in batch:
                future = executor.submit(parse_single_file, fp)
                futures[future] = fp
            
            # Collect results with timeout
            batch_results = []
            try:
                for future in as_completed(futures, timeout=120):  # 120s timeout per batch
                    try:
                        result = future.result(timeout=10)  # 10s to get result
                        if result:
                            batch_results.append(result)
                    except Exception as e:
                        fp = futures[future]
                        logger.warning(f"Error processing {os.path.basename(fp)}: {e}")
            except Exception as batch_error:
                logger.warning(f"Batch timeout or error: {batch_error}")
            
            # Process successful parses
            vectors_batch = []  # Collect vectors for batch insert
            batch_file_count = 0
            
            for result in batch_results:
                file_path, rel_path, captures, code_bytes, error = result
                
                if error:
                    logger.warning(f"Parse error in {os.path.basename(file_path)}: {error}")
                    continue
                
                if not rel_path:
                    continue
                
                batch_file_count += 1
                
                # Update graph
                try:
                    graph_db.process_file_nodes(req.repo_id, rel_path, captures, code_bytes)
                except Exception as e:
                    logger.error(f"Graph error for {rel_path}: {e}")
                    continue
                
                # Collect vectors for batch insert
                for node, capture_name in captures:
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
                    
                    _, ext = os.path.splitext(rel_path)
                    is_logic_node = node_type in ALLOWED_TYPES
                    is_config_file = ext in [".yaml", ".yml", ".toml", ".tf", ".dockerfile"]
                    
                    if not (is_logic_node or (is_config_file and lines_of_code > 0)):
                        continue
                    
                    try:
                        func_code = code_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
                        first_line = func_code.splitlines()[0] if func_code else ""
                        func_name = first_line[:100].strip()
                        
                        vectors_batch.append({
                            "repo_id": req.repo_id,
                            "func_name": func_name,
                            "func_code": func_code,
                            "file_path": rel_path,
                            "start_line": node.start_point[0]
                        })
                            
                    except Exception as e:
                        logger.warning(f"Vector prep error: {e}")
                        continue
            
            file_count += batch_file_count
            
            # Batch insert vectors
            if vectors_batch:
                try:
                    logger.info(f"💾 Inserting {len(vectors_batch)} vectors into Qdrant...")
                    for item in vectors_batch:
                        vector_db.ingest_function_chunk(
                            repo_id=item["repo_id"],
                            func_name=item["func_name"],
                            func_code=item["func_code"],
                            file_path=item["file_path"],
                            start_line=item["start_line"]
                        )
                    indexed_count += len(vectors_batch)
                    logger.info(f"✅ Successfully inserted {len(vectors_batch)} vectors")
                except Exception as e:
                    logger.error(f"Batch vector insert error: {e}")
            
            # Progress update
            batch_time = time.time() - batch_start
            progress_pct = ((i + len(batch)) / total_files) * 100
            files_per_sec = len(batch) / batch_time if batch_time > 0 else 0
            logger.info(
                f"📊 Progress: {progress_pct:.1f}% | "
                f"{file_count}/{total_files} files | "
                f"{indexed_count} functions | "
                f"{files_per_sec:.1f} files/sec"
            )
    
    # 4. Build dependencies
    logger.info(f"🔗 Building dependency graph...")
    sys.stdout.flush()
    try:
        graph_db.build_dependencies(req.repo_id)
    except Exception as e:
        logger.error(f"Error building dependencies: {e}")
        sys.stdout.flush()
    
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"✅ FULL INGESTION COMPLETE")
    logger.info(f"   Time: {elapsed_time:.1f}s")
    logger.info(f"   Files: {file_count}")
    logger.info(f"   Functions indexed: {indexed_count}")
    logger.info(f"   Speed: {file_count/elapsed_time:.1f} files/sec")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    # Cleanup
    try:
        shutil.rmtree(local_path)
    except:
        pass


# --- ENDPOINTS ---

@app.post("/ingest")
async def ingest_repo(request: Request, req: RepoRequest, background_tasks: BackgroundTasks):
    """
    Ingest repository with optional incremental mode
    
    Modes:
    - Full ingestion: incremental=False (default) - processes all files
    - Incremental: incremental=True with last_commit - only processes changed files
    
    Performance:
    - Full: ~10-60s depending on repo size
    - Incremental: ~2-10s for typical updates
    """
    # Force immediate log output
    logger.info("=" * 80)
    logger.info(f"📥 INGESTION REQUEST RECEIVED")
    logger.info(f"   Repo: {req.repo_url}")
    logger.info(f"   Repo ID: {req.repo_id}")
    logger.info(f"   Installation ID: {req.installation_id}")
    logger.info(f"   Incremental: {req.incremental}")
    logger.info(f"   Last Commit: {req.last_commit or 'N/A'}")
    logger.info("=" * 80)
    sys.stdout.flush()  # Force immediate flush
    
    if req.incremental and req.last_commit:
        # Incremental mode - much faster!
        background_tasks.add_task(process_repo_incremental_task, req)
        response = {
            "status": "queued",
            "repo_id": req.repo_id,
            "mode": "incremental",
            "from_commit": req.last_commit
        }
    else:
        # Full ingestion mode
        background_tasks.add_task(process_repo_task, req)
        response = {
            "status": "queued",
            "repo_id": req.repo_id,
            "mode": "full"
        }
    
    logger.info(f"✅ Ingestion queued: {response}")
    sys.stdout.flush()  # Force immediate flush
    return response

async def process_repo_incremental_task(req: RepoRequest):
    """
    High-performance incremental ingestion
    Only processes files that changed since last_commit
    """
    logger.info("=" * 80)
    logger.info(f"🔄 STARTING INCREMENTAL INGESTION")
    logger.info(f"   Repo URL: {req.repo_url}")
    logger.info(f"   Repo ID: {req.repo_id}")
    logger.info(f"   From commit: {req.last_commit}")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    local_path = f"./temp_repos/{req.repo_id}"
    
    try:
        # Clone or pull repository with authentication
        if os.path.exists(local_path):
            logger.info(f"Repository exists locally, pulling latest changes...")
            sys.stdout.flush()
            
            repo = git.Repo(local_path)
            
            # Update remote URL with token if GitHub client available
            if github_client:
                try:
                    logger.info(f"Updating remote URL with fresh authentication token")
                    sys.stdout.flush()
                    
                    token = github_client.get_installation_token(int(req.installation_id))
                    
                    if not token or len(token) < 10:
                        raise ValueError(f"Invalid token received: {len(token) if token else 0} chars")
                    
                    auth_url = req.repo_url.replace(
                        "https://github.com/",
                        f"https://x-access-token:{token}@github.com/"
                    )
                    
                    # Update remote URL
                    repo.remotes.origin.set_url(auth_url)
                    logger.info(f"✅ Remote URL updated with authentication")
                    sys.stdout.flush()
                    
                except Exception as e:
                    logger.error(f"❌ Failed to update remote with auth: {e}")
                    sys.stdout.flush()
                    raise
            
            origin = repo.remotes.origin
            origin.pull()
            logger.info(f"✅ Successfully pulled latest changes")
            sys.stdout.flush()
            
        else:
            logger.info(f"Repository not found locally, cloning from {req.repo_url}")
            sys.stdout.flush()
            
            # Use authenticated URL for private repos
            if github_client:
                try:
                    logger.info(f"Obtaining installation token for cloning")
                    sys.stdout.flush()
                    
                    token = github_client.get_installation_token(int(req.installation_id))
                    
                    if not token or len(token) < 10:
                        raise ValueError(f"Invalid token received: {len(token) if token else 0} chars")
                    
                    logger.info(f"✅ Token obtained ({len(token)} chars)")
                    sys.stdout.flush()
                    
                    auth_url = req.repo_url.replace(
                        "https://github.com/",
                        f"https://x-access-token:{token}@github.com/"
                    )
                    
                    git.Repo.clone_from(auth_url, local_path, depth=1)
                    logger.info(f"✅ Successfully cloned repository with authentication")
                    sys.stdout.flush()
                    
                except Exception as e:
                    logger.error("=" * 80)
                    logger.error(f"❌ CLONE FAILED WITH AUTHENTICATION")
                    logger.error(f"   Installation ID: {req.installation_id}")
                    logger.error(f"   Error: {e}")
                    logger.error("=" * 80)
                    sys.stdout.flush()
                    raise
            else:
                logger.error("=" * 80)
                logger.error(f"❌ GITHUB CLIENT NOT AVAILABLE")
                logger.error(f"   Cannot clone private repositories")
                logger.error("=" * 80)
                sys.stdout.flush()
                raise Exception("GitHub client not initialized")
        
        # Run incremental ingestion
        stats = await ingest_repo_incremental(
            repo_id=req.repo_id,
            repo_url=req.repo_url,
            local_path=local_path,
            parser=parser,
            graph_db=graph_db,
            vector_db=vector_db,
            ignore_dirs=IGNORE_DIRS,
            last_commit=req.last_commit
        )
        
        logger.info("=" * 80)
        logger.info(f"✅ INCREMENTAL INGESTION COMPLETE")
        logger.info(f"   Files processed: {stats['files_processed']}")
        logger.info(f"   Files deleted: {stats.get('files_deleted', 0)}")
        logger.info(f"   Nodes added: {stats['nodes_added']}")
        logger.info(f"   Nodes deleted: {stats.get('nodes_deleted', 0)}")
        logger.info(f"   Vectors updated: {stats['vectors_updated']}")
        logger.info("=" * 80)
        sys.stdout.flush()
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ INCREMENTAL INGESTION FAILED")
        logger.error(f"   Repo: {req.repo_url}")
        logger.error(f"   Error: {e}")
        logger.error("=" * 80)
        logger.error(f"Full traceback:", exc_info=True)
        sys.stdout.flush()
    
    # Cleanup (optional - keep for next incremental update)
    # try:
    #     shutil.rmtree(local_path)
    # except:
    #     pass

# --- ENDPOINTS ---

@app.post("/ingest_legacy")
async def ingest_repo_legacy(req: RepoRequest, background_tasks: BackgroundTasks):
    """Legacy full ingestion endpoint (kept for backwards compatibility)"""
    background_tasks.add_task(process_repo_task, req)
    return {"status": "queued", "repo_id": req.repo_id}

@app.post("/query")
async def search_repo(http_request: Request, req: QueryRequest):
    """
    GraphRAG Search:
    1. Vector Search finds the relevant 'entry points'.
    2. Graph Search expands those points to find dependencies.
    """
    # 1. Vector Search (Find top 3 matches)
    results = vector_db.search_similar(req.repo_id, req.query)
    
    data = []
    for hit in results:
        payload = hit.payload
        
        # 2. Graph Expansion (Fetch context)
        # "What does this function call? What does it rely on?"
        dependencies = graph_db.get_dependencies(
            repo_id=req.repo_id, 
            file_path=payload.get("file"), 
            start_line=payload.get("start_line")
        )
        
        data.append({
            "score": hit.score,
            "type": "Primary Match",
            "name": payload.get("name"),
            "file": payload.get("file"),
            "start_line": payload.get("start_line"),
            "code": payload.get("raw_code"),
            "related_dependencies": dependencies  # <--- The Graph Value
        })
    
    return {"results": data}


@app.delete("/repos/{repo_id}")
async def delete_repo_data(repo_id: str):
    """
    Cleanup Endpoint:
    Called when a user uninstalls the app or deletes a repository.
    Wipes data from both Vector DB and Graph DB.
    """
    logger.info(f"Received delete request for repo {repo_id}")
    
    # 1. Delete from Vector DB (Search Index)
    vector_db.delete_repo(repo_id)
    
    # 2. Delete from Graph DB (Structure)
    graph_db.delete_repo(repo_id)
    
    return {"status": "success", "message": f"Data for {repo_id} wiped."}


# ============================================================================
# CODE ANALYSIS ENDPOINTS (Phase 3)
# ============================================================================

@app.post("/analyze/pr", response_model=PRAnalysisResult)
async def analyze_pr(request: PRAnalysisRequest):
    """
    Analyze an entire Pull Request
    
    Returns comprehensive analysis including:
    - Code issues by severity and category
    - Similar code patterns found
    - Related code via graph traversal
    - Metrics for each file
    """
    try:
        logger.info(f"Received PR analysis request for PR #{request.pr_number} in repo {request.repo_id}")
        result = await code_analyzer.analyze_pr(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing PR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/file", response_model=FileAnalysisResult)
async def analyze_file(request: FileAnalysisRequest):
    """
    Analyze a single file
    
    Returns:
    - Code issues and smells
    - Similar code patterns
    - Related code dependencies
    - File metrics
    """
    try:
        logger.info(f"Received file analysis request for {request.filename} in repo {request.repo_id}")
        result = await code_analyzer.analyze_file(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/diff", response_model=FileAnalysisResult)
async def analyze_diff(request: DiffAnalysisRequest):
    """
    Analyze a diff/patch
    
    Focuses on changed lines and their context.
    Returns issues found in new code.
    """
    try:
        logger.info(f"Received diff analysis request for {request.filename} in repo {request.repo_id}")
        result = await code_analyzer.analyze_diff(request)
        return result
    except Exception as e:
        logger.error(f"Error analyzing diff: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "neo4j": "connected" if graph_db else "not configured",
            "qdrant": "connected" if vector_db else "not configured"
        }
    }


# ========================================================================
# WEBHOOK INTEGRATION ENDPOINT (GitHub PR Review)
# ========================================================================

from typing import Any

class PRReviewWebhookRequest(BaseModel):
    """Request from frontend webhook after PR processing"""
    owner: str
    repo: str
    pr_number: int
    installation_id: str
    pull_request_id: Optional[str] = None  # Optional - will be created if not provided
    repo_db_id: Optional[str] = None  # Repository UUID from frontend database
    context: Optional[Any] = None  # Optional - will be fetched if not provided
    gemini_api_key: Optional[str] = None  # User's Gemini API key (BYO system)
    
    class Config:
        extra = "allow"  # Allow extra fields


@app.post("/review")
async def process_pr_review_webhook(http_request: Request, request: PRReviewWebhookRequest, background_tasks: BackgroundTasks):
    """
    Main webhook endpoint for PR review processing
    
    Called by frontend webhook when PR is opened/updated.
    Orchestrates the complete AI review workflow:
    1. Fetches PR data from GitHub (if not provided)
    2. Triggers LangGraph workflow
    3. Posts results back to GitHub
    
    Runs as background task to avoid timeout.
    """
    try:
        logger.info(f"🔔 Received review webhook for PR #{request.pr_number} in {request.owner}/{request.repo}")
        logger.info(f"   Installation ID: {request.installation_id}")
        logger.info(f"   Pull Request ID: {request.pull_request_id or '(will generate)'}")
        logger.info(f"   Context provided: {request.context is not None}")
        
        # Generate pull_request_id if not provided
        if not request.pull_request_id:
            request.pull_request_id = str(uuid.uuid4())
            logger.info(f"   Generated Pull Request ID: {request.pull_request_id}")
        
        # Fetch PR data from GitHub if context not provided
        if not request.context:
            logger.info(f"📥 Context not provided, fetching PR data from GitHub...")
            try:
                repo_full_name = f"{request.owner}/{request.repo}"
                installation_id_int = int(request.installation_id)
                
                # Use global github_client
                if not github_client:
                    logger.error("❌ GitHub client not initialized")
                    raise HTTPException(status_code=500, detail="GitHub client not available")
                
                pr_data = await github_client.get_pull_request(
                    repo_full_name=repo_full_name,
                    pr_number=request.pr_number,
                    installation_id=installation_id_int
                )
                
                # Build context from PR data
                context = {
                    "title": pr_data["title"],
                    "description": pr_data["body"],
                    "files": pr_data["files"],
                    "base_ref": pr_data["base"]["ref"],
                    "head_ref": pr_data["head"]["ref"],
                    "additions": pr_data["additions"],
                    "deletions": pr_data["deletions"],
                    "changed_files": pr_data["changed_files"],
                }
                
                logger.info(f"✅ Fetched PR data: {len(context['files'])} files, +{context['additions']}/-{context['deletions']}")
            except Exception as e:
                logger.error(f"❌ Failed to fetch PR data from GitHub: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch PR data: {str(e)}")
        else:
            context = request.context
        
        # Extract context data
        files = context.get("files", [])
        
        if not files:
            logger.warning(f"No files found in PR #{request.pr_number} context")
            return {
                "status": "skipped",
                "message": "No files to review",
                "pr_number": request.pr_number,
                "review_id": None
            }
        
        logger.info(f"📝 Processing {len(files)} files from PR #{request.pr_number}")
        
        # Use repo_db_id if provided (from frontend database), otherwise fall back to GitHub format
        repo_id = request.repo_db_id if request.repo_db_id else f"{request.owner}/{request.repo}"
        
        if not request.repo_db_id:
            logger.warning(f"⚠️ No repo_db_id provided, using GitHub format '{repo_id}' - this may cause context lookup issues")
        
        # Extract branch refs from context
        base_ref = context.get("base_ref", "main")
        head_ref = context.get("head_ref", "unknown")
        
        # Queue the review workflow as a background task
        background_tasks.add_task(
            execute_review_task,
            pr_number=request.pr_number,
            repo_id=repo_id,
            pr_title=context.get("title", ""),
            pr_description=context.get("description"),
            files=files,
            base_ref=base_ref,
            head_ref=head_ref,
            installation_id=request.installation_id,
            pull_request_db_id=request.pull_request_id,
            owner=request.owner,
            repo=request.repo,
            gemini_api_key=request.gemini_api_key
        )
        
        # Return immediately with queued status
        review_id = f"review-{request.pull_request_id}"
        logger.info(f"✅ Review workflow queued: {review_id}")
        
        return {
            "status": "queued",
            "message": f"Review workflow started for PR #{request.pr_number}",
            "pr_number": request.pr_number,
            "review_id": review_id,
            "files_count": len(files)
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing review webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# DATABASE STORAGE HELPER
# ========================================================================

async def store_review_in_database(
    pull_request_db_id: str,
    final_state: dict,
    github_comment_id: Optional[int],
    github_comment_url: Optional[str]
):
    """
    Store review results in frontend database for analytics
    """
    try:
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        
        # Extract review summary data
        review_summary = final_state.get("review_summary", {})
        analysis = final_state.get("analysis", {})
        route_decision = final_state.get("route_decision", {})
        
        # Calculate timestamps (must be UTC timezone with Z suffix for Zod validation)
        start_time = final_state.get("start_time")
        end_time = final_state.get("end_time")
        
        if start_time:
            started_at = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
        else:
            started_at = None
            
        if end_time:
            completed_at = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
        else:
            completed_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Calculate execution time
        execution_time_ms = int((end_time - start_time) * 1000) if (start_time and end_time) else 0
        
        # Calculate total cost (dummy values for now, should come from Gemini client)
        total_cost = 0.0
        
        # Determine primary model
        model = route_decision.get("model", "gemini-2.5-flash")
        primary_model = "flash" if "flash" in model.lower() else ("pro" if "pro" in model.lower() else "flash")
        
        # Build payload matching frontend's /api/reviews endpoint
        payload = {
            "pull_request_id": pull_request_db_id,
            "status": final_state.get("status", "completed"),
            "started_at": started_at,
            "completed_at": completed_at,
            "primary_model": primary_model,
            "total_cost": total_cost,
            "execution_time_ms": execution_time_ms,
            "summary": {
                "overallScore": 85,  # Could be calculated from issues/suggestions ratio
                "filesChanged": final_state.get("total_files", 0),
                "issuesFound": review_summary.get("total_issues", 0),
                "critical": review_summary.get("critical_count", 0),
                "high": review_summary.get("high_count", 0),
                "medium": review_summary.get("medium_count", 0),
                "low": review_summary.get("low_count", 0),
                "info": 0,
            },
            "key_changes": analysis.get("key_changes", [])[:5] if analysis else [],
            "recommendations": [],
            "positives": [],
            "summary_comment_id": github_comment_id,
            "summary_comment_url": github_comment_url,
            "inline_comments_posted": 0,
        }
        
        logger.info(f"📤 Sending review data to frontend database...")
        logger.info(f"   URL: {frontend_url}/api/reviews")
        logger.info(f"   Payload: pull_request_id={pull_request_db_id}, status={payload['status']}")
        logger.info(f"   Summary: {payload['summary']}")
        
        # Send to frontend database
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{frontend_url}/api/reviews",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code != 200:
                error_body = response.text
                logger.error(f"❌ Frontend API returned {response.status_code}")
                logger.error(f"   Response: {error_body}")
                raise Exception(f"Frontend API error: {response.status_code} - {error_body}")
            
            result = response.json()
            
        logger.info(f"✅ Stored review in database for PR {pull_request_db_id}")
        logger.info(f"   Review ID: {result.get('id')}")
        logger.info(f"   Issues: {payload['summary']['issuesFound']}, Critical: {payload['summary']['critical']}")
        
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Failed to store review in database: HTTP {e.response.status_code}")
        logger.error(f"   Response: {e.response.text}")
    except Exception as e:
        logger.error(f"❌ Failed to store review in database: {e}", exc_info=True)
        # Don't raise - this is non-critical, GitHub comment is primary output


# ========================================================================
# REVIEW FORMATTING HELPER
# ========================================================================

def format_review_for_github(
    pr_number: int,
    pr_title: str,
    overall_summary: str,
    file_reviews: dict,
    status: str
) -> str:
    """
    Format review for GitHub PR comment with rich markdown
    
    Features:
    - Severity badges (shields.io)
    - Collapsible sections for long content
    - Syntax-highlighted code blocks
    - Color-coded severity indicators
    
    Args:
        pr_number: PR number
        pr_title: PR title
        overall_summary: Generated review summary
        file_reviews: Dict of file-level reviews
        status: Workflow status
    
    Returns:
        Formatted markdown for GitHub comment
    """
    parts = []
    
    # Header with badges
    parts.append(f"# 🤖 AI Code Review - PR #{pr_number}")
    parts.append(f"**{pr_title}**")
    parts.append("")
    
    # Status badge
    if status == "completed":
        parts.append("![Status](https://img.shields.io/badge/status-completed-success)")
    else:
        parts.append("![Status](https://img.shields.io/badge/status-in__progress-yellow)")
    
    # Count severity levels in summary
    critical_count = overall_summary.lower().count("critical")
    high_count = overall_summary.lower().count("high priority") + overall_summary.lower().count("high:")
    
    if critical_count > 0:
        parts.append("![Critical](https://img.shields.io/badge/critical-" + str(critical_count) + "-critical)")
    if high_count > 0:
        parts.append("![High](https://img.shields.io/badge/high-" + str(high_count) + "-orange)")
    
    parts.append("")
    
    # Overall summary with collapsible if long
    parts.append("## 📋 Overall Review")
    parts.append("")
    
    if len(overall_summary) > 3000:
        parts.append("<details>")
        parts.append("<summary><b>Click to expand full review</b></summary>")
        parts.append("")
        parts.append(overall_summary)
        parts.append("")
        parts.append("</details>")
    else:
        parts.append(overall_summary)
    
    parts.append("")
    
    # File-level reviews with rich formatting
    if file_reviews:
        parts.append("---")
        parts.append("")
        parts.append("## 📁 File-by-File Analysis")
        parts.append("")
        
        # Sort files by issues_found (most issues first)
        sorted_files = sorted(
            file_reviews.items(),
            key=lambda x: x[1].get("issues_found", 0),
            reverse=True
        )
        
        for filename, review in sorted_files[:8]:  # Show top 8 files
            if review.get("error"):
                continue
            
            issues_found = review.get("issues_found", 0)
            
            # File header with badge
            parts.append(f"### 📄 `{filename}`")
            parts.append("")
            
            # Language and severity badges
            badge_parts = []
            if review.get("language"):
                badge_parts.append(f"![{review['language']}](https://img.shields.io/badge/language-{review['language']}-blue)")
            
            if issues_found > 0:
                if issues_found >= 3:
                    badge_parts.append(f"![Issues](https://img.shields.io/badge/issues-{issues_found}-red)")
                elif issues_found >= 1:
                    badge_parts.append(f"![Issues](https://img.shields.io/badge/issues-{issues_found}-orange)")
            else:
                badge_parts.append("![Clean](https://img.shields.io/badge/issues-0-success)")
            
            if badge_parts:
                parts.append(" ".join(badge_parts))
                parts.append("")
            
            summary = review.get("summary", "")
            
            # Use collapsible for long reviews
            if len(summary) > 800:
                parts.append("<details>")
                parts.append("<summary><b>View detailed review</b></summary>")
                parts.append("")
                parts.append(summary)
                parts.append("")
                parts.append("</details>")
            else:
                parts.append(summary)
            
            parts.append("")
        
        if len(file_reviews) > 8:
            parts.append(f"<details>")
            parts.append(f"<summary>📋 <i>... and {len(file_reviews) - 8} more files reviewed</i></summary>")
            parts.append("")
            for filename in list(file_reviews.keys())[8:]:
                parts.append(f"- `{filename}`")
            parts.append("")
            parts.append("</details>")
    
    # Footer with GraphRAG branding
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("🔍 **Powered by GraphRAG Technology**")
    parts.append("")
    parts.append("*This review uses Graph Database + Vector Search to find similar code patterns, ")
    parts.append("dependencies, and architectural insights across your codebase.*")
    parts.append("")
    parts.append("**Generated by [Graph Bug AI Reviewer](https://github.com) • Google Gemini 2.5**")
    
    return "\n".join(parts)


async def execute_review_task(
    pr_number: int,
    repo_id: str,
    pr_title: str,
    pr_description: Optional[str],
    files: List[dict],
    base_ref: str,
    head_ref: str,
    installation_id: str,
    pull_request_db_id: str,
    owner: str,
    repo: str,
    gemini_api_key: Optional[str] = None
):
    """
    Background task that executes the complete review workflow
    
    This is where the actual AI review happens:
    1. Run LangGraph workflow
    2. Generate review using Gemini
    3. Post results to GitHub PR
    """
    try:
        logger.info(f"🚀 Starting review workflow for PR #{pr_number} in {repo_id}")
        
        # Track start time for analytics
        start_time = time.time()
        
        # Create user-specific workflow with their Gemini API key
        if gemini_api_key:
            from .gemini_client import create_gemini_client
            user_gemini_client = create_gemini_client(api_key=gemini_api_key)
            user_workflow = create_review_workflow(
                context_builder=context_builder,
                gemini_client=user_gemini_client
            )
            logger.info("✅ Using user's Gemini API key for review")
        else:
            user_workflow = review_workflow
            logger.info("⚠️ Using default Gemini API key (fallback)")
        
        # Execute the LangGraph workflow
        final_state = await user_workflow.run_review(
            pr_number=pr_number,
            repo_id=repo_id,
            pr_title=pr_title,
            pr_description=pr_description,
            files=files,
            base_ref=base_ref,
            head_ref=head_ref
        )
        
        # Track end time
        end_time = time.time()
        final_state["start_time"] = start_time
        final_state["end_time"] = end_time
        
        logger.info(f"✅ Review workflow completed for PR #{pr_number}")
        logger.info(f"   Status: {final_state.get('status')}")
        logger.info(f"   Files processed: {final_state.get('processed_files', 0)}/{final_state.get('total_files', 0)}")
        logger.info(f"   Execution time: {end_time - start_time:.2f}s")
        
        # Post review to GitHub PR
        overall_summary = final_state.get("overall_summary", "")
        
        if overall_summary and github_client:
            try:
                logger.info("=" * 80)
                logger.info(f"📤 Posting review to GitHub PR #{pr_number}...")
                logger.info(f"   Repo: {owner}/{repo}")
                logger.info(f"   Installation ID: {installation_id}")
                logger.info(f"   Review length: {len(overall_summary)} chars")
                
                # Format review for GitHub
                review_body = format_review_for_github(
                    pr_number=pr_number,
                    pr_title=pr_title,
                    overall_summary=overall_summary,
                    file_reviews=final_state.get("file_reviews", {}),
                    status=final_state.get("status", "completed")
                )
                
                logger.info(f"   Formatted review: {len(review_body)} chars")
                
                # Post review comment using GitHub repo format (owner/repo)
                repo_full_name = f"{owner}/{repo}"
                result = await github_client.post_review_comment(
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                    body=review_body,
                    installation_id=int(installation_id),
                    event="COMMENT"
                )
                
                logger.info("✅ Review posted successfully to GitHub!")
                logger.info(f"   Review ID: {result['id']}")
                logger.info(f"   Review URL: {result['html_url']}")
                logger.info("=" * 80)
                
                # Store review in frontend database for analytics
                await store_review_in_database(
                    pull_request_db_id=pull_request_db_id,
                    final_state=final_state,
                    github_comment_id=result['id'],
                    github_comment_url=result['html_url']
                )
                
            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"❌ Failed to post review to GitHub: {e}")
                logger.error(f"   Review was generated but not posted")
                logger.error(f"   Check GitHub App permissions (pull_requests: write)")
                logger.error("=" * 80, exc_info=True)
                
                # Still store the review even if GitHub posting failed
                try:
                    await store_review_in_database(
                        pull_request_db_id=pull_request_db_id,
                        final_state=final_state,
                        github_comment_id=None,
                        github_comment_url=None
                    )
                except Exception as db_error:
                    logger.error(f"❌ Also failed to store review in database: {db_error}")
        elif not overall_summary:
            logger.warning(f"⚠️ No review summary generated, skipping GitHub comment")
        elif not github_client:
            logger.warning(f"⚠️ GitHub client not available, skipping comment posting")
        
        # For now, just log the results
        review_summary = final_state.get("review_summary", {})
        logger.info(f"📊 Review Summary:")
        logger.info(f"   Issues found: {review_summary.get('total_issues', 0)}")
        logger.info(f"   Suggestions: {review_summary.get('total_suggestions', 0)}")
        
        logger.info(f"✨ Review task completed for PR #{pr_number}")
        
    except Exception as e:
        logger.error(f"❌ Review workflow failed for PR #{pr_number}: {e}", exc_info=True)
        # TODO: Update PR status in frontend database with error
        # TODO: Optionally post error comment to GitHub PR


# ========================================================================
# VECTOR SEARCH ENDPOINTS (Phase 3.3)
# ========================================================================

class SemanticSearchRequest(BaseModel):
    repo_id: str
    query: str
    limit: int = 10

class SimilarCodeRequest(BaseModel):
    repo_id: str
    code_snippet: str
    limit: int = 5
    min_score: float = 0.7

class RepositoryStatsResponse(BaseModel):
    total_snippets: int
    by_type: dict
    total_files: int
    top_files: List[tuple]


@app.post("/search/semantic")
async def semantic_search(http_request: Request, request: SemanticSearchRequest):
    """
    Natural language code search
    
    Examples:
    - "functions that handle user authentication"
    - "code that processes payment transactions"
    - "error handling logic"
    
    Returns relevant code snippets based on semantic similarity.
    """
    try:
        logger.info(f"Semantic search for repo {request.repo_id}: '{request.query}'")
        results = vector_db.semantic_code_search(
            repo_id=request.repo_id,
            natural_language_query=request.query,
            limit=request.limit
        )
        return {"query": request.query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/similar")
async def find_similar_code(http_request: Request, request: SimilarCodeRequest):
    """
    Find code similar to a given snippet
    
    Useful for:
    - Detecting duplicate code
    - Finding similar implementations
    - Code pattern analysis
    
    Returns code snippets with similarity scores.
    """
    try:
        logger.info(f"Finding similar code in repo {request.repo_id}")
        results = vector_db.search_similar_code(
            repo_id=request.repo_id,
            code_snippet=request.code_snippet,
            limit=request.limit,
            min_score=request.min_score
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Similar code search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/duplicates")
async def find_duplicates(http_request: Request, request: SimilarCodeRequest):
    """
    Find duplicate code (high similarity threshold: 90%+)
    
    Helps identify:
    - Copy-pasted code
    - Refactoring opportunities
    - Code that should be extracted into shared functions
    """
    try:
        logger.info(f"Finding duplicate code in repo {request.repo_id}")
        # Override min_score to 0.9 for duplicates
        results = vector_db.find_duplicate_code(
            repo_id=request.repo_id,
            code_snippet=request.code_snippet,
            limit=request.limit,
            threshold=0.9
        )
        return {
            "results": results,
            "count": len(results),
            "threshold": 0.9,
            "message": "High similarity matches (90%+) - potential duplicates"
        }
    except Exception as e:
        logger.error(f"Duplicate search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repos/{repo_id}/stats")
async def get_repository_statistics(repo_id: str):
    """
    Get statistics about indexed code in repository
    
    Returns:
    - Total code snippets indexed
    - Breakdown by type (functions, classes, etc.)
    - Total files indexed
    - Top files by snippet count
    """
    try:
        logger.info(f"Getting stats for repo {repo_id}")
        stats = vector_db.get_repository_stats(repo_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting repository stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CONTEXT BUILDER ENDPOINT (Phase 3.5)
# ========================================================================

class PRContextRequest(BaseModel):
    pr_number: int
    repo_id: str
    title: str
    description: Optional[str] = None
    files: List[dict]  # List of FileChange-like dicts


@app.post("/context/pr", response_model=PRContext)
async def build_pr_context(request: PRContextRequest):
    """
    Build comprehensive context for PR review
    
    Combines:
    - Code analysis (issues, patterns, complexity)
    - Graph queries (dependencies, impact, coupling)
    - Vector search (similar code, duplicates)
    
    Returns unified context ready for LangGraph orchestration.
    
    This is the main entry point for the review pipeline.
    """
    try:
        logger.info(f"Building PR context for PR #{request.pr_number} in repo {request.repo_id}")
        
        # Convert dict files to FileChange objects
        from .analyzer import FileChange
        file_changes = [
            FileChange(
                filename=f.get("filename", ""),
                status=f.get("status", "modified"),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                changes=f.get("changes", 0),
                patch=f.get("patch"),
                language=f.get("language")
            )
            for f in request.files
        ]
        
        # Build context
        pr_context = await context_builder.build_pr_context(
            pr_number=request.pr_number,
            repo_id=request.repo_id,
            title=request.title,
            description=request.description,
            files=file_changes
        )
        
        # Log summary
        summary = context_builder.get_context_summary(pr_context)
        logger.info(f"PR context built:\n{summary}")
        
        return pr_context
        
    except Exception as e:
        logger.error(f"Error building PR context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/pr/{pr_number}/summary")
async def get_pr_context_summary(pr_number: int, repo_id: str):
    """
    Get a text summary of PR context
    
    Useful for quick overview without full context object.
    """
    try:
        # This would typically fetch from cache or rebuild
        # For now, return a placeholder
        return {
            "pr_number": pr_number,
            "repo_id": repo_id,
            "message": "Summary endpoint - implement caching in production"
        }
    except Exception as e:
        logger.error(f"Error getting PR summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# LANGGRAPH WORKFLOW ENDPOINTS (Phase 4.1)
# ========================================================================

class WorkflowRequest(BaseModel):
    pr_number: int
    repo_id: str
    pr_title: str
    pr_description: Optional[str] = None
    files: List[dict]


@app.post("/workflow/review")
async def execute_review_workflow(http_request: Request, request: WorkflowRequest):
    """
    Execute the complete LangGraph review workflow
    
    This orchestrates:
    1. Context building (Phase 3)
    2. Review strategy routing
    3. Gemini-powered review generation
    4. Result aggregation
    
    Returns final review state.
    """
    try:
        logger.info(f"Starting workflow for PR #{request.pr_number} in {request.repo_id}")
        
        # Execute workflow
        final_state = await review_workflow.run_review(
            pr_number=request.pr_number,
            repo_id=request.repo_id,
            pr_title=request.pr_title,
            pr_description=request.pr_description,
            files=request.files
        )
        
        return {
            "status": "success",
            "pr_number": request.pr_number,
            "workflow_state": final_state
        }
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/review/stream")
async def stream_review_workflow(request: WorkflowRequest):
    """
    Stream the review workflow execution
    
    Returns Server-Sent Events (SSE) with state updates
    as the workflow progresses through each node.
    """
    try:
        from fastapi.responses import StreamingResponse
        import json
        
        async def event_generator():
            logger.info(f"Starting streaming workflow for PR #{request.pr_number}")
            
            async for state in review_workflow.stream_review(
                pr_number=request.pr_number,
                repo_id=request.repo_id,
                pr_title=request.pr_title,
                pr_description=request.pr_description,
                files=request.files
            ):
                # Send state update as SSE
                event_data = json.dumps({
                    "pr_number": request.pr_number,
                    "status": state.get("status"),
                    "current_step": state.get("current_step"),
                    "processed_files": state.get("processed_files"),
                    "total_files": state.get("total_files")
                })
                yield f"data: {event_data}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Streaming workflow failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/status/{pr_number}")
async def get_workflow_status(pr_number: int, repo_id: str):
    """
    Get current status of a workflow execution
    
    In production, this would query a state store.
    For now, returns placeholder.
    """
    return {
        "pr_number": pr_number,
        "repo_id": repo_id,
        "message": "Status tracking - implement with Redis/DB in production"
    }
