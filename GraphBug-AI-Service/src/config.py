import os
from pathlib import Path
from typing import Set

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, use environment variables directly
    pass

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphbug123")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional, for Qdrant Cloud

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Application Settings
TEMP_REPOS_DIR = os.getenv("TEMP_REPOS_DIR", "./temp_repos")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Directories to ignore during repository scanning
IGNORE_DIRS: Set[str] = {
    ".git", ".github", ".vscode", ".idea",
    "node_modules", "venv", "env", "ENV",
    "dist", "build", "out", "target", "bin", "obj",
    "__pycache__", "coverage", "tmp", "temp", 
    "migrations", "vendor", ".next", ".nuxt",
    "bower_components", "jspm_packages"
}

# Ensure temp repos directory exists
Path(TEMP_REPOS_DIR).mkdir(parents=True, exist_ok=True)
