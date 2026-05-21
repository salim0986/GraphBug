import os
from tree_sitter_languages import get_language, get_parser
import tree_sitter
from .logger import setup_logger

logger = setup_logger(__name__)

EXTENSION_MAP = {
    # Web / JS
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    
    # Backend / Core
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",     # C/C++ Header
    ".hpp": "cpp", # C++ Header
    
    # Systems / Scripting
    ".sh": "bash",
    ".lua": "lua",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    
    # Config / Infra
    ".tf": "hcl",        # Terraform
    ".dockerfile": "dockerfile",
    ".make": "make",
    
    # Docs
    ".md": "markdown"
}

def detect_language_from_filename(filename: str) -> str:
    """
    Detect programming language from filename extension
    
    Args:
        filename: File path or name
        
    Returns:
        Language name or None if unsupported
    """
    _, ext = os.path.splitext(filename)
    return EXTENSION_MAP.get(ext.lower())

class UniversalParser:
    def __init__(self):
        # FIX: Use relative path to this file, not CWD.
        # This ensures it finds 'src/queries' even if you run the script from root.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_query_path = os.path.join(current_dir, "queries")

    def parse_file(self, file_path):
        """
        Returns: (captures, code_bytes) or (None, None)
        """
        _, ext = os.path.splitext(file_path)
        lang_name = EXTENSION_MAP.get(ext)
        
        if not lang_name:
            return None, None

        try:
            # 1. Load Language & Parser
            language = get_language(lang_name)
            parser = get_parser(lang_name)
            
            # 2. Read Code as BYTES (Tree-sitter native)
            with open(file_path, "rb") as f:
                code_bytes = f.read()
            
            tree = parser.parse(code_bytes)

            # 3. Load Query
            query_file = os.path.join(self.base_query_path, lang_name, "tags.scm")
            
            # Fallback: specific languages might not have tags.scm, skip them gracefully
            if not os.path.exists(query_file):
                return None, None

            with open(query_file, "r") as f:
                query_scm = f.read()

            # 4. Execute Query
            # Use language.query() method (correct API for tree-sitter-languages)
            query = language.query(query_scm)
            captures = query.captures(tree.root_node)
            
            return captures, code_bytes

        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None, None
        except Exception as e:
            logger.warning(f"Parser error in {file_path}: {e}")
            return None, None
    
    def parse_code(self, code_bytes: bytes, lang_name: str):
        """
        Parse code from bytes (in-memory)
        
        Args:
            code_bytes: Code content as bytes
            lang_name: Language name (python, javascript, etc.)
            
        Returns:
            (captures, code_bytes) or (None, None)
        """
        if not lang_name or lang_name == "text":
            logger.warning(f"Invalid language '{lang_name}', skipping parse")
            return None, None
        
        try:
            # 1. Load Language & Parser
            language = get_language(lang_name)
            parser = get_parser(lang_name)
            
            # 2. Parse bytes
            tree = parser.parse(code_bytes)
            
            # 3. Load Query
            query_file = os.path.join(self.base_query_path, lang_name, "tags.scm")
            
            # Fallback: specific languages might not have tags.scm, skip them gracefully
            if not os.path.exists(query_file):
                return None, None
            
            with open(query_file, "r") as f:
                query_scm = f.read()
            
            # 4. Execute Query  
            # Use language.query() method (correct API for tree-sitter-languages)
            query = language.query(query_scm)
            captures = query.captures(tree.root_node)
            
            return captures, code_bytes
        
        except Exception as e:
            logger.warning(f"Parser error for {lang_name}: {e}")
            return None, None