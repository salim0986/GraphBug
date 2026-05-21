"""
Security utilities for GraphBug AI Service
Provides path traversal protection, input sanitization, and validation helpers
"""

import os
import re
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator, Field


def safe_join(base_path: str, *paths: str) -> str:
    """
    Safely join paths and prevent directory traversal attacks
    
    Args:
        base_path: The base directory that must contain the result
        *paths: Path components to join
        
    Returns:
        str: Safe joined path
        
    Raises:
        ValueError: If path traversal is detected
        
    Example:
        >>> safe_join("/app/repos", "user_repo", "file.py")
        "/app/repos/user_repo/file.py"
        
        >>> safe_join("/app/repos", "../etc/passwd")
        ValueError: Path traversal detected
    """
    base = Path(base_path).resolve()
    target = (base / Path(*paths)).resolve()
    
    # Ensure target is within base
    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(f"Path traversal detected: {target} is outside {base}")
    
    return str(target)


def sanitize_repo_id(repo_id: str) -> str:
    """
    Sanitize repository ID to prevent injection attacks
    
    Args:
        repo_id: Repository identifier
        
    Returns:
        str: Sanitized repo ID (alphanumeric, dash, underscore only)
        
    Raises:
        ValueError: If repo_id contains invalid characters
    """
    # Allow only alphanumeric, dash, and underscore
    if not re.match(r'^[a-zA-Z0-9_-]+$', repo_id):
        raise ValueError(f"Invalid repo_id: {repo_id}. Only alphanumeric, dash, and underscore allowed.")
    
    if len(repo_id) > 200:
        raise ValueError(f"repo_id too long: {len(repo_id)} characters (max 200)")
    
    return repo_id


def sanitize_owner_repo(owner: str, repo: str) -> tuple[str, str]:
    """
    Sanitize GitHub owner and repo names
    
    Args:
        owner: GitHub username/org
        repo: Repository name
        
    Returns:
        tuple: (sanitized_owner, sanitized_repo)
        
    Raises:
        ValueError: If inputs contain invalid characters
    """
    # GitHub usernames: alphanumeric and dash
    if not re.match(r'^[a-zA-Z0-9-]+$', owner):
        raise ValueError(f"Invalid owner: {owner}")
    
    # GitHub repo names: alphanumeric, dash, underscore, dot
    if not re.match(r'^[a-zA-Z0-9._-]+$', repo):
        raise ValueError(f"Invalid repo: {repo}")
    
    if len(owner) > 100 or len(repo) > 100:
        raise ValueError("Owner or repo name too long (max 100 characters)")
    
    return owner, repo


def sanitize_file_path(file_path: str, max_length: int = 500) -> str:
    """
    Sanitize file path for safe operations
    
    Args:
        file_path: File path to sanitize
        max_length: Maximum allowed path length
        
    Returns:
        str: Sanitized file path
        
    Raises:
        ValueError: If path is invalid
    """
    # Check for null bytes
    if '\0' in file_path:
        raise ValueError("Null byte detected in file path")
    
    # Check length
    if len(file_path) > max_length:
        raise ValueError(f"File path too long: {len(file_path)} > {max_length}")
    
    # Normalize path to prevent tricks like "///" or "\.\"
    normalized = os.path.normpath(file_path)
    
    # Check for parent directory references
    if normalized.startswith('..') or '/..' in normalized or '\\..' in normalized:
        raise ValueError(f"Parent directory reference detected: {file_path}")
    
    return normalized


# Pydantic Models for Input Validation

class GitHubIdentifierModel(BaseModel):
    """Base model for GitHub identifiers with validation"""
    owner: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9-]+$')
    repo: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9._-]+$')


class PRNumberModel(BaseModel):
    """Model for PR number validation"""
    pr_number: int = Field(..., gt=0, lt=1000000, description="PR number must be between 1 and 999999")


class RepoIdModel(BaseModel):
    """Model for repository ID validation"""
    repo_id: str = Field(..., min_length=1, max_length=200, pattern=r'^[a-zA-Z0-9_-]+$')


class FilePathModel(BaseModel):
    """Model for file path validation"""
    file_path: str = Field(..., min_length=1, max_length=500)
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path for security"""
        return sanitize_file_path(v)


class CommitSHAModel(BaseModel):
    """Model for Git commit SHA validation"""
    sha: str = Field(..., min_length=7, max_length=40, pattern=r'^[a-fA-F0-9]+$')


def validate_installation_id(installation_id: str) -> str:
    """
    Validate GitHub App installation ID
    
    Args:
        installation_id: Installation ID to validate
        
    Returns:
        str: Validated installation ID
        
    Raises:
        ValueError: If installation_id is invalid
    """
    # Installation IDs should be numeric strings
    if not installation_id.isdigit():
        raise ValueError(f"Invalid installation_id: {installation_id}")
    
    if len(installation_id) > 20:
        raise ValueError("Installation ID too long")
    
    return installation_id


def sanitize_error_message(error: Exception, include_details: bool = False) -> dict:
    """
    Sanitize error messages to prevent information leakage
    
    Args:
        error: Exception to sanitize
        include_details: Whether to include stack trace (dev mode only)
        
    Returns:
        dict: Sanitized error response
    """
    # Generic error types
    error_type = type(error).__name__
    
    # Don't expose internal paths or sensitive details
    if include_details:
        return {
            "error": str(error),
            "type": error_type
        }
    else:
        # Production: generic error messages
        if "permission" in str(error).lower() or "access" in str(error).lower():
            return {"error": "Access denied", "code": "ACCESS_DENIED"}
        elif "not found" in str(error).lower():
            return {"error": "Resource not found", "code": "NOT_FOUND"}
        else:
            return {"error": "An error occurred", "code": "INTERNAL_ERROR"}


# Request size limits
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CODE_SNIPPET_SIZE = 100 * 1024   # 100 KB
MAX_DIFF_SIZE = 500 * 1024            # 500 KB


def validate_request_size(content_length: Optional[int]) -> None:
    """
    Validate request size to prevent DOS attacks
    
    Args:
        content_length: Content-Length header value
        
    Raises:
        ValueError: If request is too large
    """
    if content_length and content_length > MAX_REQUEST_SIZE:
        raise ValueError(f"Request too large: {content_length} bytes (max {MAX_REQUEST_SIZE})")
