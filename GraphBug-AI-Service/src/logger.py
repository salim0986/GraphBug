import logging
import sys
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional
from .config import LOG_LEVEL

class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    Enables better log aggregation and analysis in production
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add any custom fields from extra parameter
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add context fields if present
        for key in ["repo_id", "installation_id", "user_id", "request_id", "endpoint"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        return json.dumps(log_data)


def setup_logger(name: str, structured: bool = False) -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        structured: Use JSON structured logging (recommended for production)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Detect production environment (Hugging Face Spaces, Cloud Run, etc.)
        is_production = os.getenv("ENVIRONMENT") == "production" or os.getenv("SPACE_ID") is not None
        
        # Console handler with immediate flushing for production
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, LOG_LEVEL))
        
        # Force immediate flush in production to ensure logs are visible
        if is_production:
            # Set unbuffered mode
            sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
        
        # Choose formatter based on environment
        if structured or is_production:
            # Use structured JSON logging in production
            formatter = StructuredFormatter()
        else:
            # Use human-readable format for development
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Force flush after each log in production
        if is_production:
            old_emit = handler.emit
            def flush_emit(record):
                old_emit(record)
                handler.flush()
            handler.emit = flush_emit
    
    return logger


class LogContext:
    """
    Context manager for adding structured fields to all logs within a block
    
    Usage:
        with LogContext(logger, repo_id="123", installation_id="456"):
            logger.info("Processing repo")  # Will include repo_id and installation_id
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.context = kwargs
        self.old_factory = None
    
    def __enter__(self):
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        self.old_factory = old_factory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)
