"""
Logging configuration for the API testing framework.
"""

import os
import sys
from loguru import logger as loguru_logger
from typing import Optional

def get_logger(name: Optional[str] = None):
    """
    Get a configured logger instance.
    
    Args:
        name: Optional name for the logger
        
    Returns:
        Configured logger instance
    """
    # Remove default handler
    loguru_logger.remove()
    
    # Get configuration from environment
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "./logs/api_testing.log")
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # Console handler
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    if debug_mode:
        loguru_logger.add(
            sys.stdout,
            format=console_format,
            level="DEBUG",
            colorize=True
        )
    else:
        loguru_logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True
        )
    
    # File handler
    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    
    loguru_logger.add(
        log_file,
        format=file_format,
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return loguru_logger

# Global logger instance
logger = get_logger()
