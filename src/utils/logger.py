"""
Logging configuration for the API testing framework.
"""

import os
import sys
import json
import hashlib
from loguru import logger as loguru_logger
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

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
    
    # File handler with JSON format for production
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # JSON format for production logs
        file_format = (
            '{"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"logger": "{name}", '
            '"module": "{module}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}", '
            '"process_id": {process.id}, '
            '"thread_id": {thread.id}, '
            '"environment": "' + environment + '"}'
        )
    else:
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    loguru_logger.add(
        log_file,
        format=file_format,
        level=log_level,
        rotation="50 MB",
        retention="60 days",
        compression="zip"
    )
    
    # Separate audit log file
    audit_enabled = os.getenv("AUDIT_LOGGING", "true").lower() == "true"
    if audit_enabled:
        audit_file = os.getenv("AUDIT_LOG_FILE", "./logs/audit.log")
        audit_path = Path(audit_file)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            audit_file,
            format=file_format,
            level="INFO",
            rotation="25 MB",
            retention="90 days",
            compression="zip",
            filter=lambda record: "AUDIT:" in record["message"] or "SECURITY:" in record["message"]
        )
    
    return loguru_logger

class SecurityLogger:
    """Enhanced security logging utilities using loguru."""
    
    def __init__(self, logger_name: Optional[str] = None):
        self.logger = get_logger(logger_name or "security")
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for logging."""
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    
    def log_authentication(self, success: bool, user_identifier: str, 
                          ip_address: str = "unknown", details: Optional[Dict[str, Any]] = None):
        """Log authentication events."""
        hashed_id = self.hash_sensitive_data(user_identifier)
        
        self.logger.info(
            f"AUDIT: Authentication {'successful' if success else 'failed'} for user {hashed_id}",
            audit=True,
            user_id=hashed_id,
            ip_address=ip_address,
            success=success,
            details=details or {}
        )
    
    def log_api_access(self, endpoint: str, method: str, user_id: str = "anonymous",
                      status_code: int = 200, ip_address: str = "unknown",
                      details: Optional[Dict[str, Any]] = None):
        """Log API access events."""
        hashed_id = self.hash_sensitive_data(user_id) if user_id != "anonymous" else "anonymous"
        
        self.logger.info(
            f"AUDIT: {method} {endpoint} - Status: {status_code} - User: {hashed_id}",
            audit=True,
            endpoint=endpoint,
            method=method,
            user_id=hashed_id,
            status_code=status_code,
            ip_address=ip_address,
            details=details or {}
        )
    
    def log_security_event(self, event_type: str, severity: str, message: str,
                          user_id: str = "system", details: Optional[Dict[str, Any]] = None):
        """Log security events."""
        hashed_id = self.hash_sensitive_data(user_id) if user_id != "system" else "system"
        
        log_message = f"SECURITY: {event_type} - {message}"
        
        if severity.upper() == "ERROR":
            self.logger.error(log_message, security=True, event_type=event_type, 
                            user_id=hashed_id, details=details or {})
        elif severity.upper() == "WARNING":
            self.logger.warning(log_message, security=True, event_type=event_type,
                              user_id=hashed_id, details=details or {})
        else:
            self.logger.info(log_message, security=True, event_type=event_type,
                           user_id=hashed_id, details=details or {})
    
    def log_data_access(self, resource_type: str, resource_id: str, action: str,
                       user_id: str = "anonymous", success: bool = True,
                       details: Optional[Dict[str, Any]] = None):
        """Log data access events."""
        hashed_id = self.hash_sensitive_data(user_id) if user_id != "anonymous" else "anonymous"
        
        self.logger.info(
            f"AUDIT: Data access - {action} {resource_type} {resource_id} - User: {hashed_id}",
            audit=True,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            user_id=hashed_id,
            success=success,
            details=details or {}
        )
    
    def log_rate_limit_violation(self, endpoint: str, user_id: str = "anonymous",
                                ip_address: str = "unknown"):
        """Log rate limit violations."""
        hashed_id = self.hash_sensitive_data(user_id) if user_id != "anonymous" else "anonymous"
        
        self.logger.warning(
            f"SECURITY: Rate limit exceeded for {endpoint} - User: {hashed_id}",
            security=True,
            event_type="rate_limit_violation",
            endpoint=endpoint,
            user_id=hashed_id,
            ip_address=ip_address
        )

# Global logger instances
logger = get_logger()
security_logger = SecurityLogger("security")
