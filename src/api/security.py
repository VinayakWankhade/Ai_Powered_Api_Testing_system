"""
Security utilities for the AI-Powered API Testing System.
"""

import os
import secrets
import hashlib
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from functools import wraps

from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import bleach
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security bearer token scheme
security = HTTPBearer()

class SecurityConfig:
    """Security configuration."""
    
    # API Keys configuration
    VALID_API_KEYS = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()
    MASTER_API_KEY = os.getenv("MASTER_API_KEY")
    
    # CORS configuration
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))
    
    # Input validation
    MAX_SPEC_SIZE = int(os.getenv("MAX_SPEC_SIZE", "10485760"))  # 10MB
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

def get_utc_timestamp() -> datetime:
    """Get timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)

def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication."""
    api_key = credentials.credentials
    
    # Check against valid API keys
    if SecurityConfig.VALID_API_KEYS and api_key not in SecurityConfig.VALID_API_KEYS:
        # Also check against master key
        if SecurityConfig.MASTER_API_KEY and api_key != SecurityConfig.MASTER_API_KEY:
            logger.warning(f"Invalid API key attempted: {hash_api_key(api_key)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Log successful authentication
    logger.info(f"Successful API key authentication: {hash_api_key(api_key)}")
    return api_key

def optional_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Optional authentication for public endpoints with enhanced features for authenticated users."""
    if not credentials:
        return None
    
    try:
        return verify_api_key(credentials)
    except HTTPException:
        return None

class InputValidator:
    """Input validation and sanitization utilities."""
    
    # Regex patterns for validation
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9._/-]+$')
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    @staticmethod
    def sanitize_html(content: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not content:
            return ""
        
        # Allow only safe tags for documentation
        allowed_tags = ['p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li', 'h1', 'h2', 'h3']
        allowed_attributes = {}
        
        return bleach.clean(content, tags=allowed_tags, attributes=allowed_attributes, strip=True)
    
    @staticmethod
    def validate_json_size(content: str, max_size: int = None) -> bool:
        """Validate JSON content size."""
        max_size = max_size or SecurityConfig.MAX_REQUEST_SIZE
        return len(content.encode('utf-8')) <= max_size
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for safety."""
        if not filename or len(filename) > 255:
            return False
        return bool(InputValidator.SAFE_FILENAME_PATTERN.match(filename))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        if not url or len(url) > 2048:
            return False
        return bool(InputValidator.URL_PATTERN.match(url))
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not value:
            return ""
        
        # Truncate if too long
        value = value[:max_length]
        
        # Remove null bytes and control characters
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Basic HTML encoding for safety
        value = (value.replace('&', '&amp;')
                     .replace('<', '&lt;')
                     .replace('>', '&gt;')
                     .replace('"', '&quot;')
                     .replace("'", '&#x27;'))
        
        return value.strip()

class SecureBaseModel(BaseModel):
    """Base model with input validation and sanitization."""
    
    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        """Sanitize string fields."""
        if isinstance(v, str):
            return InputValidator.sanitize_string(v)
        return v

def audit_log(action: str, user_id: str = "anonymous", details: Dict[str, Any] = None):
    """Log security-relevant actions."""
    log_entry = {
        "timestamp": get_utc_timestamp().isoformat(),
        "action": action,
        "user_id": hash_api_key(user_id) if user_id != "anonymous" else "anonymous",
        "details": details or {},
        "ip_address": "redacted"  # Would get from request context
    }
    
    logger.info(f"AUDIT: {log_entry}")

def require_auth(f):
    """Decorator to require authentication for endpoints."""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # Check if api_key is in kwargs (from dependency injection)
        if 'api_key' not in kwargs or not kwargs['api_key']:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await f(*args, **kwargs)
    return wrapper

class SecurityMiddleware:
    """Custom security middleware."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add security headers
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    for key, value in SecurityConfig.SECURITY_HEADERS.items():
                        headers.append([key.encode(), value.encode()])
                    message["headers"] = headers
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# Rate limiting decorators
def rate_limit_strict(max_calls: str):
    """Strict rate limiting for sensitive endpoints."""
    return limiter.limit(max_calls)

def rate_limit_generous(max_calls: str):
    """Generous rate limiting for general endpoints."""
    return limiter.limit(max_calls)

# Input validation models
class SecureAPISpecRequest(SecureBaseModel):
    """Secure API specification request model."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    specification: Dict[str, Any] = Field(...)
    spec_type: str = Field(..., pattern=r'^(openapi|swagger|postman|insomnia)$')
    
    @validator('specification')
    def validate_specification_size(cls, v):
        """Validate specification size."""
        import json
        spec_str = json.dumps(v)
        if not InputValidator.validate_json_size(spec_str, SecurityConfig.MAX_SPEC_SIZE):
            raise ValueError(f"Specification too large. Maximum size: {SecurityConfig.MAX_SPEC_SIZE} bytes")
        return v
    
    @validator('name')
    def validate_name_safety(cls, v):
        """Validate name for safety."""
        if not InputValidator.validate_filename(v):
            raise ValueError("Invalid specification name format")
        return v

class SecureTestGenerationRequest(SecureBaseModel):
    """Secure test generation request model."""
    api_spec_id: str = Field(..., min_length=1, max_length=100)
    test_types: List[str] = Field(default=["functional"], max_items=10)
    coverage_goals: Optional[Dict[str, Any]] = Field(default={})
    
    @validator('test_types')
    def validate_test_types(cls, v):
        """Validate test types."""
        allowed_types = ["functional", "security", "performance", "edge_case", "negative"]
        for test_type in v:
            if test_type not in allowed_types:
                raise ValueError(f"Invalid test type: {test_type}")
        return v

class SecureTestExecutionRequest(SecureBaseModel):
    """Secure test execution request model."""
    test_case_ids: List[str] = Field(..., min_items=1, max_items=100)
    environment: Optional[str] = Field(default="development", max_length=50)
    parallel_execution: bool = Field(default=False)
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment name."""
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid environment name format")
        return v

def validate_request_size(request: Request):
    """Middleware to validate request size."""
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > SecurityConfig.MAX_REQUEST_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request too large. Maximum size: {SecurityConfig.MAX_REQUEST_SIZE} bytes"
        )

# Environment validation
def validate_environment():
    """Validate required environment variables."""
    required_vars = []
    warnings = []
    
    if not SecurityConfig.VALID_API_KEYS and not SecurityConfig.MASTER_API_KEY:
        warnings.append("No API keys configured. Authentication is disabled.")
    
    if not os.getenv("DATABASE_URL"):
        warnings.append("DATABASE_URL not set. Using SQLite fallback.")
    
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set. AI features may be limited.")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"SECURITY WARNING: {warning}")
    
    return {"required_missing": required_vars, "warnings": warnings}

class SecurityAuditLogger:
    """Enhanced security audit logging."""
    
    @staticmethod
    def log_authentication_attempt(success: bool, api_key_hash: str, ip_address: str = "unknown"):
        """Log authentication attempts."""
        audit_log(
            action="authentication_attempt",
            user_id=api_key_hash,
            details={
                "success": success,
                "ip_address": ip_address,
                "timestamp": get_utc_timestamp().isoformat()
            }
        )
    
    @staticmethod
    def log_api_access(endpoint: str, method: str, user_id: str = "anonymous", status_code: int = 200):
        """Log API access."""
        audit_log(
            action="api_access",
            user_id=user_id,
            details={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "timestamp": get_utc_timestamp().isoformat()
            }
        )
    
    @staticmethod
    def log_security_event(event_type: str, severity: str, details: Dict[str, Any]):
        """Log security events."""
        audit_log(
            action="security_event",
            details={
                "event_type": event_type,
                "severity": severity,
                "details": details,
                "timestamp": get_utc_timestamp().isoformat()
            }
        )

def get_secure_cors_origins() -> List[str]:
    """Get secure CORS origins."""
    # In development, allow localhost
    if os.getenv("ENVIRONMENT", "development") == "development":
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080"
        ]
    
    # In production, use configured origins
    return SecurityConfig.ALLOWED_ORIGINS

# Content Security Policy
CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none';"
)

def add_security_headers(response):
    """Add security headers to response."""
    for key, value in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[key] = value
    response.headers["Content-Security-Policy"] = CSP_POLICY
    return response
