"""
Secure configuration management for the AI-Powered API Testing System.
"""

import os
import secrets
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.logger import get_logger

logger = get_logger(__name__)

class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = self.environment == "production"
        self.is_development = self.environment == "development"
        
        # Initialize encryption
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher_suite = Fernet(self._encryption_key)
        
        # Validate configuration
        self._validate_configuration()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data."""
        key_file = Path(".encryption_key")
        
        if key_file.exists() and not self.is_development:
            # In production, read from file
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            password = os.getenv("ENCRYPTION_PASSWORD", "default-dev-password").encode()
            salt = os.getenv("ENCRYPTION_SALT", "default-salt").encode()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Save key in development
            if self.is_development:
                with open(key_file, "wb") as f:
                    f.write(key)
                    
            return key
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not value:
            return ""
        return self._cipher_suite.encrypt(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        if not encrypted_value:
            return ""
        try:
            return self._cipher_suite.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return ""
    
    def _validate_configuration(self):
        """Validate configuration for security issues."""
        issues = []
        warnings = []
        
        # Check for production-specific requirements
        if self.is_production:
            required_production_vars = [
                "DATABASE_URL",
                "SECRET_KEY", 
                "ALLOWED_ORIGINS",
                "API_KEYS",
                "ENCRYPTION_PASSWORD"
            ]
            
            for var in required_production_vars:
                if not os.getenv(var):
                    issues.append(f"Required production environment variable missing: {var}")
            
            # Check for insecure defaults
            if os.getenv("DEBUG", "").lower() == "true":
                warnings.append("DEBUG mode enabled in production")
                
            if not os.getenv("SSL_KEYFILE") or not os.getenv("SSL_CERTFILE"):
                warnings.append("SSL certificates not configured for production")
        
        # Log issues and warnings
        for issue in issues:
            logger.error(f"SECURITY ISSUE: {issue}")
        
        for warning in warnings:
            logger.warning(f"SECURITY WARNING: {warning}")
        
        if issues:
            raise RuntimeError(f"Security configuration issues found: {issues}")
    
    # Database configuration
    @property
    def database_url(self) -> str:
        """Get database URL with fallback to SQLite."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            if self.is_production:
                raise ValueError("DATABASE_URL required in production")
            return "sqlite:///./api_testing.db"
        return db_url
    
    # API Keys
    @property
    def api_keys(self) -> List[str]:
        """Get valid API keys."""
        keys_str = os.getenv("API_KEYS", "")
        if not keys_str:
            if self.is_production:
                raise ValueError("API_KEYS required in production")
            return []
        return [key.strip() for key in keys_str.split(",") if key.strip()]
    
    @property
    def master_api_key(self) -> Optional[str]:
        """Get master API key."""
        return os.getenv("MASTER_API_KEY")
    
    # AI Service configuration
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        key = os.getenv("OPENAI_API_KEY")
        if self.is_production and not key:
            logger.warning("OpenAI API key not configured in production")
        return key
    
    @property
    def ai_rate_limit(self) -> int:
        """Get AI service rate limit."""
        return int(os.getenv("AI_RATE_LIMIT_PER_HOUR", "1000"))
    
    # CORS configuration
    @property
    def allowed_origins(self) -> List[str]:
        """Get allowed CORS origins."""
        if self.is_development:
            return [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8080",
                "http://127.0.0.1:8080"
            ]
        
        origins_str = os.getenv("ALLOWED_ORIGINS", "")
        if not origins_str:
            logger.warning("ALLOWED_ORIGINS not configured for production")
            return []
        
        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]
    
    # Rate limiting
    @property
    def rate_limits(self) -> Dict[str, str]:
        """Get rate limit configurations."""
        return {
            "default": os.getenv("RATE_LIMIT_DEFAULT", "100/minute"),
            "auth": os.getenv("RATE_LIMIT_AUTH", "10/minute"),
            "api_spec": os.getenv("RATE_LIMIT_API_SPEC", "50/hour"),
            "test_generation": os.getenv("RATE_LIMIT_TEST_GEN", "20/hour"),
            "test_execution": os.getenv("RATE_LIMIT_TEST_EXEC", "100/hour"),
            "ai_requests": os.getenv("RATE_LIMIT_AI", "200/hour")
        }
    
    # File size limits
    @property
    def max_file_sizes(self) -> Dict[str, int]:
        """Get maximum file size limits."""
        return {
            "api_spec": int(os.getenv("MAX_SPEC_SIZE", str(10 * 1024 * 1024))),  # 10MB
            "request": int(os.getenv("MAX_REQUEST_SIZE", str(1 * 1024 * 1024))),  # 1MB
            "upload": int(os.getenv("MAX_UPLOAD_SIZE", str(5 * 1024 * 1024)))   # 5MB
        }
    
    # SSL/TLS configuration
    @property
    def ssl_config(self) -> Dict[str, Optional[str]]:
        """Get SSL/TLS configuration."""
        return {
            "keyfile": os.getenv("SSL_KEYFILE"),
            "certfile": os.getenv("SSL_CERTFILE"),
            "ca_certs": os.getenv("SSL_CA_CERTS"),
            "verify_mode": os.getenv("SSL_VERIFY_MODE", "required")
        }
    
    # Security headers
    @property
    def security_headers(self) -> Dict[str, str]:
        """Get security headers configuration."""
        base_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        if self.is_production:
            base_headers.update({
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                "Content-Security-Policy": self._get_csp_policy()
            })
        
        return base_headers
    
    def _get_csp_policy(self) -> str:
        """Get Content Security Policy."""
        if self.is_development:
            return (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss:;"
            )
        
        return (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self';"
        )
    
    # Logging configuration
    @property
    def log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "format": os.getenv("LOG_FORMAT", "json" if self.is_production else "text"),
            "file": os.getenv("LOG_FILE"),
            "max_size": int(os.getenv("LOG_MAX_SIZE", str(100 * 1024 * 1024))),  # 100MB
            "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
            "audit_enabled": os.getenv("AUDIT_LOGGING", "true").lower() == "true"
        }
    
    # Feature flags
    @property
    def feature_flags(self) -> Dict[str, bool]:
        """Get feature flags."""
        return {
            "ai_generation": os.getenv("FEATURE_AI_GENERATION", "true").lower() == "true",
            "rl_optimization": os.getenv("FEATURE_RL_OPTIMIZATION", "true").lower() == "true",
            "self_healing": os.getenv("FEATURE_SELF_HEALING", "true").lower() == "true",
            "parallel_execution": os.getenv("FEATURE_PARALLEL_EXECUTION", "true").lower() == "true",
            "metrics_collection": os.getenv("FEATURE_METRICS", "true").lower() == "true",
            "advanced_analytics": os.getenv("FEATURE_ANALYTICS", "false").lower() == "true"
        }
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)
    
    def validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key or len(api_key) < 16:
            return False
        
        # Check for obvious weak patterns
        if api_key in ["test", "demo", "admin", "password", "12345"]:
            return False
        
        return True

# Global configuration instance
config = SecureConfig()

def get_config() -> SecureConfig:
    """Get global configuration instance."""
    return config

def reload_config():
    """Reload configuration (useful for configuration updates)."""
    global config
    config = SecureConfig()
    logger.info("Configuration reloaded")
    return config
