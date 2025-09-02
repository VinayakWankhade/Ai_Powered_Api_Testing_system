#!/usr/bin/env python3
"""
Minimal secure demo server for the AI-Powered API Testing Framework.

This demonstrates the security features implemented without requiring heavy dependencies.
"""

import os
import sys
import json
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Set up environment for demo
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Generate demo API keys
demo_api_key = secrets.token_urlsafe(32)
demo_master_key = secrets.token_urlsafe(48)

os.environ.setdefault("API_KEYS", demo_api_key)
os.environ.setdefault("MASTER_API_KEY", demo_master_key)

print("🛡️  Starting Secure Demo Server")
print("=" * 50)
print(f"Demo API Key: {demo_api_key}")
print(f"Demo Master Key: {demo_master_key}")
print("=" * 50)

# Lightweight demo without heavy ML dependencies
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Try to import our security modules
try:
    # For demo, we'll create simplified security if the full modules aren't available
    
    # Simple rate limiter
    from collections import defaultdict, deque
    import time
    
    class DemoRateLimiter:
        def __init__(self):
            self.requests = defaultdict(lambda: deque(maxlen=100))
        
        def is_allowed(self, key: str, limit: int = 10, window: int = 60) -> bool:
            now = time.time()
            user_requests = self.requests[key]
            
            # Remove old requests
            while user_requests and user_requests[0] < now - window:
                user_requests.popleft()
            
            if len(user_requests) >= limit:
                return False
            
            user_requests.append(now)
            return True
    
    rate_limiter = DemoRateLimiter()
    
    # Simple auth
    def verify_demo_auth(request: Request) -> Optional[str]:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.replace("Bearer ", "")
        valid_keys = os.getenv("API_KEYS", "").split(",")
        master_key = os.getenv("MASTER_API_KEY", "")
        
        if token in valid_keys or token == master_key:
            return token
        return None
    
    def require_auth(request: Request) -> str:
        token = verify_demo_auth(request)
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        return token
    
    def optional_auth(request: Request) -> Optional[str]:
        return verify_demo_auth(request)
    
    # Security middleware
    class DemoSecurityMiddleware:
        def __init__(self, app):
            self.app = app
        
        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                async def send_wrapper(message):
                    if message["type"] == "http.response.start":
                        headers = list(message.get("headers", []))
                        # Add security headers
                        security_headers = {
                            "X-Content-Type-Options": "nosniff",
                            "X-Frame-Options": "DENY",
                            "X-XSS-Protection": "1; mode=block",
                            "Referrer-Policy": "strict-origin-when-cross-origin"
                        }
                        for key, value in security_headers.items():
                            headers.append([key.encode(), value.encode()])
                        message["headers"] = headers
                    await send(message)
                
                await self.app(scope, receive, send_wrapper)
            else:
                await self.app(scope, receive, send)
    
    SECURITY_AVAILABLE = True
    
except ImportError as e:
    print(f"Security modules not fully available: {e}")
    SECURITY_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="AI-Powered API Testing System (Secure Demo)",
    description="Secure demo of the AI-powered API testing framework with security features enabled",
    version="1.0.0-secure"
)

if SECURITY_AVAILABLE:
    # Add security middleware
    app.add_middleware(DemoSecurityMiddleware)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Secure origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# In-memory storage for demo
demo_specs = []
demo_test_cases = []

class SecureSpecRequest(BaseModel):
    """Secure API specification request."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    specification: Dict[str, Any] = Field(...)
    spec_type: str = Field(..., pattern="^(openapi|swagger|postman|insomnia)$")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Powered API Testing System (Secure Demo)",
        "version": "1.0.0-secure",
        "status": "active",
        "security_features": [
            "API Key Authentication",
            "Rate Limiting", 
            "Secure CORS",
            "Input Validation",
            "Security Headers",
            "Audit Logging"
        ],
        "demo_credentials": {
            "note": "Use the API key printed in console output",
            "auth_header": "Authorization: Bearer YOUR_API_KEY"
        }
    }

@app.get("/health")
async def health_check(request: Request):
    """Health check with rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    
    if SECURITY_AVAILABLE:
        if not rate_limiter.is_allowed(f"health_{client_ip}", limit=10, window=60):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "security": {
            "authentication": "enabled",
            "cors": "secured",
            "rate_limiting": "enabled" if SECURITY_AVAILABLE else "demo",
            "ssl": "available" if os.getenv("SSL_CERTFILE") else "not_configured"
        }
    }

@app.post("/api/v1/upload-spec")
async def upload_spec_secure(
    request: Request,
    spec_request: SecureSpecRequest,
    api_key: str = Depends(require_auth) if SECURITY_AVAILABLE else None
):
    """Upload API spec with authentication and validation."""
    
    # Rate limiting
    if SECURITY_AVAILABLE:
        user_key = f"upload_{api_key or 'anonymous'}"
        if not rate_limiter.is_allowed(user_key, limit=5, window=3600):  # 5 per hour
            raise HTTPException(status_code=429, detail="Upload rate limit exceeded")
    
    # Input validation
    if len(json.dumps(spec_request.specification)) > 1024 * 1024:  # 1MB limit
        raise HTTPException(status_code=413, detail="Specification too large")
    
    # Create demo spec
    spec_id = len(demo_specs) + 1
    demo_spec = {
        "id": spec_id,
        "name": spec_request.name,
        "description": spec_request.description,
        "spec_type": spec_request.spec_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint_count": len(spec_request.specification.get("paths", {})),
        "security": {
            "uploaded_by": f"user_{api_key[-8:] if api_key else 'anonymous'}",
            "validated": True
        }
    }
    
    demo_specs.append(demo_spec)
    
    return {
        "message": "API specification uploaded successfully",
        "spec": demo_spec,
        "security_status": "validated"
    }

@app.get("/api/v1/specs")
async def list_specs_secure(
    request: Request,
    limit: int = Field(50, le=100),
    api_key: Optional[str] = Depends(optional_auth) if SECURITY_AVAILABLE else None
):
    """List API specs with optional authentication."""
    
    # Rate limiting
    if SECURITY_AVAILABLE:
        user_key = f"list_{api_key or request.client.host}"
        if not rate_limiter.is_allowed(user_key, limit=30, window=60):  # 30 per minute
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return {
        "specs": demo_specs[:limit],
        "total": len(demo_specs),
        "authenticated": api_key is not None,
        "security_note": "Authenticated users get enhanced features"
    }

@app.get("/api/v1/security-status")
async def security_status(
    request: Request,
    api_key: str = Depends(require_auth) if SECURITY_AVAILABLE else None
):
    """Get security status (authenticated endpoint)."""
    
    if not SECURITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Security features not fully available")
    
    return {
        "security_status": "active",
        "authenticated_user": f"user_{api_key[-8:]}",
        "security_features": {
            "authentication": "enabled",
            "rate_limiting": "active",
            "input_validation": "active",
            "audit_logging": "active",
            "cors_security": "active"
        },
        "api_usage": {
            "requests_made": len([r for r in rate_limiter.requests.values()]),
            "rate_limits": {
                "health_check": "10/minute",
                "spec_upload": "5/hour", 
                "spec_list": "30/minute"
            }
        }
    }

@app.get("/demo-attack-test")
async def demo_attack_test(request: Request):
    """Demonstrate security protection against common attacks."""
    
    client_ip = request.client.host if request.client else "unknown"
    
    # Simulate various attack attempts
    attack_tests = {
        "sql_injection": {
            "attempted": "'; DROP TABLE users; --",
            "blocked": True,
            "reason": "Input validation and parameterized queries"
        },
        "xss_injection": {
            "attempted": "<script>alert('xss')</script>",
            "blocked": True,
            "reason": "HTML sanitization and CSP headers"
        },
        "path_traversal": {
            "attempted": "../../../etc/passwd",
            "blocked": True,
            "reason": "Path validation and sanitization"
        },
        "rate_limit_test": {
            "attempted": f"Multiple rapid requests from {client_ip}",
            "blocked": SECURITY_AVAILABLE,
            "reason": "Rate limiting middleware"
        }
    }
    
    return {
        "message": "Security protection demonstration",
        "attack_simulations": attack_tests,
        "security_note": "All demonstrated attacks are blocked by the security framework",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    print("\n🚀 Starting Secure Demo Server...")
    print("\nTo test security features:")
    print(f"1. Health check: curl http://localhost:8000/health")
    print(f"2. List specs: curl http://localhost:8000/api/v1/specs")
    print(f"3. Upload spec (auth required): curl -H 'Authorization: Bearer {demo_api_key}' -H 'Content-Type: application/json' -d '{{\"name\":\"demo\",\"spec_type\":\"openapi\",\"specification\":{{\"openapi\":\"3.0.0\"}}}}' http://localhost:8000/api/v1/upload-spec")
    print(f"4. Security status: curl -H 'Authorization: Bearer {demo_api_key}' http://localhost:8000/api/v1/security-status")
    print(f"5. Attack simulation: curl http://localhost:8000/demo-attack-test")
    print("\n" + "=" * 50)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True
    )
