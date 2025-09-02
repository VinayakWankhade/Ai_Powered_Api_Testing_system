"""
FastAPI application for the AI-Powered API Testing System.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..database.connection import create_tables, get_db
from ..database.models import SpecType, TestType, RLAlgorithm
from .endpoints import (
    specs, test_generation, test_execution, 
    test_healing, coverage, rl_optimization, dashboard
)
from ..utils.logger import get_logger
from .security import (
    SecurityConfig, SecurityMiddleware, limiter, 
    get_secure_cors_origins, get_utc_timestamp, 
    validate_environment, SecurityAuditLogger
)

logger = get_logger(__name__)

# Validate environment and log security warnings
env_validation = validate_environment()
for warning in env_validation["warnings"]:
    logger.warning(f"SECURITY: {warning}")

# Initialize database
create_tables()

# Create FastAPI app
app = FastAPI(
    title="AI-Powered API Testing System",
    description="""
    An advanced, agentic framework for automated API testing that integrates 
    Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), 
    and hybrid Reinforcement Learning (RL) for intelligent test generation, 
    execution, and optimization.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure security middleware
app.add_middleware(SecurityMiddleware)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure secure CORS
secure_origins = get_secure_cors_origins()
logger.info(f"Configuring CORS for origins: {secure_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=secure_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Include routers
app.include_router(specs.router, prefix="/api/v1", tags=["API Specifications"])
app.include_router(test_generation.router, prefix="/api/v1", tags=["Test Generation"])
app.include_router(test_execution.router, prefix="/api/v1", tags=["Test Execution"])
app.include_router(test_healing.router, prefix="/api/v1", tags=["Test Healing"])
app.include_router(coverage.router, prefix="/api/v1", tags=["Coverage & Analytics"])
app.include_router(rl_optimization.router, prefix="/api/v1", tags=["RL Optimization"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard & Metrics"])

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "AI-Powered API Testing System",
        "version": "1.0.0",
        "status": "active",
        "features": [
            "AI-Powered Test Generation",
            "Sandboxed Test Execution", 
            "Hybrid RL Optimization",
            "Self-Healing Mechanism",
            "Coverage Analytics",
            "RAG-Enhanced Context"
        ],
        "endpoints": {
            "upload_spec": "/api/v1/upload-spec",
            "generate_tests": "/api/v1/generate-tests",
            "run_tests": "/api/v1/run-tests",
            "heal_tests": "/api/v1/heal-tests",
            "coverage_report": "/api/v1/coverage-report/{api_spec_id}",
            "optimize_tests": "/api/v1/optimize-tests"
        }
    }

@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check endpoint with rate limiting."""
    return {
        "status": "healthy",
        "timestamp": get_utc_timestamp().isoformat(),
        "database": "connected",
        "ai_services": "active",
        "security": {
            "authentication": "enabled" if SecurityConfig.VALID_API_KEYS or SecurityConfig.MASTER_API_KEY else "disabled",
            "cors": "secured",
            "rate_limiting": "enabled"
        }
    }

@app.get("/status")
@limiter.limit("5/minute")
async def system_status(request: Request):
    """Get detailed system status."""
    try:
        from ..database.connection import get_db_session
        from ..database.models import APISpecification, TestCase, ExecutionSession
        
        db = get_db_session()
        
        # Get system statistics
        total_specs = db.query(APISpecification).count()
        total_test_cases = db.query(TestCase).count()
        total_sessions = db.query(ExecutionSession).count()
        
        db.close()
        
        return {
            "system": {
                "status": "operational",
                "uptime": "active",
                "version": "1.0.0"
            },
            "statistics": {
                "api_specifications": total_specs,
                "test_cases": total_test_cases,
                "execution_sessions": total_sessions
            },
            "services": {
                "database": "connected",
                "ai_engine": "active",
                "rl_optimizer": "active",
                "test_executor": "active",
                "healing_system": "active"
            },
            "timestamp": get_utc_timestamp().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "system": {
                    "status": "degraded",
                    "error": str(e)
                },
                "timestamp": get_utc_timestamp().isoformat()
            }
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": get_utc_timestamp().isoformat(),
            "request_id": getattr(exc, 'request_id', None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred" if not os.getenv("DEBUG") else str(exc),
            "timestamp": get_utc_timestamp().isoformat(),
            "support": "Contact support with this timestamp for assistance"
        }
    )

if __name__ == "__main__":
    import uvicorn
    from ..config.ssl_config import ssl_config
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # SSL configuration
    ssl_config_dict = ssl_config.get_uvicorn_ssl_config()
    
    # Validate SSL setup
    ssl_validation = ssl_config.validate_certificates()
    for issue in ssl_validation["issues"]:
        logger.error(f"SSL ISSUE: {issue}")
    for warning in ssl_validation["warnings"]:
        logger.warning(f"SSL WARNING: {warning}")
    
    # Server configuration
    server_config = {
        "app": "src.api.main:app",
        "host": host,
        "port": port,
        "reload": debug,
        "log_level": "debug" if debug else "info",
        "access_log": True,
        "server_header": False,  # Hide server header for security
        "date_header": False     # Hide date header for security
    }
    
    # Add SSL configuration if available
    if ssl_config_dict:
        server_config.update(ssl_config_dict)
        logger.info(f"Starting server with HTTPS on {host}:{port}")
    else:
        logger.info(f"Starting server with HTTP on {host}:{port}")
        if os.getenv("ENVIRONMENT") == "production":
            logger.error("WARNING: Running in production without HTTPS!")
    
    uvicorn.run(**server_config)
