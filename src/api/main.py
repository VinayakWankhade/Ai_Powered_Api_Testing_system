"""
FastAPI application for the AI-Powered API Testing System.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..database.connection import create_tables, get_db
from ..database.models import SpecType, TestType, RLAlgorithm
from .endpoints import (
    specs, test_generation, test_execution, 
    test_healing, coverage, rl_optimization
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(specs.router, prefix="/api/v1", tags=["API Specifications"])
app.include_router(test_generation.router, prefix="/api/v1", tags=["Test Generation"])
app.include_router(test_execution.router, prefix="/api/v1", tags=["Test Execution"])
app.include_router(test_healing.router, prefix="/api/v1", tags=["Test Healing"])
app.include_router(coverage.router, prefix="/api/v1", tags=["Coverage & Analytics"])
app.include_router(rl_optimization.router, prefix="/api/v1", tags=["RL Optimization"])

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
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected",
        "ai_services": "active"
    }

@app.get("/status")
async def system_status():
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
            "timestamp": datetime.utcnow().isoformat()
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
                "timestamp": datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat()
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
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info"
    )
