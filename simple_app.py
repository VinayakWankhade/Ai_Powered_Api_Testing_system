"""
Simple FastAPI application for initial testing.
"""

import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "simple_app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info"
    )
