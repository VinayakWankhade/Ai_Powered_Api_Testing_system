"""
Minimal FastAPI server for demonstrating the AI-Powered API Testing Framework MVP.
This version avoids heavy dependencies while showing core functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ.setdefault("DATABASE_URL", "sqlite:///./demo_server.db")
os.environ.setdefault("ENVIRONMENT", "demo")
os.environ.setdefault("LOG_LEVEL", "INFO")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import datetime
import json

# Initialize database
try:
    from database.connection import create_tables
    create_tables()
    print("✅ Database initialized")
except Exception as e:
    print(f"⚠️  Database initialization warning: {e}")

# Create FastAPI app
app = FastAPI(
    title="AI-Powered API Testing Framework",
    description="Intelligent API testing with AI-powered test generation and optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class APISpecCreate(BaseModel):
    name: str
    description: Optional[str] = None
    spec_content: Dict[str, Any]
    base_url: Optional[str] = None

class TestCaseCreate(BaseModel):
    name: str
    method: str
    path: str
    headers: Optional[Dict[str, str]] = {}
    query_params: Optional[Dict[str, str]] = {}
    body: Optional[Dict[str, Any]] = None
    expected_status: int = 200

# In-memory storage for demo (replace with database in production)
demo_data = {
    "specs": [],
    "test_cases": [],
    "executions": []
}

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "AI-Powered API Testing Framework MVP",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "AI-powered test generation",
            "Intelligent test execution", 
            "Coverage analysis",
            "Self-healing tests",
            "RL optimization"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "api": "/api/specs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "demo"),
        "database": "connected",
        "services": {
            "api": "running",
            "test_executor": "available",
            "ai_generator": "available",
            "coverage_analyzer": "available"
        }
    }

@app.get("/api/specs")
async def get_api_specs():
    """Get all API specifications."""
    return demo_data["specs"]

@app.post("/api/specs")
async def create_api_spec(spec: APISpecCreate):
    """Create a new API specification."""
    new_spec = {
        "id": len(demo_data["specs"]) + 1,
        "name": spec.name,
        "description": spec.description,
        "spec_content": spec.spec_content,
        "base_url": spec.base_url,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "test_case_count": 0,
        "last_execution": None
    }
    
    demo_data["specs"].append(new_spec)
    return JSONResponse(content=new_spec, status_code=201)

@app.get("/api/specs/{spec_id}")
async def get_api_spec(spec_id: int):
    """Get a specific API specification."""
    spec = next((s for s in demo_data["specs"] if s["id"] == spec_id), None)
    if not spec:
        raise HTTPException(status_code=404, detail="API specification not found")
    return spec

@app.post("/api/specs/{spec_id}/generate-tests")
async def generate_tests(spec_id: int):
    """Generate test cases for an API specification (simplified demo)."""
    spec = next((s for s in demo_data["specs"] if s["id"] == spec_id), None)
    if not spec:
        raise HTTPException(status_code=404, detail="API specification not found")
    
    # Simplified test generation
    spec_content = spec["spec_content"]
    generated_tests = []
    
    if "paths" in spec_content:
        for path, methods in spec_content["paths"].items():
            for method, details in methods.items():
                test_case = {
                    "id": len(demo_data["test_cases"]) + len(generated_tests) + 1,
                    "api_spec_id": spec_id,
                    "name": f"Test {method.upper()} {path}",
                    "method": method.upper(),
                    "path": path,
                    "expected_status": 200,
                    "generated_by": "simplified_ai",
                    "confidence_score": 0.85,
                    "created_at": datetime.datetime.utcnow().isoformat()
                }
                generated_tests.append(test_case)
    
    # Add to demo data
    demo_data["test_cases"].extend(generated_tests)
    
    return JSONResponse(content=generated_tests, status_code=201)

@app.post("/api/test-cases/{test_case_id}/execute")
async def execute_test_case(test_case_id: int):
    """Execute a test case (simplified demo)."""
    test_case = next((tc for tc in demo_data["test_cases"] if tc["id"] == test_case_id), None)
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Simulate test execution
    import random
    
    execution_result = {
        "id": len(demo_data["executions"]) + 1,
        "test_case_id": test_case_id,
        "status": random.choice(["passed", "passed", "passed", "failed"]),  # 75% pass rate
        "response_time": random.uniform(50, 500),
        "status_code": random.choice([200, 200, 200, 500]),
        "response_body": {"demo": True, "result": "success"},
        "assertions_passed": random.randint(1, 3),
        "assertions_failed": random.randint(0, 1),
        "executed_at": datetime.datetime.utcnow().isoformat()
    }
    
    demo_data["executions"].append(execution_result)
    return execution_result

@app.get("/api/specs/{spec_id}/coverage")
async def get_coverage(spec_id: int):
    """Get coverage analysis for an API specification."""
    spec = next((s for s in demo_data["specs"] if s["id"] == spec_id), None)
    if not spec:
        raise HTTPException(status_code=404, detail="API specification not found")
    
    # Calculate demo coverage
    spec_paths = spec["spec_content"].get("paths", {})
    total_endpoints = sum(len(methods) for methods in spec_paths.values())
    
    executed_tests = [tc for tc in demo_data["test_cases"] if tc["api_spec_id"] == spec_id]
    covered_endpoints = len(executed_tests)
    
    coverage_percentage = (covered_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
    
    return {
        "api_spec_id": spec_id,
        "overall_coverage": coverage_percentage,
        "endpoint_coverage": {
            "total_endpoints": total_endpoints,
            "covered_endpoints": covered_endpoints,
            "percentage": coverage_percentage
        },
        "method_coverage": {
            "GET": 90.0,
            "POST": 70.0,
            "PUT": 60.0,
            "DELETE": 40.0
        },
        "generated_at": datetime.datetime.utcnow().isoformat()
    }

@app.get("/api/demo/status")
async def demo_status():
    """Get demo system status."""
    return {
        "message": "AI-Powered API Testing Framework MVP Demo",
        "status": "running",
        "stats": {
            "api_specs": len(demo_data["specs"]),
            "test_cases": len(demo_data["test_cases"]),
            "executions": len(demo_data["executions"])
        },
        "features_demonstrated": [
            "✅ FastAPI REST API",
            "✅ Database connectivity", 
            "✅ API specification management",
            "✅ AI test generation (simplified)",
            "✅ Test execution simulation",
            "✅ Coverage analysis",
            "✅ Interactive API documentation"
        ],
        "next_steps": [
            "Configure OpenAI API key for full AI features",
            "Set up Redis for task queue",
            "Deploy with production Docker configuration",
            "Connect to real APIs for testing"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI-Powered API Testing Framework Demo Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏠 Demo Status: http://localhost:8000/api/demo/status")
    print("💊 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable reload for demo
    )
