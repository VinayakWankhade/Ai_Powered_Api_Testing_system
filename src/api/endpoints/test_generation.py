"""
API endpoints for AI-powered test generation.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database.connection import get_db
from ...database.models import TestType
from ...ai.test_generator import AITestGenerator
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class TestGenerationRequest(BaseModel):
    """Request model for test generation."""
    api_spec_id: int = Field(..., description="API specification ID")
    endpoint_path: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method")
    test_types: Optional[List[TestType]] = Field(None, description="Types of tests to generate")
    count: int = Field(3, description="Number of test cases to generate")
    include_edge_cases: bool = Field(True, description="Include edge cases")
    custom_context: Optional[str] = Field(None, description="Additional context for generation")

class TestSuiteGenerationRequest(BaseModel):
    """Request model for test suite generation."""
    api_spec_id: int = Field(..., description="API specification ID")
    include_all_endpoints: bool = Field(True, description="Include all endpoints")
    endpoint_filter: Optional[List[str]] = Field(None, description="Specific endpoints to include")
    test_types: Optional[List[TestType]] = Field(None, description="Types of tests to generate")

@router.post("/generate-tests")
async def generate_tests(
    request: TestGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Generate AI-powered test cases for a specific endpoint.
    
    Uses LLM with RAG to create context-aware test cases.
    """
    try:
        generator = AITestGenerator()
        
        test_cases = await generator.generate_test_cases(
            api_spec_id=request.api_spec_id,
            endpoint_path=request.endpoint_path,
            method=request.method,
            test_types=request.test_types,
            count=request.count,
            include_edge_cases=request.include_edge_cases,
            custom_context=request.custom_context
        )
        
        return {
            "message": f"Generated {len(test_cases)} test cases",
            "test_cases": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "description": tc.description,
                    "test_type": tc.test_type.value,
                    "endpoint": tc.endpoint,
                    "method": tc.method,
                    "test_data": tc.test_data,
                    "expected_response": tc.expected_response,
                    "assertions": tc.assertions,
                    "generated_by_llm": tc.generated_by_llm
                }
                for tc in test_cases
            ],
            "generation_metadata": {
                "endpoint": f"{request.method} {request.endpoint_path}",
                "test_types": [tt.value for tt in (request.test_types or [])],
                "count_requested": request.count,
                "count_generated": len(test_cases)
            }
        }
        
    except Exception as e:
        logger.error(f"Test generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test generation failed: {str(e)}"
        )

@router.post("/generate-test-suite")
async def generate_test_suite(
    request: TestSuiteGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Generate a complete test suite for an API specification.
    
    Creates test cases for all or specified endpoints.
    """
    try:
        generator = AITestGenerator()
        
        result = await generator.generate_test_suite(
            api_spec_id=request.api_spec_id,
            include_all_endpoints=request.include_all_endpoints,
            endpoint_filter=request.endpoint_filter,
            test_types=request.test_types
        )
        
        return {
            "message": f"Generated test suite with {result['statistics']['total_test_cases']} test cases",
            "test_suite": {
                "api_spec_id": request.api_spec_id,
                "total_test_cases": len(result['test_cases']),
                "test_cases": [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "description": tc.description,
                        "test_type": tc.test_type.value,
                        "endpoint": tc.endpoint,
                        "method": tc.method,
                        "generated_by_llm": tc.generated_by_llm
                    }
                    for tc in result['test_cases']
                ]
            },
            "statistics": result['statistics']
        }
        
    except Exception as e:
        logger.error(f"Test suite generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test suite generation failed: {str(e)}"
        )

@router.get("/test-cases/{api_spec_id}")
async def get_test_cases(
    api_spec_id: int,
    test_type: Optional[TestType] = None,
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get test cases for an API specification.
    """
    try:
        from ...database.models import TestCase
        
        query = db.query(TestCase).filter(TestCase.api_spec_id == api_spec_id)
        
        if test_type:
            query = query.filter(TestCase.test_type == test_type)
        
        if active_only:
            query = query.filter(TestCase.is_active == True)
        
        test_cases = query.offset(offset).limit(limit).all()
        
        return {
            "api_spec_id": api_spec_id,
            "test_cases": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "description": tc.description,
                    "test_type": tc.test_type.value,
                    "endpoint": tc.endpoint,
                    "method": tc.method,
                    "test_data": tc.test_data,
                    "expected_response": tc.expected_response,
                    "assertions": tc.assertions,
                    "generated_by_llm": tc.generated_by_llm,
                    "success_rate": tc.success_rate,
                    "selection_count": tc.selection_count,
                    "created_at": tc.created_at.isoformat()
                }
                for tc in test_cases
            ],
            "total": len(test_cases)
        }
        
    except Exception as e:
        logger.error(f"Failed to get test cases: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get test cases: {str(e)}"
        )

@router.get("/generation-history")
async def get_generation_history(
    api_spec_id: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get AI generation history and statistics.
    """
    try:
        generator = AITestGenerator()
        
        history = generator.get_generation_history(
            api_spec_id=api_spec_id,
            limit=limit
        )
        
        return {
            "generation_history": [
                {
                    "id": log.id,
                    "ai_model": log.ai_model,
                    "prompt_template": log.prompt_template[:200] + "..." if len(log.prompt_template) > 200 else log.prompt_template,
                    "generation_time_ms": log.generation_time_ms,
                    "token_usage": log.token_usage,
                    "validation_passed": log.validation_passed,
                    "validation_errors": log.validation_errors,
                    "created_at": log.created_at.isoformat()
                }
                for log in history
            ],
            "total_records": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get generation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get generation history: {str(e)}"
        )
