"""
API endpoints for test execution.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database.connection import get_db
from ...database.models import APISpecification, TestCase, ExecutionSession, TestType
from ...execution.test_executor import HybridTestExecutor
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class TestExecutionRequest(BaseModel):
    """Request model for test execution."""
    api_spec_id: int = Field(..., description="API specification ID")
    test_case_ids: Optional[List[int]] = Field(None, description="Specific test cases to run (optional)")
    execution_config: Optional[Dict[str, Any]] = Field(None, description="Execution configuration")
    max_concurrent_tests: Optional[int] = Field(5, description="Maximum concurrent tests")
    timeout_seconds: Optional[int] = Field(300, description="Test execution timeout")
    include_performance_tests: Optional[bool] = Field(True, description="Include performance tests")
    include_security_tests: Optional[bool] = Field(True, description="Include security tests")

class TestExecutionResponse(BaseModel):
    """Response model for test execution."""
    session_id: int
    api_spec_id: int
    status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time_seconds: float
    success_rate: float
    coverage_percentage: float
    started_at: str
    completed_at: Optional[str]
    execution_summary: Dict[str, Any]

class TestExecutionStatus(BaseModel):
    """Status model for ongoing test execution."""
    session_id: int
    status: str
    progress_percentage: float
    current_test: Optional[str]
    tests_completed: int
    tests_remaining: int
    estimated_completion_time: Optional[str]

@router.post("/run-tests", response_model=TestExecutionResponse)
async def run_tests(
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Execute API tests for a given specification.
    
    Can run all tests or specific test cases based on the request.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == request.api_spec_id,
            APISpecification.is_active == True
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {request.api_spec_id} not found"
            )
        
        # Get test cases to execute
        if request.test_case_ids:
            test_cases = db.query(TestCase).filter(
                TestCase.id.in_(request.test_case_ids),
                TestCase.api_spec_id == request.api_spec_id,
                TestCase.is_active == True
            ).all()
        else:
            # Run all active test cases for the API spec
            test_cases = db.query(TestCase).filter(
                TestCase.api_spec_id == request.api_spec_id,
                TestCase.is_active == True
            ).all()
        
        if not test_cases:
            raise HTTPException(
                status_code=404,
                detail="No active test cases found for execution"
            )
        
        # Initialize test executor
        executor = HybridTestExecutor()
        
        # Execute tests
        execution_result = await executor.execute_test_suite(
            api_spec_id=request.api_spec_id,
            test_cases=test_cases,
            execution_config=request.execution_config or {},
            max_concurrent=request.max_concurrent_tests,
            timeout=request.timeout_seconds
        )
        
        return TestExecutionResponse(
            session_id=execution_result["session_id"],
            api_spec_id=request.api_spec_id,
            status=execution_result["status"],
            total_tests=execution_result["total_tests"],
            passed_tests=execution_result["passed_tests"],
            failed_tests=execution_result["failed_tests"],
            skipped_tests=execution_result["skipped_tests"],
            execution_time_seconds=execution_result["execution_time"],
            success_rate=execution_result["success_rate"],
            coverage_percentage=execution_result.get("coverage_percentage", 0),
            started_at=execution_result["started_at"],
            completed_at=execution_result.get("completed_at"),
            execution_summary=execution_result.get("summary", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test execution failed: {str(e)}"
        )

@router.post("/run-tests-async", response_model=Dict[str, Any])
async def run_tests_async(
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Execute API tests asynchronously in the background.
    
    Returns immediately with a session ID to track progress.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == request.api_spec_id,
            APISpecification.is_active == True
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {request.api_spec_id} not found"
            )
        
        # Initialize test executor
        executor = HybridTestExecutor()
        
        # Create execution session
        session_info = await executor.create_execution_session(
            api_spec_id=request.api_spec_id,
            test_case_ids=request.test_case_ids
        )
        
        # Start tests in background
        background_tasks.add_task(
            executor.execute_test_suite_background,
            session_info["session_id"],
            request.api_spec_id,
            request.test_case_ids,
            request.execution_config or {},
            request.max_concurrent_tests,
            request.timeout_seconds
        )
        
        return {
            "message": "Test execution started in background",
            "session_id": session_info["session_id"],
            "api_spec_id": request.api_spec_id,
            "status": "running",
            "progress_url": f"/api/v1/execution-status/{session_info['session_id']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async test execution failed to start: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start async test execution: {str(e)}"
        )

@router.get("/execution-status/{session_id}", response_model=TestExecutionStatus)
async def get_execution_status(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the status of an ongoing or completed test execution.
    """
    try:
        executor = HybridTestExecutor()
        status_info = await executor.get_execution_status(session_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Execution session with ID {session_id} not found"
            )
        
        return TestExecutionStatus(
            session_id=session_id,
            status=status_info["status"],
            progress_percentage=status_info.get("progress_percentage", 0),
            current_test=status_info.get("current_test"),
            tests_completed=status_info.get("tests_completed", 0),
            tests_remaining=status_info.get("tests_remaining", 0),
            estimated_completion_time=status_info.get("estimated_completion")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution status: {str(e)}"
        )

@router.get("/execution-sessions")
async def list_execution_sessions(
    api_spec_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List execution sessions with optional filtering.
    """
    try:
        query = db.query(ExecutionSession)
        
        if api_spec_id:
            query = query.filter(ExecutionSession.api_spec_id == api_spec_id)
        
        if status:
            query = query.filter(ExecutionSession.status == status)
        
        sessions = query.offset(offset).limit(limit).order_by(
            ExecutionSession.created_at.desc()
        ).all()
        
        return {
            "sessions": [
                {
                    "session_id": session.id,
                    "api_spec_id": session.api_spec_id,
                    "status": session.status,
                    "total_tests": session.total_tests,
                    "passed_tests": session.passed_tests,
                    "failed_tests": session.failed_tests,
                    "skipped_tests": session.skipped_tests,
                    "success_rate": session.success_rate,
                    "execution_time": session.execution_time_seconds,
                    "created_at": session.created_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None
                }
                for session in sessions
            ],
            "total_sessions": len(sessions),
            "has_more": len(sessions) == limit
        }
        
    except Exception as e:
        logger.error(f"Failed to list execution sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list execution sessions: {str(e)}"
        )

@router.get("/execution-sessions/{session_id}/details")
async def get_execution_session_details(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific execution session.
    """
    try:
        from ...database.models import TestExecution
        
        session = db.query(ExecutionSession).filter(
            ExecutionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Execution session with ID {session_id} not found"
            )
        
        # Get individual test executions
        test_executions = db.query(TestExecution).filter(
            TestExecution.session_id == session_id
        ).all()
        
        return {
            "session": {
                "session_id": session.id,
                "api_spec_id": session.api_spec_id,
                "status": session.status,
                "total_tests": session.total_tests,
                "passed_tests": session.passed_tests,
                "failed_tests": session.failed_tests,
                "skipped_tests": session.skipped_tests,
                "success_rate": session.success_rate,
                "execution_time": session.execution_time_seconds,
                "coverage_percentage": session.coverage_percentage,
                "configuration": session.configuration,
                "created_at": session.created_at.isoformat(),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None
            },
            "test_executions": [
                {
                    "test_execution_id": te.id,
                    "test_case_id": te.test_case_id,
                    "test_name": te.test_name,
                    "endpoint": te.endpoint,
                    "method": te.method,
                    "status": te.status.value,
                    "execution_time": te.execution_time_seconds,
                    "request_data": te.request_data,
                    "response_data": te.response_data,
                    "error_message": te.error_message,
                    "assertions_passed": te.assertions_passed,
                    "assertions_failed": te.assertions_failed,
                    "started_at": te.started_at.isoformat() if te.started_at else None,
                    "completed_at": te.completed_at.isoformat() if te.completed_at else None
                }
                for te in test_executions
            ],
            "execution_summary": {
                "total_test_executions": len(test_executions),
                "fastest_test": min(te.execution_time_seconds for te in test_executions if te.execution_time_seconds) if test_executions else 0,
                "slowest_test": max(te.execution_time_seconds for te in test_executions if te.execution_time_seconds) if test_executions else 0,
                "average_test_time": sum(te.execution_time_seconds for te in test_executions if te.execution_time_seconds) / len(test_executions) if test_executions else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution session details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get execution session details: {str(e)}"
        )

@router.delete("/execution-sessions/{session_id}")
async def cancel_execution_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Cancel a running execution session.
    """
    try:
        executor = HybridTestExecutor()
        success = await executor.cancel_execution(session_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Execution session with ID {session_id} not found or cannot be cancelled"
            )
        
        return {"message": f"Execution session {session_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel execution session: {str(e)}"
        )
