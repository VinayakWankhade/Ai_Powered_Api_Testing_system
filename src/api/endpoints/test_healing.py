"""
API endpoints for test healing functionality.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database.connection import get_db
from ...database.models import APISpecification, TestCase, TestExecution
from ...healing.self_healing import SelfHealingSystem
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class HealingRequest(BaseModel):
    """Request model for test healing."""
    test_case_ids: Optional[List[int]] = Field(None, description="Specific test cases to heal (optional)")
    api_spec_id: Optional[int] = Field(None, description="API specification ID to heal all failed tests")
    session_id: Optional[int] = Field(None, description="Execution session ID to heal failed tests")
    healing_strategies: Optional[List[str]] = Field(None, description="Specific healing strategies to apply")
    auto_apply_fixes: Optional[bool] = Field(False, description="Automatically apply suggested fixes")
    max_healing_attempts: Optional[int] = Field(3, description="Maximum healing attempts per test")

class HealingResponse(BaseModel):
    """Response model for test healing."""
    healing_session_id: str
    total_tests_analyzed: int
    tests_healed: int
    tests_failed_healing: int
    healing_success_rate: float
    healing_actions: List[Dict[str, Any]]
    execution_time_seconds: float
    recommendations: List[Dict[str, Any]]

class HealingStatus(BaseModel):
    """Status model for ongoing test healing."""
    healing_session_id: str
    status: str
    progress_percentage: float
    current_test: Optional[str]
    tests_processed: int
    tests_remaining: int
    estimated_completion_time: Optional[str]

class HealingActionRequest(BaseModel):
    """Request to apply a specific healing action."""
    action_id: int
    approve: bool = Field(..., description="Whether to approve and apply the healing action")
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for the healing action")

@router.post("/heal-tests", response_model=HealingResponse)
async def heal_tests(
    request: HealingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze and heal failed test cases.
    
    Can heal specific test cases, all tests from an API spec, or tests from an execution session.
    """
    try:
        healing_system = SelfHealingSystem()
        
        # Determine which tests to heal
        test_cases_to_heal = []
        
        if request.test_case_ids:
            # Heal specific test cases
            test_cases_to_heal = db.query(TestCase).filter(
                TestCase.id.in_(request.test_case_ids),
                TestCase.is_active == True
            ).all()
            
        elif request.api_spec_id:
            # Heal all failed tests from API specification
            api_spec = db.query(APISpecification).filter(
                APISpecification.id == request.api_spec_id
            ).first()
            
            if not api_spec:
                raise HTTPException(
                    status_code=404,
                    detail=f"API specification with ID {request.api_spec_id} not found"
                )
            
            # Find test cases with recent failures
            test_cases_to_heal = healing_system.get_failed_test_cases(
                api_spec_id=request.api_spec_id
            )
            
        elif request.session_id:
            # Heal failed tests from execution session
            test_cases_to_heal = healing_system.get_failed_test_cases_from_session(
                session_id=request.session_id
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Must specify test_case_ids, api_spec_id, or session_id"
            )
        
        if not test_cases_to_heal:
            return HealingResponse(
                healing_session_id="no-healing-required",
                total_tests_analyzed=0,
                tests_healed=0,
                tests_failed_healing=0,
                healing_success_rate=0.0,
                healing_actions=[],
                execution_time_seconds=0.0,
                recommendations=[{
                    "type": "info",
                    "message": "No failed test cases found that require healing",
                    "priority": "low"
                }]
            )
        
        # Execute healing process
        healing_result = await healing_system.heal_tests(
            test_cases=test_cases_to_heal,
            strategies=request.healing_strategies,
            auto_apply=request.auto_apply_fixes,
            max_attempts=request.max_healing_attempts
        )
        
        return HealingResponse(
            healing_session_id=healing_result["session_id"],
            total_tests_analyzed=healing_result["total_analyzed"],
            tests_healed=healing_result["healed_count"],
            tests_failed_healing=healing_result["failed_count"],
            healing_success_rate=healing_result["success_rate"],
            healing_actions=healing_result["actions"],
            execution_time_seconds=healing_result["execution_time"],
            recommendations=healing_result["recommendations"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test healing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test healing failed: {str(e)}"
        )

@router.post("/heal-tests-async")
async def heal_tests_async(
    request: HealingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start test healing process asynchronously in the background.
    
    Returns immediately with a healing session ID to track progress.
    """
    try:
        healing_system = SelfHealingSystem()
        
        # Create healing session
        session_info = await healing_system.create_healing_session(
            test_case_ids=request.test_case_ids,
            api_spec_id=request.api_spec_id,
            session_id=request.session_id
        )
        
        # Start healing in background
        background_tasks.add_task(
            healing_system.heal_tests_background,
            session_info["healing_session_id"],
            request.test_case_ids,
            request.api_spec_id,
            request.session_id,
            request.healing_strategies,
            request.auto_apply_fixes,
            request.max_healing_attempts
        )
        
        return {
            "message": "Test healing started in background",
            "healing_session_id": session_info["healing_session_id"],
            "status": "running",
            "progress_url": f"/api/v1/healing-status/{session_info['healing_session_id']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start async test healing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start async test healing: {str(e)}"
        )

@router.get("/healing-status/{healing_session_id}", response_model=HealingStatus)
async def get_healing_status(
    healing_session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the status of an ongoing or completed test healing process.
    """
    try:
        healing_system = SelfHealingSystem()
        status_info = await healing_system.get_healing_status(healing_session_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Healing session with ID {healing_session_id} not found"
            )
        
        return HealingStatus(
            healing_session_id=healing_session_id,
            status=status_info["status"],
            progress_percentage=status_info.get("progress_percentage", 0),
            current_test=status_info.get("current_test"),
            tests_processed=status_info.get("tests_processed", 0),
            tests_remaining=status_info.get("tests_remaining", 0),
            estimated_completion_time=status_info.get("estimated_completion")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get healing status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get healing status: {str(e)}"
        )

@router.get("/healing-actions")
async def list_healing_actions(
    api_spec_id: Optional[int] = None,
    test_case_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List healing actions with optional filtering.
    """
    try:
        # For MVP, return empty list since HealingAction model doesn't exist
        # In full implementation, would query actual healing actions
        actions = []
        
        return {
            "healing_actions": [],  # Empty for MVP
            "total_actions": 0,
            "has_more": False,
            "message": "Healing actions not implemented in MVP"
        }
        
    except Exception as e:
        logger.error(f"Failed to list healing actions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list healing actions: {str(e)}"
        )

@router.post("/healing-actions/{action_id}/apply")
async def apply_healing_action(
    action_id: int,
    request: HealingActionRequest,
    db: Session = Depends(get_db)
):
    """
    Apply or reject a specific healing action.
    """
    try:
        # For MVP, healing actions are not implemented
        raise HTTPException(
            status_code=404,
            detail=f"Healing actions not implemented in MVP"
        )
        
        healing_system = SelfHealingSystem()
        
        if request.approve:
            # Apply the healing action
            result = await healing_system.apply_healing_action(
                action_id=action_id,
                custom_parameters=request.custom_parameters
            )
            
            return {
                "message": "Healing action applied successfully",
                "action_id": action_id,
                "success": result["success"],
                "details": result.get("details", {}),
                "test_case_updated": result.get("test_case_updated", False)
            }
        else:
            # Reject the healing action
            result = await healing_system.reject_healing_action(action_id)
            
            return {
                "message": "Healing action rejected",
                "action_id": action_id,
                "status": "rejected"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply healing action: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply healing action: {str(e)}"
        )

@router.get("/healing-recommendations/{api_spec_id}")
async def get_healing_recommendations(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get healing recommendations for an API specification.
    """
    try:
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        healing_system = SelfHealingSystem()
        recommendations = await healing_system.get_healing_recommendations(api_spec_id)
        
        return {
            "api_spec_id": api_spec_id,
            "recommendations": recommendations,
            "generated_at": healing_system.get_current_timestamp()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get healing recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get healing recommendations: {str(e)}"
        )

@router.get("/healing-statistics/{api_spec_id}")
async def get_healing_statistics(
    api_spec_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get healing statistics for an API specification.
    """
    try:
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        healing_system = SelfHealingSystem()
        statistics = await healing_system.get_healing_statistics(
            api_spec_id=api_spec_id,
            days=days
        )
        
        return {
            "api_spec_id": api_spec_id,
            "time_period_days": days,
            "statistics": statistics,
            "generated_at": healing_system.get_current_timestamp()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get healing statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get healing statistics: {str(e)}"
        )

@router.get("/healing-history")
async def get_healing_history(
    api_spec_id: Optional[int] = None,
    test_case_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get healing history with optional filtering.
    """
    try:
        healing_system = SelfHealingSystem()
        history = await healing_system.get_healing_history(
            api_spec_id=api_spec_id,
            test_case_id=test_case_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "healing_history": history,
            "total_records": len(history),
            "has_more": len(history) == limit,
            "filters": {
                "api_spec_id": api_spec_id,
                "test_case_id": test_case_id
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get healing history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get healing history: {str(e)}"
        )
