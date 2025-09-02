"""
API endpoints for coverage tracking and analytics.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from ...database.connection import get_db
from ...database.models import APISpecification, TestCase, ExecutionSession, TestExecution, CoverageMetrics
from ...coverage.coverage_tracker import CoverageTracker
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class CoverageResponse(BaseModel):
    """Response model for coverage information."""
    api_spec_id: int
    overall_coverage_percentage: float
    endpoint_coverage: Dict[str, Any]
    method_coverage: Dict[str, Any]
    status_code_coverage: Dict[str, Any]
    parameter_coverage: Dict[str, Any]
    test_type_coverage: Dict[str, Any]
    coverage_gaps: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    last_updated: str

class AnalyticsResponse(BaseModel):
    """Response model for analytics information."""
    api_spec_id: int
    time_period: str
    execution_statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    quality_trends: Dict[str, Any]
    failure_analysis: Dict[str, Any]
    improvement_suggestions: List[Dict[str, Any]]
    generated_at: str

@router.get("/coverage-report/{api_spec_id}", response_model=CoverageResponse)
async def get_coverage_report(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive coverage report for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id,
            APISpecification.is_active == True
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        # Initialize coverage tracker
        coverage_tracker = CoverageTracker()
        
        # Get comprehensive coverage data
        coverage_data = await coverage_tracker.get_coverage_report(api_spec_id)
        
        return CoverageResponse(
            api_spec_id=api_spec_id,
            overall_coverage_percentage=coverage_data["overall_coverage"],
            endpoint_coverage=coverage_data["endpoint_coverage"],
            method_coverage=coverage_data["method_coverage"],
            status_code_coverage=coverage_data["status_code_coverage"],
            parameter_coverage=coverage_data["parameter_coverage"],
            test_type_coverage=coverage_data["test_type_coverage"],
            coverage_gaps=coverage_data["coverage_gaps"],
            recommendations=coverage_data["recommendations"],
            last_updated=coverage_data["last_updated"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get coverage report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get coverage report: {str(e)}"
        )

@router.get("/analytics/{api_spec_id}", response_model=AnalyticsResponse)
async def get_analytics_report(
    api_spec_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive analytics report for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id,
            APISpecification.is_active == True
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        # Initialize coverage tracker for analytics
        coverage_tracker = CoverageTracker()
        
        # Get analytics data
        analytics_data = await coverage_tracker.get_analytics_report(
            api_spec_id=api_spec_id,
            days=days
        )
        
        return AnalyticsResponse(
            api_spec_id=api_spec_id,
            time_period=f"{days} days",
            execution_statistics=analytics_data["execution_stats"],
            performance_metrics=analytics_data["performance_metrics"],
            quality_trends=analytics_data["quality_trends"],
            failure_analysis=analytics_data["failure_analysis"],
            improvement_suggestions=analytics_data["suggestions"],
            generated_at=analytics_data["generated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analytics report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics report: {str(e)}"
        )

@router.get("/coverage-trends/{api_spec_id}")
async def get_coverage_trends(
    api_spec_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get coverage trends over time for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        trends_data = await coverage_tracker.get_coverage_trends(
            api_spec_id=api_spec_id,
            days=days
        )
        
        return {
            "api_spec_id": api_spec_id,
            "time_period_days": days,
            "coverage_trends": trends_data["trends"],
            "trend_analysis": trends_data["analysis"],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get coverage trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get coverage trends: {str(e)}"
        )

@router.get("/test-quality-metrics/{api_spec_id}")
async def get_test_quality_metrics(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get test quality metrics for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        quality_metrics = await coverage_tracker.get_test_quality_metrics(api_spec_id)
        
        return {
            "api_spec_id": api_spec_id,
            "quality_metrics": quality_metrics,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test quality metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get test quality metrics: {str(e)}"
        )

@router.get("/performance-analysis/{api_spec_id}")
async def get_performance_analysis(
    api_spec_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get performance analysis for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        performance_data = await coverage_tracker.get_performance_analysis(
            api_spec_id=api_spec_id,
            days=days
        )
        
        return {
            "api_spec_id": api_spec_id,
            "analysis_period_days": days,
            "performance_analysis": performance_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance analysis: {str(e)}"
        )

@router.get("/failure-patterns/{api_spec_id}")
async def get_failure_patterns(
    api_spec_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Analyze failure patterns for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        failure_patterns = await coverage_tracker.analyze_failure_patterns(
            api_spec_id=api_spec_id,
            days=days
        )
        
        return {
            "api_spec_id": api_spec_id,
            "analysis_period_days": days,
            "failure_patterns": failure_patterns,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze failure patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze failure patterns: {str(e)}"
        )

@router.get("/coverage-gaps/{api_spec_id}")
async def get_coverage_gaps(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Identify coverage gaps for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        coverage_gaps = await coverage_tracker.identify_coverage_gaps(api_spec_id)
        
        return {
            "api_spec_id": api_spec_id,
            "coverage_gaps": coverage_gaps["gaps"],
            "prioritized_recommendations": coverage_gaps["recommendations"],
            "gap_analysis": coverage_gaps["analysis"],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to identify coverage gaps: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to identify coverage gaps: {str(e)}"
        )

@router.get("/test-effectiveness/{api_spec_id}")
async def get_test_effectiveness(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Analyze test effectiveness for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        effectiveness_data = await coverage_tracker.analyze_test_effectiveness(api_spec_id)
        
        return {
            "api_spec_id": api_spec_id,
            "test_effectiveness": effectiveness_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze test effectiveness: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze test effectiveness: {str(e)}"
        )

@router.post("/update-coverage/{api_spec_id}")
async def update_coverage_data(
    api_spec_id: int,
    session_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Update coverage data for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        
        if session_id:
            # Update coverage based on specific session
            result = await coverage_tracker.update_coverage_from_session(
                api_spec_id=api_spec_id,
                session_id=session_id
            )
        else:
            # Update coverage based on all recent executions
            result = await coverage_tracker.update_coverage_data(api_spec_id)
        
        return {
            "api_spec_id": api_spec_id,
            "session_id": session_id,
            "update_result": result,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update coverage data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update coverage data: {str(e)}"
        )

@router.get("/dashboard/{api_spec_id}")
async def get_dashboard_data(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive dashboard data for an API specification.
    """
    try:
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        coverage_tracker = CoverageTracker()
        
        # Get dashboard data
        dashboard_data = await coverage_tracker.get_dashboard_data(api_spec_id)
        
        return {
            "api_spec_id": api_spec_id,
            "api_name": api_spec.name,
            "api_version": api_spec.version,
            "dashboard": dashboard_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard data: {str(e)}"
        )

@router.get("/comparative-analysis")
async def get_comparative_analysis(
    api_spec_ids: List[int],
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get comparative analysis across multiple API specifications.
    """
    try:
        if not api_spec_ids:
            raise HTTPException(
                status_code=400,
                detail="At least one API specification ID must be provided"
            )
        
        # Verify all API specifications exist
        api_specs = db.query(APISpecification).filter(
            APISpecification.id.in_(api_spec_ids)
        ).all()
        
        if len(api_specs) != len(api_spec_ids):
            missing_ids = set(api_spec_ids) - {spec.id for spec in api_specs}
            raise HTTPException(
                status_code=404,
                detail=f"API specifications not found: {list(missing_ids)}"
            )
        
        coverage_tracker = CoverageTracker()
        
        # Get comparative analysis
        comparative_data = await coverage_tracker.get_comparative_analysis(
            api_spec_ids=api_spec_ids,
            days=days
        )
        
        return {
            "api_specifications": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "version": spec.version
                }
                for spec in api_specs
            ],
            "analysis_period_days": days,
            "comparative_analysis": comparative_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comparative analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get comparative analysis: {str(e)}"
        )
