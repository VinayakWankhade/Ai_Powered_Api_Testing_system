"""
Dashboard API endpoints for metrics and analytics.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from ...database.connection import get_db
from ...database.models import (
    APISpecification, TestCase, ExecutionSession, TestExecution, 
    CoverageMetrics, RLModel, TestType, TestStatus, AIGenerationLog
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.get("/dashboard/metrics")
async def get_dashboard_metrics(db: Session = Depends(get_db)):
    """
    Get comprehensive dashboard metrics for the entire system.
    """
    try:
        # System status
        system_status = {
            "status": "healthy",
            "uptime": "operational",
            "version": "1.0.0"
        }
        
        # API specifications metrics
        total_specs = db.query(APISpecification).count()
        active_specs = db.query(APISpecification).filter(APISpecification.is_active == True).count()
        recent_specs = db.query(APISpecification).filter(
            APISpecification.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        api_specifications = {
            "total": total_specs,
            "active": active_specs,
            "recent_uploads": recent_specs
        }
        
        # Test cases metrics
        total_test_cases = db.query(TestCase).count()
        ai_generated_tests = db.query(TestCase).filter(TestCase.generated_by_llm == True).count()
        
        # Calculate average success rate
        avg_success_rate = db.query(func.avg(TestCase.success_rate)).scalar() or 0
        
        # Test type distribution
        test_type_distribution = {}
        for test_type in TestType:
            count = db.query(TestCase).filter(TestCase.test_type == test_type).count()
            test_type_distribution[test_type.value] = count
        
        test_cases = {
            "total": total_test_cases,
            "generated_by_ai": ai_generated_tests,
            "success_rate": avg_success_rate * 100,
            "types_distribution": test_type_distribution
        }
        
        # Execution metrics
        total_sessions = db.query(ExecutionSession).count()
        
        # Tests executed today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        tests_today = db.query(TestExecution).filter(
            TestExecution.started_at >= today_start
        ).count()
        
        # Average success rate from recent sessions
        recent_sessions = db.query(ExecutionSession).filter(
            ExecutionSession.created_at >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if recent_sessions:
            total_tests_recent = sum(s.total_tests for s in recent_sessions if s.total_tests)
            total_passed_recent = sum(s.passed_tests for s in recent_sessions if s.passed_tests)
            avg_session_success_rate = (total_passed_recent / total_tests_recent * 100) if total_tests_recent > 0 else 0
        else:
            avg_session_success_rate = 0
        
        # Average execution time
        avg_execution_time = db.query(func.avg(ExecutionSession.duration_seconds)).scalar() or 0
        
        executions = {
            "total_sessions": total_sessions,
            "tests_executed_today": tests_today,
            "average_success_rate": avg_session_success_rate,
            "average_execution_time": avg_execution_time
        }
        
        # Coverage metrics
        latest_coverage = db.query(CoverageMetrics).order_by(
            CoverageMetrics.measured_at.desc()
        ).limit(10).all()
        
        if latest_coverage:
            avg_endpoint_coverage = sum(c.endpoint_coverage_pct for c in latest_coverage) / len(latest_coverage)
            avg_method_coverage = sum(c.method_coverage_pct for c in latest_coverage) / len(latest_coverage)
            
            # Create coverage trend data (last 7 days)
            coverage_trend = []
            for i in range(7):
                date = datetime.utcnow() - timedelta(days=i)
                date_coverage = [c for c in latest_coverage if c.measured_at.date() == date.date()]
                
                if date_coverage:
                    endpoint_avg = sum(c.endpoint_coverage_pct for c in date_coverage) / len(date_coverage)
                    method_avg = sum(c.method_coverage_pct for c in date_coverage) / len(date_coverage)
                else:
                    endpoint_avg = avg_endpoint_coverage
                    method_avg = avg_method_coverage
                
                coverage_trend.append({
                    "date": date.isoformat(),
                    "endpoint_coverage": endpoint_avg,
                    "method_coverage": method_avg
                })
            
            coverage_trend.reverse()  # Show oldest to newest
        else:
            avg_endpoint_coverage = 0
            avg_method_coverage = 0
            coverage_trend = []
        
        coverage = {
            "average_endpoint_coverage": avg_endpoint_coverage,
            "average_method_coverage": avg_method_coverage,
            "coverage_trend": coverage_trend
        }
        
        # RL optimization metrics
        total_rl_models = db.query(RLModel).count()
        active_rl_models = db.query(RLModel).filter(RLModel.is_active == True).count()
        
        # Calculate average optimization score
        avg_optimization_score = 0
        if active_rl_models > 0:
            # Simulate optimization score based on coverage and success rates
            avg_optimization_score = min((avg_endpoint_coverage + avg_session_success_rate) / 2, 100)
        
        # Calculate performance improvement (simplified)
        performance_improvement = max(0, avg_optimization_score - 70)  # Assume baseline of 70%
        
        rl_optimization = {
            "models_trained": total_rl_models,
            "optimization_score": avg_optimization_score,
            "performance_improvement": performance_improvement
        }
        
        # Healing metrics (simplified for MVP)
        healing = {
            "total_healing_attempts": 0,
            "successful_healings": 0,
            "healing_success_rate": 0.0
        }
        
        return {
            "system_status": system_status,
            "api_specifications": api_specifications,
            "test_cases": test_cases,
            "executions": executions,
            "coverage": coverage,
            "rl_optimization": rl_optimization,
            "healing": healing,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard metrics: {str(e)}"
        )

@router.get("/dashboard/recent-activity")
async def get_recent_activity(
    limit: int = 20,
    activity_types: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """
    Get recent activity across the system.
    """
    try:
        activities = []
        
        # Recent API spec uploads
        recent_specs = db.query(APISpecification).order_by(
            APISpecification.created_at.desc()
        ).limit(5).all()
        
        for spec in recent_specs:
            activities.append({
                "id": f"spec_upload_{spec.id}",
                "type": "api_spec_upload",
                "title": f"API Specification Uploaded",
                "description": f"'{spec.name}' v{spec.version} was uploaded",
                "icon": "upload",
                "timestamp": spec.created_at.isoformat(),
                "metadata": {
                    "api_spec_id": spec.id,
                    "spec_name": spec.name,
                    "spec_version": spec.version
                }
            })
        
        # Recent test generations
        recent_generations = db.query(AIGenerationLog).order_by(
            AIGenerationLog.created_at.desc()
        ).limit(5).all()
        
        for gen in recent_generations:
            activities.append({
                "id": f"test_generation_{gen.id}",
                "type": "test_generation",
                "title": "AI Test Cases Generated",
                "description": f"Generated using {gen.ai_model}",
                "icon": "cog",
                "timestamp": gen.created_at.isoformat(),
                "metadata": {
                    "generation_id": gen.id,
                    "ai_model": gen.ai_model,
                    "validation_passed": gen.validation_passed
                }
            })
        
        # Recent executions
        recent_sessions = db.query(ExecutionSession).order_by(
            ExecutionSession.created_at.desc()
        ).limit(5).all()
        
        for session in recent_sessions:
            success_rate = (session.passed_tests / session.total_tests * 100) if session.total_tests > 0 else 0
            
            activities.append({
                "id": f"execution_{session.id}",
                "type": "test_execution",
                "title": "Test Execution Completed",
                "description": f"{session.passed_tests}/{session.total_tests} tests passed ({success_rate:.1f}%)",
                "icon": "play",
                "timestamp": session.created_at.isoformat(),
                "metadata": {
                    "session_id": session.id,
                    "api_spec_id": session.api_spec_id,
                    "success_rate": success_rate,
                    "total_tests": session.total_tests
                }
            })
        
        # Recent RL optimizations
        recent_rl_models = db.query(RLModel).order_by(
            RLModel.updated_at.desc()
        ).limit(3).all()
        
        for model in recent_rl_models:
            activities.append({
                "id": f"rl_optimization_{model.id}",
                "type": "rl_optimization",
                "title": "RL Model Training",
                "description": f"{model.algorithm.value} model trained with {model.episodes_trained} episodes",
                "icon": "chart-bar",
                "timestamp": model.updated_at.isoformat(),
                "metadata": {
                    "model_id": model.id,
                    "algorithm": model.algorithm.value,
                    "episodes_trained": model.episodes_trained
                }
            })
        
        # Sort all activities by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "activities": activities[:limit],
            "total_activities": len(activities),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent activity: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent activity: {str(e)}"
        )

@router.get("/dashboard/system-health")
async def get_system_health(db: Session = Depends(get_db)):
    """
    Get detailed system health information.
    """
    try:
        # Database health
        try:
            db.execute("SELECT 1")
            db_health = "healthy"
        except Exception:
            db_health = "unhealthy"
        
        # Service health checks
        services_health = {
            "database": db_health,
            "ai_engine": "healthy",  # Simplified for MVP
            "rl_optimizer": "healthy",
            "test_executor": "healthy",
            "healing_system": "healthy"
        }
        
        # Performance metrics
        total_test_cases = db.query(TestCase).count()
        total_executions = db.query(TestExecution).count()
        
        # Recent performance
        recent_executions = db.query(TestExecution).filter(
            TestExecution.completed_at >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        if recent_executions:
            avg_response_time = sum(
                ex.response_time_ms for ex in recent_executions if ex.response_time_ms
            ) / len([ex for ex in recent_executions if ex.response_time_ms])
            
            recent_success_rate = len([
                ex for ex in recent_executions if ex.status == TestStatus.PASSED
            ]) / len(recent_executions) * 100
        else:
            avg_response_time = 0
            recent_success_rate = 0
        
        performance_metrics = {
            "average_response_time_ms": avg_response_time,
            "recent_success_rate": recent_success_rate,
            "total_test_cases": total_test_cases,
            "total_executions": total_executions
        }
        
        # Overall health score
        healthy_services = sum(1 for status in services_health.values() if status == "healthy")
        health_score = (healthy_services / len(services_health)) * 100
        
        return {
            "overall_health": "healthy" if health_score >= 80 else "degraded",
            "health_score": health_score,
            "services": services_health,
            "performance": performance_metrics,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system health: {str(e)}"
        )

@router.get("/dashboard/quick-stats")
async def get_quick_stats(db: Session = Depends(get_db)):
    """
    Get quick statistics for dashboard widgets.
    """
    try:
        # Quick counts
        stats = {
            "api_specs": db.query(APISpecification).filter(APISpecification.is_active == True).count(),
            "test_cases": db.query(TestCase).filter(TestCase.is_active == True).count(),
            "sessions_today": db.query(ExecutionSession).filter(
                ExecutionSession.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            ).count(),
            "ai_generations": db.query(AIGenerationLog).count(),
            "rl_models": db.query(RLModel).filter(RLModel.is_active == True).count()
        }
        
        # Recent trends (7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        trends = {
            "new_specs_week": db.query(APISpecification).filter(
                APISpecification.created_at >= week_ago
            ).count(),
            "new_tests_week": db.query(TestCase).filter(
                TestCase.created_at >= week_ago
            ).count(),
            "executions_week": db.query(ExecutionSession).filter(
                ExecutionSession.created_at >= week_ago
            ).count()
        }
        
        return {
            "stats": stats,
            "trends": trends,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get quick stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quick stats: {str(e)}"
        )

@router.get("/dashboard/performance-overview")
async def get_performance_overview(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get performance overview with trends and metrics.
    """
    try:
        # Get execution sessions from specified period
        start_date = datetime.utcnow() - timedelta(days=days)
        sessions = db.query(ExecutionSession).filter(
            ExecutionSession.created_at >= start_date
        ).all()
        
        if not sessions:
            return {
                "performance_data": [],
                "summary": {
                    "total_sessions": 0,
                    "average_success_rate": 0,
                    "average_execution_time": 0,
                    "total_tests_executed": 0
                },
                "period_days": days
            }
        
        # Daily performance data
        daily_data = {}
        for session in sessions:
            date_key = session.created_at.date().isoformat()
            
            if date_key not in daily_data:
                daily_data[date_key] = {
                    "date": date_key,
                    "sessions": 0,
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "execution_time": 0
                }
            
            daily_data[date_key]["sessions"] += 1
            daily_data[date_key]["total_tests"] += session.total_tests or 0
            daily_data[date_key]["passed_tests"] += session.passed_tests or 0
            daily_data[date_key]["failed_tests"] += session.failed_tests or 0
            daily_data[date_key]["execution_time"] += session.duration_seconds or 0
        
        # Calculate daily success rates
        performance_data = []
        for date_key, data in daily_data.items():
            success_rate = (data["passed_tests"] / data["total_tests"] * 100) if data["total_tests"] > 0 else 0
            avg_execution_time = data["execution_time"] / data["sessions"] if data["sessions"] > 0 else 0
            
            performance_data.append({
                "date": date_key,
                "success_rate": success_rate,
                "total_tests": data["total_tests"],
                "sessions": data["sessions"],
                "avg_execution_time": avg_execution_time
            })
        
        performance_data.sort(key=lambda x: x["date"])
        
        # Summary statistics
        total_sessions = len(sessions)
        total_tests = sum(s.total_tests for s in sessions if s.total_tests)
        total_passed = sum(s.passed_tests for s in sessions if s.passed_tests)
        avg_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        avg_execution_time = sum(s.duration_seconds for s in sessions if s.duration_seconds) / total_sessions if total_sessions > 0 else 0
        
        summary = {
            "total_sessions": total_sessions,
            "average_success_rate": avg_success_rate,
            "average_execution_time": avg_execution_time,
            "total_tests_executed": total_tests
        }
        
        return {
            "performance_data": performance_data,
            "summary": summary,
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance overview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance overview: {str(e)}"
        )
