"""API endpoints package."""

# Import all endpoint modules
from . import specs, test_generation

# Create placeholder modules for remaining endpoints
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List

# Test Execution Router
test_execution = APIRouter()

class TestExecutionRequest(BaseModel):
    api_spec_id: int
    test_case_ids: Optional[List[int]] = None
    session_name: Optional[str] = None
    trigger: str = "manual"

@test_execution.post("/run-tests")
async def run_tests(request: TestExecutionRequest):
    """Execute test cases."""
    from ...execution.test_executor import APITestExecutor
    executor = APITestExecutor()
    
    session = await executor.execute_test_session(
        api_spec_id=request.api_spec_id,
        test_case_ids=request.test_case_ids,
        session_name=request.session_name,
        trigger=request.trigger
    )
    
    return {
        "session_id": session.id,
        "status": "completed",
        "results": {
            "total_tests": session.total_tests,
            "passed": session.passed_tests,
            "failed": session.failed_tests,
            "errors": session.error_tests
        }
    }

@test_execution.get("/execution-history")
async def get_execution_history(api_spec_id: Optional[int] = None, limit: int = 50):
    """Get execution history."""
    from ...execution.test_executor import APITestExecutor
    executor = APITestExecutor()
    
    history = executor.get_execution_history(api_spec_id, limit)
    
    return {
        "sessions": [
            {
                "id": session.id,
                "name": session.name,
                "trigger": session.trigger,
                "total_tests": session.total_tests,
                "passed_tests": session.passed_tests,
                "failed_tests": session.failed_tests,
                "duration_seconds": session.duration_seconds,
                "created_at": session.created_at.isoformat()
            }
            for session in history
        ]
    }

# Test Healing Router
test_healing = APIRouter()

class TestHealingRequest(BaseModel):
    session_id: int
    max_healing_attempts: int = 3
    auto_revalidate: bool = True

@test_healing.post("/heal-tests")
async def heal_tests(request: TestHealingRequest):
    """Heal failed tests."""
    from ...healing.test_healer import AITestHealer
    healer = AITestHealer()
    
    results = await healer.heal_failed_tests(
        session_id=request.session_id,
        max_healing_attempts=request.max_healing_attempts,
        auto_revalidate=request.auto_revalidate
    )
    
    return results

@test_healing.get("/healing-statistics")
async def get_healing_statistics(api_spec_id: Optional[int] = None):
    """Get healing statistics."""
    from ...healing.test_healer import AITestHealer
    healer = AITestHealer()
    
    return healer.get_healing_statistics(api_spec_id)

# Coverage Router
coverage = APIRouter()

@coverage.get("/coverage-report/{api_spec_id}")
async def get_coverage_report(api_spec_id: int):
    """Get coverage report."""
    from ...database.connection import get_db_session
    from ...database.models import CoverageMetrics, ExecutionSession
    
    db = get_db_session()
    
    latest_session = db.query(ExecutionSession).filter(
        ExecutionSession.api_spec_id == api_spec_id
    ).order_by(ExecutionSession.created_at.desc()).first()
    
    if not latest_session:
        return {"message": "No execution sessions found"}
    
    coverage = db.query(CoverageMetrics).filter(
        CoverageMetrics.session_id == latest_session.id
    ).first()
    
    db.close()
    
    if not coverage:
        return {"message": "No coverage data found"}
    
    return {
        "api_spec_id": api_spec_id,
        "session_id": latest_session.id,
        "coverage": {
            "endpoint_coverage_pct": coverage.endpoint_coverage_pct,
            "method_coverage_pct": coverage.method_coverage_pct,
            "response_code_coverage_pct": coverage.response_code_coverage_pct,
            "covered_endpoints": coverage.covered_endpoints,
            "missed_endpoints": coverage.missed_endpoints,
            "bugs_found": coverage.bugs_found,
            "quality_score": coverage.quality_score
        },
        "measured_at": coverage.measured_at.isoformat()
    }

# RL Optimization Router
rl_optimization = APIRouter()

class RLOptimizationRequest(BaseModel):
    api_spec_id: int
    algorithm: str = "ppo"
    training_episodes: int = 1000
    max_tests: int = 50

@rl_optimization.post("/optimize-tests")
async def optimize_tests(request: RLOptimizationRequest):
    """Optimize test selection using RL."""
    from ...rl.hybrid_optimizer import HybridRLOptimizer
    from ...database.models import RLAlgorithm
    
    optimizer = HybridRLOptimizer()
    
    # Convert string to enum
    algorithm_map = {
        "ppo": RLAlgorithm.PPO,
        "q_learning": RLAlgorithm.Q_LEARNING,
        "evolutionary": RLAlgorithm.EVOLUTIONARY
    }
    
    algorithm = algorithm_map.get(request.algorithm, RLAlgorithm.PPO)
    
    results = await optimizer.optimize_test_selection(
        api_spec_id=request.api_spec_id,
        algorithm=algorithm,
        max_tests=request.max_tests,
        training_episodes=request.training_episodes
    )
    
    return results

@rl_optimization.get("/rl-performance/{api_spec_id}")
async def get_rl_performance(api_spec_id: int):
    """Get RL model performance."""
    from ...rl.hybrid_optimizer import HybridRLOptimizer
    optimizer = HybridRLOptimizer()
    
    return optimizer.get_rl_model_performance(api_spec_id)

@rl_optimization.get("/recommendations/{api_spec_id}")
async def get_recommendations(api_spec_id: int, current_selection: Optional[List[int]] = None):
    """Get optimization recommendations."""
    from ...rl.hybrid_optimizer import HybridRLOptimizer
    optimizer = HybridRLOptimizer()
    
    return optimizer.get_optimization_recommendations(
        api_spec_id=api_spec_id,
        current_test_selection=current_selection or []
    )
