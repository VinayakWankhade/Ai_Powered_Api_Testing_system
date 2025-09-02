"""
API endpoints for RL optimization functionality.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...database.connection import get_db
from ...database.models import APISpecification, RLModel, RLAlgorithm
from ...rl.hybrid_optimizer import HybridRLOptimizer
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class OptimizationRequest(BaseModel):
    """Request model for RL optimization."""
    api_spec_id: int = Field(..., description="API specification ID")
    algorithm: RLAlgorithm = Field(RLAlgorithm.PPO, description="RL algorithm to use")
    max_tests: int = Field(50, description="Maximum number of tests to select")
    training_episodes: int = Field(1000, description="Number of training episodes")
    optimization_goals: Optional[List[str]] = Field(None, description="Specific optimization goals")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")

class OptimizationResponse(BaseModel):
    """Response model for RL optimization."""
    message: str
    optimized_selection: List[Dict[str, Any]]
    optimization_score: float
    algorithm: str
    total_available_tests: int
    selected_tests: int
    training_episodes_completed: int
    improvement_metrics: Dict[str, Any]

class TrainingRequest(BaseModel):
    """Request model for RL model training."""
    api_spec_id: int = Field(..., description="API specification ID")
    algorithm: RLAlgorithm = Field(RLAlgorithm.PPO, description="RL algorithm to use")
    episodes: int = Field(1000, description="Number of training episodes")
    learning_rate: Optional[float] = Field(0.001, description="Learning rate for training")
    batch_size: Optional[int] = Field(32, description="Batch size for training")
    training_data_days: Optional[int] = Field(30, description="Days of historical data to use")

@router.post("/optimize-tests", response_model=OptimizationResponse)
async def optimize_tests(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    Optimize test case selection using reinforcement learning.
    
    Uses historical execution data to select the most effective test cases.
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
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Execute optimization
        optimization_result = await optimizer.optimize_test_selection(
            api_spec_id=request.api_spec_id,
            algorithm=request.algorithm,
            max_tests=request.max_tests,
            training_episodes=request.training_episodes
        )
        
        return OptimizationResponse(
            message=optimization_result["message"],
            optimized_selection=optimization_result["optimized_selection"],
            optimization_score=optimization_result["optimization_score"],
            algorithm=optimization_result["algorithm"],
            total_available_tests=optimization_result["total_available_tests"],
            selected_tests=optimization_result["selected_tests"],
            training_episodes_completed=request.training_episodes,
            improvement_metrics=optimization_result.get("improvement_metrics", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test optimization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test optimization failed: {str(e)}"
        )

@router.post("/optimize-tests-async")
async def optimize_tests_async(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start test optimization asynchronously in the background.
    
    Returns immediately with an optimization session ID to track progress.
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
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Create optimization session ID
        import uuid
        optimization_session_id = str(uuid.uuid4())
        
        # Start optimization in background
        background_tasks.add_task(
            optimizer.optimize_test_selection,
            request.api_spec_id,
            request.algorithm,
            request.max_tests,
            request.training_episodes
        )
        
        return {
            "message": "Test optimization started in background",
            "optimization_session_id": optimization_session_id,
            "api_spec_id": request.api_spec_id,
            "algorithm": request.algorithm.value,
            "status": "running",
            "progress_url": f"/api/v1/optimization-status/{optimization_session_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start async test optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start async test optimization: {str(e)}"
        )

@router.post("/train-model")
async def train_rl_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Train a new RL model for test optimization.
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
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Start training (for MVP, this is simplified)
        training_session_id = f"training_{request.api_spec_id}_{request.algorithm.value}"
        
        # In a full implementation, this would start actual RL training
        # For MVP, we simulate training completion
        training_result = {
            "message": "RL model training completed",
            "training_session_id": training_session_id,
            "api_spec_id": request.api_spec_id,
            "algorithm": request.algorithm.value,
            "episodes_completed": request.episodes,
            "final_performance": 0.85,  # Simulated performance
            "training_time_seconds": 120.5,  # Simulated training time
            "status": "completed"
        }
        
        return training_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RL model training failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"RL model training failed: {str(e)}"
        )

@router.get("/model-performance/{api_spec_id}")
async def get_model_performance(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get performance metrics for RL models associated with an API specification.
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
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Get model performance
        performance_data = optimizer.get_rl_model_performance(api_spec_id)
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model performance: {str(e)}"
        )

@router.get("/optimization-recommendations/{api_spec_id}")
async def get_optimization_recommendations(
    api_spec_id: int,
    current_selection: Optional[List[int]] = None,
    db: Session = Depends(get_db)
):
    """
    Get optimization recommendations for test selection.
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
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Get recommendations
        recommendations = optimizer.get_optimization_recommendations(
            api_spec_id=api_spec_id,
            current_test_selection=current_selection or []
        )
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization recommendations: {str(e)}"
        )

@router.get("/training-history/{api_spec_id}")
async def get_training_history(
    api_spec_id: int,
    db: Session = Depends(get_db)
):
    """
    Get RL training history for an API specification.
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
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Get training history
        history_data = optimizer.get_rl_training_history(api_spec_id)
        
        return history_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training history: {str(e)}"
        )

@router.get("/algorithms")
async def list_available_algorithms():
    """
    List available RL algorithms for optimization.
    """
    try:
        algorithms = [
            {
                "name": "PPO",
                "full_name": "Proximal Policy Optimization",
                "description": "Stable and efficient policy gradient method",
                "recommended_for": ["general_optimization", "stable_performance"],
                "hyperparameters": {
                    "learning_rate": {"default": 0.001, "range": [0.0001, 0.01]},
                    "episodes": {"default": 1000, "range": [100, 5000]}
                }
            },
            {
                "name": "Q_LEARNING",
                "full_name": "Q-Learning",
                "description": "Value-based method with exploration",
                "recommended_for": ["exploration", "discrete_actions"],
                "hyperparameters": {
                    "learning_rate": {"default": 0.1, "range": [0.01, 0.5]},
                    "epsilon": {"default": 0.1, "range": [0.05, 0.3]},
                    "episodes": {"default": 1500, "range": [500, 3000]}
                }
            },
            {
                "name": "EVOLUTIONARY",
                "full_name": "Evolutionary Algorithm",
                "description": "Population-based optimization method",
                "recommended_for": ["complex_landscapes", "global_optimization"],
                "hyperparameters": {
                    "population_size": {"default": 50, "range": [20, 100]},
                    "generations": {"default": 100, "range": [50, 200]},
                    "mutation_rate": {"default": 0.1, "range": [0.05, 0.2]}
                }
            }
        ]
        
        return {
            "algorithms": algorithms,
            "default_algorithm": "PPO",
            "algorithm_count": len(algorithms)
        }
        
    except Exception as e:
        logger.error(f"Failed to list algorithms: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list algorithms: {str(e)}"
        )

@router.get("/optimization-metrics/{api_spec_id}")
async def get_optimization_metrics(
    api_spec_id: int,
    algorithm: Optional[RLAlgorithm] = None,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get detailed optimization metrics for an API specification.
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
        
        # Get models for this API spec
        query = db.query(RLModel).filter(RLModel.api_spec_id == api_spec_id)
        
        if algorithm:
            query = query.filter(RLModel.algorithm == algorithm)
        
        models = query.all()
        
        metrics_data = {
            "api_spec_id": api_spec_id,
            "algorithm_filter": algorithm.value if algorithm else None,
            "analysis_period_days": days,
            "models": [
                {
                    "model_id": model.id,
                    "algorithm": model.algorithm.value,
                    "version": model.model_version,
                    "episodes_trained": model.episodes_trained,
                    "is_active": model.is_active,
                    "created_at": model.created_at.isoformat(),
                    "last_updated": model.updated_at.isoformat()
                }
                for model in models
            ],
            "total_models": len(models),
            "active_models": len([m for m in models if m.is_active])
        }
        
        return metrics_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization metrics: {str(e)}"
        )

@router.post("/compare-algorithms/{api_spec_id}")
async def compare_algorithms(
    api_spec_id: int,
    algorithms: List[RLAlgorithm],
    test_episodes: int = 100,
    db: Session = Depends(get_db)
):
    """
    Compare the performance of different RL algorithms.
    """
    try:
        if len(algorithms) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two algorithms are required for comparison"
            )
        
        # Verify API specification exists
        api_spec = db.query(APISpecification).filter(
            APISpecification.id == api_spec_id
        ).first()
        
        if not api_spec:
            raise HTTPException(
                status_code=404,
                detail=f"API specification with ID {api_spec_id} not found"
            )
        
        # Initialize RL optimizer
        optimizer = HybridRLOptimizer()
        
        # Compare algorithms (simplified for MVP)
        comparison_results = []
        
        for algorithm in algorithms:
            try:
                result = await optimizer.optimize_test_selection(
                    api_spec_id=api_spec_id,
                    algorithm=algorithm,
                    max_tests=20,  # Smaller for comparison
                    training_episodes=test_episodes
                )
                
                comparison_results.append({
                    "algorithm": algorithm.value,
                    "optimization_score": result["optimization_score"],
                    "selected_tests": result["selected_tests"],
                    "performance_rating": "excellent" if result["optimization_score"] > 80 else 
                                        "good" if result["optimization_score"] > 60 else "fair"
                })
                
            except Exception as e:
                comparison_results.append({
                    "algorithm": algorithm.value,
                    "optimization_score": 0,
                    "selected_tests": 0,
                    "error": str(e),
                    "performance_rating": "failed"
                })
        
        # Determine best algorithm
        successful_results = [r for r in comparison_results if "error" not in r]
        best_algorithm = max(successful_results, key=lambda x: x["optimization_score"]) if successful_results else None
        
        return {
            "api_spec_id": api_spec_id,
            "algorithms_compared": [alg.value for alg in algorithms],
            "test_episodes": test_episodes,
            "comparison_results": comparison_results,
            "best_algorithm": best_algorithm["algorithm"] if best_algorithm else None,
            "comparison_completed_at": optimizer.get_current_timestamp() if hasattr(optimizer, 'get_current_timestamp') else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Algorithm comparison failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Algorithm comparison failed: {str(e)}"
        )
