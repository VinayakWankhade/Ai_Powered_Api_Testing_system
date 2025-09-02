"""
Simplified RL optimization system for MVP.
Focuses on basic test selection optimization without complex RL algorithms.
"""

import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..database.connection import get_db_session
from ..database.models import (
    TestCase, RLModel, RLAlgorithm, ExecutionSession, TestExecution, TestStatus
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

class HybridRLOptimizer:
    """Simplified RL optimizer for test case selection."""
    
    def __init__(self):
        self.db = get_db_session()
    
    async def optimize_test_selection(
        self,
        api_spec_id: int,
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        max_tests: int = 50,
        training_episodes: int = 1000
    ) -> Dict[str, Any]:
        """Optimize test case selection using simplified RL approach."""
        
        try:
            # Get all test cases for the API specification
            test_cases = self.db.query(TestCase).filter(
                TestCase.api_spec_id == api_spec_id,
                TestCase.is_active == True
            ).all()
            
            if not test_cases:
                return {
                    "message": "No test cases found for optimization",
                    "optimized_selection": [],
                    "optimization_score": 0,
                    "algorithm": algorithm.value
                }
            
            # Calculate scores for each test case based on historical performance
            test_scores = {}
            for test_case in test_cases:
                score = self._calculate_test_case_score(test_case)
                test_scores[test_case.id] = {
                    "test_case": test_case,
                    "score": score,
                    "selection_count": test_case.selection_count,
                    "success_rate": test_case.success_rate
                }
            
            # Select best test cases based on algorithm
            if algorithm == RLAlgorithm.Q_LEARNING:
                optimized_selection = self._q_learning_selection(test_scores, max_tests)
            elif algorithm == RLAlgorithm.EVOLUTIONARY:
                optimized_selection = self._evolutionary_selection(test_scores, max_tests)
            else:  # PPO or default
                optimized_selection = self._ppo_selection(test_scores, max_tests)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(optimized_selection, test_scores)
            
            # Save RL model state
            await self._save_rl_model_state(api_spec_id, algorithm, test_scores, optimized_selection)
            
            result = {
                "message": f"Optimized test selection using {algorithm.value}",
                "optimized_selection": [
                    {
                        "test_case_id": tc_id,
                        "test_name": test_scores[tc_id]["test_case"].name,
                        "endpoint": test_scores[tc_id]["test_case"].endpoint,
                        "method": test_scores[tc_id]["test_case"].method,
                        "score": test_scores[tc_id]["score"],
                        "success_rate": test_scores[tc_id]["success_rate"]
                    }
                    for tc_id in optimized_selection
                ],
                "optimization_score": optimization_score,
                "algorithm": algorithm.value,
                "total_available_tests": len(test_cases),
                "selected_tests": len(optimized_selection)
            }
            
            logger.info(f"RL optimization completed: selected {len(optimized_selection)}/{len(test_cases)} tests with score {optimization_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"RL optimization failed: {str(e)}")
            raise ValueError(f"Optimization failed: {str(e)}")
    
    def _calculate_test_case_score(self, test_case: TestCase) -> float:
        """Calculate a score for test case selection based on multiple factors."""
        
        try:
            # Base score from success rate (higher success rate = better test)
            success_score = test_case.success_rate * 0.3
            
            # Frequency score (less frequently selected tests get higher score for diversity)
            max_selection_count = 100  # Assume max selections for normalization
            frequency_score = (1 - min(test_case.selection_count / max_selection_count, 1)) * 0.2
            
            # Test type score (edge cases and security tests get higher priority)
            type_scores = {
                "functional": 0.5,
                "edge_case": 0.8,
                "security": 0.9,
                "performance": 0.7,
                "generated": 0.6
            }
            type_score = type_scores.get(test_case.test_type.value, 0.5) * 0.3
            
            # Coverage contribution score (tests covering untested areas get higher score)
            coverage_score = self._calculate_coverage_contribution_score(test_case) * 0.2
            
            total_score = success_score + frequency_score + type_score + coverage_score
            
            # Add some randomness to avoid always selecting the same tests
            randomness = random.uniform(0, 0.1)
            
            return min(total_score + randomness, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate test case score: {str(e)}")
            return 0.5  # Default score
    
    def _calculate_coverage_contribution_score(self, test_case: TestCase) -> float:
        """Calculate how much this test case contributes to coverage."""
        
        try:
            # For MVP, use simple heuristics
            # Tests for less common HTTP methods get higher scores
            method_scores = {
                "GET": 0.3,
                "POST": 0.5,
                "PUT": 0.7,
                "DELETE": 0.8,
                "PATCH": 0.9,
                "HEAD": 0.6,
                "OPTIONS": 0.4
            }
            
            return method_scores.get(test_case.method, 0.5)
            
        except Exception as e:
            logger.error(f"Failed to calculate coverage score: {str(e)}")
            return 0.5
    
    def _q_learning_selection(self, test_scores: Dict[int, Dict], max_tests: int) -> List[int]:
        """Q-learning inspired test selection."""
        
        # Sort by score with exploration factor
        scored_tests = []
        for test_id, test_data in test_scores.items():
            # Q-learning inspired scoring with exploration
            q_value = test_data["score"]
            exploration_bonus = 1 / (1 + test_data["selection_count"])  # Explore less-selected tests
            final_score = q_value + 0.1 * exploration_bonus
            
            scored_tests.append((test_id, final_score))
        
        scored_tests.sort(key=lambda x: x[1], reverse=True)
        return [test_id for test_id, _ in scored_tests[:max_tests]]
    
    def _evolutionary_selection(self, test_scores: Dict[int, Dict], max_tests: int) -> List[int]:
        """Evolutionary algorithm inspired test selection."""
        
        # Create population of test combinations
        population_size = 10
        generations = 5
        
        all_test_ids = list(test_scores.keys())
        
        # Initialize population with random selections
        population = []
        for _ in range(population_size):
            selection = random.sample(all_test_ids, min(max_tests, len(all_test_ids)))
            population.append(selection)
        
        # Evolve the population
        for generation in range(generations):
            # Evaluate fitness of each individual
            fitness_scores = []
            for individual in population:
                fitness = sum(test_scores[test_id]["score"] for test_id in individual) / len(individual)
                fitness_scores.append((individual, fitness))
            
            # Select best individuals
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            survivors = [individual for individual, _ in fitness_scores[:population_size//2]]
            
            # Create new generation through crossover and mutation
            new_population = survivors.copy()
            
            while len(new_population) < population_size:
                # Crossover
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Simple crossover: take half from each parent
                child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
                
                # Mutation: replace some tests randomly
                if random.random() < 0.1:  # 10% mutation rate
                    mutation_count = max(1, len(child) // 10)
                    for _ in range(mutation_count):
                        if child:  # Only mutate if child is not empty
                            idx = random.randint(0, len(child) - 1)
                            child[idx] = random.choice(all_test_ids)
                
                # Remove duplicates and maintain size
                child = list(set(child))[:max_tests]
                new_population.append(child)
            
            population = new_population
        
        # Return the best individual from final generation
        final_fitness = [(ind, sum(test_scores[tid]["score"] for tid in ind) / len(ind)) for ind in population]
        best_individual = max(final_fitness, key=lambda x: x[1])[0]
        
        return best_individual[:max_tests]
    
    def _ppo_selection(self, test_scores: Dict[int, Dict], max_tests: int) -> List[int]:
        """PPO-inspired test selection (simplified)."""
        
        # PPO-like approach: balance between exploitation and exploration
        scored_tests = []
        
        for test_id, test_data in test_scores.items():
            # Calculate advantage (how much better this test is than average)
            avg_score = sum(td["score"] for td in test_scores.values()) / len(test_scores)
            advantage = test_data["score"] - avg_score
            
            # PPO-like clipped advantage
            clipped_advantage = max(-0.2, min(0.2, advantage))
            
            # Final score combines original score with clipped advantage
            final_score = test_data["score"] + 0.1 * clipped_advantage
            
            scored_tests.append((test_id, final_score))
        
        # Select top scoring tests
        scored_tests.sort(key=lambda x: x[1], reverse=True)
        return [test_id for test_id, _ in scored_tests[:max_tests]]
    
    def _calculate_optimization_score(
        self,
        selected_test_ids: List[int],
        test_scores: Dict[int, Dict]
    ) -> float:
        """Calculate overall optimization score for the selection."""
        
        if not selected_test_ids:
            return 0.0
        
        # Calculate average score of selected tests
        total_score = sum(test_scores[test_id]["score"] for test_id in selected_test_ids)
        avg_score = total_score / len(selected_test_ids)
        
        # Bonus for diversity (different endpoints and methods)
        endpoints_covered = len(set(test_scores[tid]["test_case"].endpoint for tid in selected_test_ids))
        methods_covered = len(set(test_scores[tid]["test_case"].method for tid in selected_test_ids))
        
        diversity_bonus = (endpoints_covered + methods_covered) / (len(selected_test_ids) + 1)
        
        # Final optimization score
        optimization_score = (avg_score * 0.7 + diversity_bonus * 0.3) * 100
        
        return min(optimization_score, 100.0)
    
    async def _save_rl_model_state(
        self,
        api_spec_id: int,
        algorithm: RLAlgorithm,
        test_scores: Dict[int, Dict],
        optimized_selection: List[int]
    ):
        """Save the RL model state for future reference."""
        
        try:
            # Check if model already exists
            existing_model = self.db.query(RLModel).filter(
                RLModel.api_spec_id == api_spec_id,
                RLModel.algorithm == algorithm,
                RLModel.is_active == True
            ).first()
            
            model_state = {
                "test_scores": {str(k): v["score"] for k, v in test_scores.items()},
                "last_selection": optimized_selection,
                "selection_timestamp": datetime.utcnow().isoformat()
            }
            
            if existing_model:
                # Update existing model
                existing_model.model_state = model_state
                existing_model.episodes_trained += 1
                existing_model.updated_at = datetime.utcnow()
            else:
                # Create new model
                new_model = RLModel(
                    api_spec_id=api_spec_id,
                    algorithm=algorithm,
                    model_version="1.0",
                    description=f"Simplified {algorithm.value} model for test selection",
                    model_state=model_state,
                    episodes_trained=1,
                    is_trained=True
                )
                self.db.add(new_model)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to save RL model state: {str(e)}")
    
    def get_rl_model_performance(self, api_spec_id: int) -> Dict[str, Any]:
        """Get RL model performance metrics."""
        
        try:
            models = self.db.query(RLModel).filter(
                RLModel.api_spec_id == api_spec_id,
                RLModel.is_active == True
            ).all()
            
            if not models:
                return {
                    "message": "No RL models found for this API specification",
                    "models": []
                }
            
            model_performance = []
            
            for model in models:
                # Calculate basic performance metrics
                test_cases = self.db.query(TestCase).filter(
                    TestCase.api_spec_id == api_spec_id
                ).all()
                
                avg_success_rate = sum(tc.success_rate for tc in test_cases) / len(test_cases) if test_cases else 0
                
                model_performance.append({
                    "model_id": model.id,
                    "algorithm": model.algorithm.value,
                    "version": model.model_version,
                    "episodes_trained": model.episodes_trained,
                    "average_success_rate": avg_success_rate,
                    "is_active": model.is_active,
                    "created_at": model.created_at.isoformat(),
                    "updated_at": model.updated_at.isoformat()
                })
            
            return {
                "models": model_performance,
                "api_spec_id": api_spec_id,
                "total_models": len(models)
            }
            
        except Exception as e:
            logger.error(f"Failed to get RL model performance: {str(e)}")
            return {
                "models": [],
                "error": str(e)
            }
    
    def get_optimization_recommendations(
        self,
        api_spec_id: int,
        current_test_selection: List[int]
    ) -> Dict[str, Any]:
        """Get optimization recommendations for test selection."""
        
        try:
            # Get all test cases
            test_cases = self.db.query(TestCase).filter(
                TestCase.api_spec_id == api_spec_id,
                TestCase.is_active == True
            ).all()
            
            recommendations = []
            
            # Analyze current selection
            if current_test_selection:
                selected_tests = [tc for tc in test_cases if tc.id in current_test_selection]
                
                # Check for coverage gaps
                all_endpoints = set(tc.endpoint for tc in test_cases)
                selected_endpoints = set(tc.endpoint for tc in selected_tests)
                missing_endpoints = all_endpoints - selected_endpoints
                
                if missing_endpoints:
                    recommendations.append({
                        "type": "coverage_gap",
                        "message": f"Consider adding tests for uncovered endpoints: {list(missing_endpoints)}",
                        "priority": "high"
                    })
                
                # Check test type diversity
                selected_types = set(tc.test_type.value for tc in selected_tests)
                if len(selected_types) < 3:
                    recommendations.append({
                        "type": "test_diversity",
                        "message": "Consider adding more diverse test types (functional, edge_case, security)",
                        "priority": "medium"
                    })
                
                # Check for low-performing tests
                low_performers = [tc for tc in selected_tests if tc.success_rate < 0.5]
                if low_performers:
                    recommendations.append({
                        "type": "low_performers",
                        "message": f"Consider reviewing {len(low_performers)} low-performing tests",
                        "priority": "medium"
                    })
            
            # General recommendations
            if not recommendations:
                recommendations.append({
                    "type": "general",
                    "message": "Current test selection appears well-optimized",
                    "priority": "info"
                })
            
            return {
                "recommendations": recommendations,
                "current_selection_analysis": {
                    "total_selected": len(current_test_selection),
                    "avg_success_rate": sum(tc.success_rate for tc in test_cases if tc.id in current_test_selection) / len(current_test_selection) if current_test_selection else 0,
                    "endpoints_covered": len(set(tc.endpoint for tc in test_cases if tc.id in current_test_selection)),
                    "test_types_covered": len(set(tc.test_type.value for tc in test_cases if tc.id in current_test_selection))
                },
                "api_spec_id": api_spec_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {str(e)}")
            return {
                "recommendations": [{
                    "type": "error",
                    "message": f"Failed to generate recommendations: {str(e)}",
                    "priority": "high"
                }],
                "error": str(e)
            }
    
    def get_rl_training_history(self, api_spec_id: int) -> Dict[str, Any]:
        """Get RL training history and metrics."""
        
        try:
            models = self.db.query(RLModel).filter(
                RLModel.api_spec_id == api_spec_id
            ).all()
            
            training_history = []
            
            for model in models:
                # Get execution sessions to calculate performance over time
                sessions = self.db.query(ExecutionSession).filter(
                    ExecutionSession.api_spec_id == api_spec_id,
                    ExecutionSession.rl_algorithm_used == model.algorithm
                ).order_by(ExecutionSession.created_at).all()
                
                session_performance = []
                for session in sessions:
                    success_rate = session.passed_tests / session.total_tests if session.total_tests > 0 else 0
                    session_performance.append({
                        "session_id": session.id,
                        "success_rate": success_rate,
                        "total_tests": session.total_tests,
                        "timestamp": session.created_at.isoformat()
                    })
                
                training_history.append({
                    "model_id": model.id,
                    "algorithm": model.algorithm.value,
                    "episodes_trained": model.episodes_trained,
                    "session_performance": session_performance,
                    "created_at": model.created_at.isoformat()
                })
            
            return {
                "training_history": training_history,
                "api_spec_id": api_spec_id,
                "total_models_trained": len(models)
            }
            
        except Exception as e:
            logger.error(f"Failed to get RL training history: {str(e)}")
            return {
                "training_history": [],
                "error": str(e)
            }
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
