"""
Hybrid Reinforcement Learning optimization engine combining Q-learning, PPO, and evolutionary search.
"""

import os
import json
import numpy as np
import gymnasium as gym
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch

from ..database.models import (
    TestCase, ExecutionSession, TestExecution, TestStatus, 
    RLModel, RLAlgorithm, APISpecification
)
from ..database.connection import get_db_session
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RLOptimizationError(Exception):
    """Custom exception for RL optimization errors."""
    pass

class TestSelectionEnv(gym.Env):
    """
    Custom Gym environment for test case selection optimization.
    """
    
    def __init__(self, api_spec_id: int, test_cases: List[TestCase]):
        super(TestSelectionEnv, self).__init__()
        
        self.api_spec_id = api_spec_id
        self.test_cases = test_cases
        self.current_step = 0
        self.max_steps = len(test_cases)
        
        # State: features of current test case + execution history
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        # Action: select (1) or skip (0) test case
        self.action_space = gym.spaces.Discrete(2)
        
        # Track execution history for state
        self.execution_history = []
        self.coverage_achieved = set()
        self.bugs_found = 0
        
    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.execution_history = []
        self.coverage_achieved = set()
        self.bugs_found = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return new state, reward, done flag."""
        
        if self.current_step >= len(self.test_cases):
            return self._get_observation(), 0, True, True, {}
        
        test_case = self.test_cases[self.current_step]
        reward = 0
        
        if action == 1:  # Select test case
            # Simulate test execution result (in real scenario, this would come from actual execution)
            reward = self._calculate_reward(test_case)
            self.execution_history.append({
                'test_case_id': test_case.id,
                'selected': True,
                'reward': reward
            })
            
            # Update coverage and bug tracking
            self._update_coverage(test_case)
        else:  # Skip test case
            reward = -0.1  # Small penalty for skipping
            self.execution_history.append({
                'test_case_id': test_case.id,
                'selected': False,
                'reward': reward
            })
        
        self.current_step += 1
        done = self.current_step >= len(self.test_cases)
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        """Get current observation/state."""
        if self.current_step >= len(self.test_cases):
            return np.zeros(20, dtype=np.float32)
        
        test_case = self.test_cases[self.current_step]
        
        # Feature vector representing test case and execution state
        features = [
            # Test case features
            float(test_case.success_rate),
            float(test_case.selection_count) / 100.0,  # Normalized
            float(len(test_case.assertions or [])) / 10.0,  # Normalized
            float(test_case.test_type.value == "functional"),
            float(test_case.test_type.value == "edge_case"),
            float(test_case.test_type.value == "security"),
            float(test_case.test_type.value == "performance"),
            
            # Execution context features
            float(len(self.execution_history)) / float(self.max_steps),
            float(len(self.coverage_achieved)) / 20.0,  # Normalized coverage
            float(self.bugs_found) / 10.0,  # Normalized bug count
            
            # Time-based features
            float(self.current_step) / float(self.max_steps),
            
            # Endpoint diversity features
            float(self._endpoint_coverage_ratio()),
            float(self._method_diversity_ratio()),
            
            # Recent performance features
            float(self._recent_success_rate()),
            float(self._recent_bug_rate()),
            
            # Remaining slots (can be extended)
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, test_case: TestCase) -> float:
        """Calculate reward for selecting a test case."""
        
        base_reward = 1.0
        
        # Reward based on historical success rate
        success_bonus = test_case.success_rate * 0.5
        
        # Reward for new coverage
        coverage_bonus = self._get_coverage_bonus(test_case)
        
        # Reward for test diversity
        diversity_bonus = self._get_diversity_bonus(test_case)
        
        # Penalty for over-selection
        selection_penalty = min(test_case.selection_count / 50.0, 1.0) * 0.3
        
        total_reward = base_reward + success_bonus + coverage_bonus + diversity_bonus - selection_penalty
        return max(total_reward, -1.0)  # Clip minimum reward
    
    def _update_coverage(self, test_case: TestCase):
        """Update coverage tracking."""
        endpoint_key = f"{test_case.method}_{test_case.endpoint}"
        self.coverage_achieved.add(endpoint_key)
        
        # Simulate bug finding (in real scenario, comes from execution results)
        if test_case.test_type.value in ["edge_case", "security"]:
            if np.random.random() < 0.1:  # 10% chance of finding bug
                self.bugs_found += 1
    
    def _get_coverage_bonus(self, test_case: TestCase) -> float:
        """Calculate bonus for new coverage."""
        endpoint_key = f"{test_case.method}_{test_case.endpoint}"
        if endpoint_key not in self.coverage_achieved:
            return 2.0  # High reward for new coverage
        return 0.0
    
    def _get_diversity_bonus(self, test_case: TestCase) -> float:
        """Calculate bonus for test diversity."""
        recent_types = [h.get('test_type') for h in self.execution_history[-5:]]
        if test_case.test_type.value not in recent_types:
            return 0.5
        return 0.0
    
    def _endpoint_coverage_ratio(self) -> float:
        """Calculate endpoint coverage ratio."""
        total_endpoints = len(set(f"{tc.method}_{tc.endpoint}" for tc in self.test_cases))
        return len(self.coverage_achieved) / max(total_endpoints, 1)
    
    def _method_diversity_ratio(self) -> float:
        """Calculate HTTP method diversity."""
        methods_used = set(tc.method for tc in self.test_cases[:self.current_step])
        total_methods = set(tc.method for tc in self.test_cases)
        return len(methods_used) / max(len(total_methods), 1)
    
    def _recent_success_rate(self) -> float:
        """Calculate recent success rate."""
        recent_results = self.execution_history[-10:]
        if not recent_results:
            return 0.5  # Default
        successes = sum(1 for r in recent_results if r.get('reward', 0) > 0)
        return successes / len(recent_results)
    
    def _recent_bug_rate(self) -> float:
        """Calculate recent bug discovery rate."""
        return min(self.bugs_found / max(len(self.execution_history), 1), 1.0)

class EvolutionaryOptimizer:
    """
    Evolutionary algorithm for test case selection and parameter optimization.
    """
    
    def __init__(self, population_size: int = 20, generations: int = 50):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    def evolve_test_selection(
        self,
        test_cases: List[TestCase],
        target_coverage: float = 0.8,
        max_tests: int = 50
    ) -> List[int]:
        """
        Evolve optimal test case selection using genetic algorithm.
        
        Returns:
            List of selected test case IDs
        """
        
        # Initialize population (binary strings representing test selection)
        population = []
        for _ in range(self.population_size):
            individual = np.random.choice([0, 1], size=len(test_cases), p=[0.7, 0.3])
            population.append(individual)
        
        best_fitness = -float('inf')
        best_individual = None
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, test_cases, target_coverage, max_tests)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep best individuals
            elite_count = max(2, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            logger.debug(f"Generation {generation}: Best fitness = {best_fitness:.3f}")
        
        # Return selected test case IDs
        selected_indices = np.where(best_individual == 1)[0]
        return [test_cases[i].id for i in selected_indices]
    
    def _evaluate_fitness(
        self,
        individual: np.ndarray,
        test_cases: List[TestCase],
        target_coverage: float,
        max_tests: int
    ) -> float:
        """Evaluate fitness of an individual (test selection)."""
        
        selected_tests = [test_cases[i] for i in range(len(individual)) if individual[i] == 1]
        
        if len(selected_tests) == 0:
            return -1000  # Invalid solution
        
        # Coverage fitness
        unique_endpoints = set(f"{tc.method}_{tc.endpoint}" for tc in selected_tests)
        all_endpoints = set(f"{tc.method}_{tc.endpoint}" for tc in test_cases)
        coverage_ratio = len(unique_endpoints) / len(all_endpoints)
        coverage_fitness = coverage_ratio * 100
        
        # Efficiency fitness (prefer fewer tests)
        efficiency_fitness = max(0, (max_tests - len(selected_tests)) / max_tests) * 20
        
        # Quality fitness (based on success rates)
        quality_fitness = np.mean([tc.success_rate for tc in selected_tests]) * 30
        
        # Diversity fitness
        test_types = [tc.test_type.value for tc in selected_tests]
        diversity_fitness = len(set(test_types)) / 5.0 * 15  # Assuming 5 test types
        
        # Penalty for exceeding test limit
        penalty = max(0, len(selected_tests) - max_tests) * 10
        
        total_fitness = coverage_fitness + efficiency_fitness + quality_fitness + diversity_fitness - penalty
        
        return total_fitness
    
    def _tournament_selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        """Tournament selection for choosing parents."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Bit-flip mutation."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        return mutated

class HybridRLOptimizer:
    """
    Hybrid RL optimization system combining Q-learning, PPO, and evolutionary search.
    """
    
    def __init__(self):
        self.db = get_db_session()
        self.model_path = os.getenv("RL_MODEL_PATH", "./data/rl_models")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize optimizers
        self.evolutionary_optimizer = EvolutionaryOptimizer()
        
        logger.info("Hybrid RL Optimizer initialized")
    
    async def optimize_test_selection(
        self,
        api_spec_id: int,
        optimization_goal: str = "coverage",
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        max_tests: int = 50,
        training_episodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Optimize test case selection using the specified RL algorithm.
        
        Args:
            api_spec_id: API specification ID
            optimization_goal: Goal to optimize for ('coverage', 'bugs', 'efficiency')
            algorithm: RL algorithm to use
            max_tests: Maximum number of tests to select
            training_episodes: Number of training episodes
            
        Returns:
            Optimization results and selected test cases
        """
        
        try:
            # Get test cases for the API
            test_cases = self.db.query(TestCase).filter(
                TestCase.api_spec_id == api_spec_id,
                TestCase.is_active == True
            ).all()
            
            if not test_cases:
                raise RLOptimizationError(f"No test cases found for API specification {api_spec_id}")
            
            logger.info(f"Starting RL optimization with {len(test_cases)} test cases using {algorithm.value}")
            
            if algorithm == RLAlgorithm.PPO:
                results = await self._optimize_with_ppo(
                    api_spec_id, test_cases, optimization_goal, training_episodes
                )
            elif algorithm == RLAlgorithm.Q_LEARNING:
                results = await self._optimize_with_dqn(
                    api_spec_id, test_cases, optimization_goal, training_episodes
                )
            elif algorithm == RLAlgorithm.EVOLUTIONARY:
                results = await self._optimize_with_evolutionary(
                    api_spec_id, test_cases, max_tests
                )
            else:
                raise RLOptimizationError(f"Unsupported algorithm: {algorithm}")
            
            # Save RL model and results
            await self._save_rl_model(api_spec_id, algorithm, results)
            
            return results
            
        except Exception as e:
            logger.error(f"RL optimization failed: {str(e)}")
            raise RLOptimizationError(f"Optimization failed: {str(e)}")
    
    async def _optimize_with_ppo(
        self,
        api_spec_id: int,
        test_cases: List[TestCase],
        optimization_goal: str,
        training_episodes: int
    ) -> Dict[str, Any]:
        """Optimize using Proximal Policy Optimization (PPO)."""
        
        # Create environment
        env = TestSelectionEnv(api_spec_id, test_cases)
        
        # Initialize PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.getenv("TENSORBOARD_LOG_DIR", "./logs/tensorboard"),
            learning_rate=0.0003,
            n_steps=64,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2
        )
        
        # Train the model
        logger.info(f"Training PPO model for {training_episodes} episodes")
        model.learn(total_timesteps=training_episodes)
        
        # Evaluate the trained model
        selected_test_ids = await self._evaluate_model(model, env, test_cases)
        
        return {
            "algorithm": "PPO",
            "selected_test_ids": selected_test_ids,
            "training_episodes": training_episodes,
            "model": model,
            "optimization_score": len(selected_test_ids) / len(test_cases)
        }
    
    async def _optimize_with_dqn(
        self,
        api_spec_id: int,
        test_cases: List[TestCase],
        optimization_goal: str,
        training_episodes: int
    ) -> Dict[str, Any]:
        """Optimize using Deep Q-Network (DQN)."""
        
        # Create environment
        env = TestSelectionEnv(api_spec_id, test_cases)
        
        # Initialize DQN model
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.getenv("TENSORBOARD_LOG_DIR", "./logs/tensorboard"),
            learning_rate=0.001,
            buffer_size=10000,
            learning_starts=100,
            target_update_interval=500,
            train_freq=1,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            gamma=0.99
        )
        
        # Train the model
        logger.info(f"Training DQN model for {training_episodes} episodes")
        model.learn(total_timesteps=training_episodes)
        
        # Evaluate the trained model
        selected_test_ids = await self._evaluate_model(model, env, test_cases)
        
        return {
            "algorithm": "DQN",
            "selected_test_ids": selected_test_ids,
            "training_episodes": training_episodes,
            "model": model,
            "optimization_score": len(selected_test_ids) / len(test_cases)
        }
    
    async def _optimize_with_evolutionary(
        self,
        api_spec_id: int,
        test_cases: List[TestCase],
        max_tests: int
    ) -> Dict[str, Any]:
        """Optimize using evolutionary algorithm."""
        
        logger.info(f"Running evolutionary optimization with population_size={self.evolutionary_optimizer.population_size}")
        
        selected_test_ids = self.evolutionary_optimizer.evolve_test_selection(
            test_cases=test_cases,
            max_tests=max_tests
        )
        
        return {
            "algorithm": "Evolutionary",
            "selected_test_ids": selected_test_ids,
            "population_size": self.evolutionary_optimizer.population_size,
            "generations": self.evolutionary_optimizer.generations,
            "optimization_score": len(selected_test_ids) / len(test_cases)
        }
    
    async def _evaluate_model(
        self,
        model: Any,
        env: TestSelectionEnv,
        test_cases: List[TestCase]
    ) -> List[int]:
        """Evaluate trained model and return selected test case IDs."""
        
        selected_test_ids = []
        obs, _ = env.reset()
        
        for i in range(len(test_cases)):
            action, _ = model.predict(obs, deterministic=True)
            
            if action == 1:  # Select test case
                selected_test_ids.append(test_cases[i].id)
            
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        logger.info(f"Model selected {len(selected_test_ids)} test cases out of {len(test_cases)}")
        return selected_test_ids
    
    async def _save_rl_model(
        self,
        api_spec_id: int,
        algorithm: RLAlgorithm,
        results: Dict[str, Any]
    ):
        """Save RL model and results to database."""
        
        try:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Serialize model if present
            model_state = None
            if "model" in results:
                model_file = os.path.join(
                    self.model_path,
                    f"{algorithm.value}_{api_spec_id}_{model_version}.pkl"
                )
                
                # Save model to file
                if hasattr(results["model"], "save"):
                    results["model"].save(model_file)
                else:
                    with open(model_file, "wb") as f:
                        pickle.dump(results["model"], f)
                
                model_state = {"model_file": model_file}
            
            # Create database record
            rl_model = RLModel(
                api_spec_id=api_spec_id,
                algorithm=algorithm,
                model_version=model_version,
                description=f"Hybrid RL optimization using {algorithm.value}",
                model_state=model_state,
                hyperparameters={
                    "training_episodes": results.get("training_episodes", 0),
                    "optimization_goal": "coverage",
                    "max_tests": results.get("max_tests", 50)
                },
                episodes_trained=results.get("training_episodes", 0),
                average_reward=results.get("optimization_score", 0.0),
                best_reward=results.get("optimization_score", 0.0),
                convergence_score=results.get("optimization_score", 0.0),
                training_history={
                    "selected_tests_count": len(results.get("selected_test_ids", [])),
                    "optimization_results": results
                },
                is_active=True,
                is_trained=True
            )
            
            self.db.add(rl_model)
            self.db.commit()
            
            logger.info(f"Saved RL model {rl_model.id} for API {api_spec_id}")
            
        except Exception as e:
            logger.error(f"Failed to save RL model: {str(e)}")
    
    def get_optimization_recommendations(
        self,
        api_spec_id: int,
        current_test_selection: List[int]
    ) -> Dict[str, Any]:
        """Get recommendations for improving test selection."""
        
        try:
            # Get latest RL model
            rl_model = self.db.query(RLModel).filter(
                RLModel.api_spec_id == api_spec_id,
                RLModel.is_active == True
            ).order_by(RLModel.created_at.desc()).first()
            
            if not rl_model:
                return {"message": "No trained RL model found. Train a model first."}
            
            # Get all test cases
            test_cases = self.db.query(TestCase).filter(
                TestCase.api_spec_id == api_spec_id,
                TestCase.is_active == True
            ).all()
            
            # Analyze current selection
            selected_tests = [tc for tc in test_cases if tc.id in current_test_selection]
            unselected_tests = [tc for tc in test_cases if tc.id not in current_test_selection]
            
            # Coverage analysis
            selected_endpoints = set(f"{tc.method}_{tc.endpoint}" for tc in selected_tests)
            all_endpoints = set(f"{tc.method}_{tc.endpoint}" for tc in test_cases)
            coverage_pct = len(selected_endpoints) / len(all_endpoints) * 100
            
            # Diversity analysis
            selected_types = set(tc.test_type.value for tc in selected_tests)
            
            # Recommendations
            recommendations = []
            
            if coverage_pct < 80:
                missed_endpoints = all_endpoints - selected_endpoints
                recommendations.append({
                    "type": "coverage",
                    "priority": "high",
                    "message": f"Coverage is only {coverage_pct:.1f}%. Consider adding tests for: {list(missed_endpoints)[:3]}"
                })
            
            if len(selected_types) < 3:
                missing_types = {"functional", "edge_case", "security", "performance"} - selected_types
                recommendations.append({
                    "type": "diversity",
                    "priority": "medium",
                    "message": f"Low test diversity. Consider adding {', '.join(missing_types)} tests"
                })
            
            # Performance recommendations
            low_performing_tests = [tc for tc in selected_tests if tc.success_rate < 0.5]
            if low_performing_tests:
                recommendations.append({
                    "type": "performance",
                    "priority": "medium",
                    "message": f"Consider reviewing {len(low_performing_tests)} low-performing test cases"
                })
            
            return {
                "current_coverage": coverage_pct,
                "test_diversity": len(selected_types),
                "recommendations": recommendations,
                "model_used": rl_model.algorithm.value,
                "last_trained": rl_model.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return {"error": str(e)}
    
    def get_rl_model_performance(self, api_spec_id: int) -> Dict[str, Any]:
        """Get performance metrics for RL models."""
        
        try:
            models = self.db.query(RLModel).filter(
                RLModel.api_spec_id == api_spec_id
            ).order_by(RLModel.created_at.desc()).all()
            
            if not models:
                return {"message": "No RL models found"}
            
            performance_data = []
            for model in models:
                performance_data.append({
                    "algorithm": model.algorithm.value,
                    "version": model.model_version,
                    "episodes_trained": model.episodes_trained,
                    "average_reward": model.average_reward,
                    "best_reward": model.best_reward,
                    "convergence_score": model.convergence_score,
                    "created_at": model.created_at.isoformat(),
                    "is_active": model.is_active
                })
            
            return {
                "models": performance_data,
                "total_models": len(models),
                "best_performing": max(performance_data, key=lambda x: x["best_reward"])
            }
            
        except Exception as e:
            logger.error(f"Failed to get RL model performance: {str(e)}")
            return {"error": str(e)}
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'db'):
            self.db.close()
