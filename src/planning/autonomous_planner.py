"""
Autonomous Planner Agent for intelligent test planning and strategy selection.
This is the brain of the autonomous testing system that decides what to test next.
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..database.connection import get_db_session
from ..database.models import (
    APISpecification, TestCase, ExecutionSession, TestExecution, 
    CoverageMetrics, TestStatus, TestType, AIGenerationLog
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TestStrategy(Enum):
    """Available test strategies for the planner to choose from."""
    CONTRACT_CRUD = "contract_crud"
    BOUNDARY_NEGATIVE = "boundary_negative"
    AUTH_MATRIX = "auth_matrix"
    STATEFUL_FLOWS = "stateful_flows"
    CHAOS_INJECTION = "chaos_injection"
    LLM_FUZZING = "llm_fuzzing"
    CONCURRENCY_RACE = "concurrency_race"
    PROPERTY_BASED = "property_based"
    DELTA_TESTING = "delta_testing"

@dataclass
class PlannerContext:
    """Context information for the planner to make decisions."""
    api_spec_id: int
    coverage_gaps: Dict[str, Any]
    recent_failures: List[Dict[str, Any]]
    execution_history: List[Dict[str, Any]]
    token_budget: int
    time_budget_minutes: int
    priority_endpoints: List[str]
    flaky_tests: List[int]

@dataclass
class TestPlan:
    """Generated test plan with selections and rationale."""
    plan_version: str
    budget_tokens: int
    selections: List[Dict[str, Any]]
    rationale: str
    estimated_coverage_gain: float
    estimated_defect_discovery: float
    risk_assessment: str

class AutonomousPlannerAgent:
    """
    Autonomous agent that plans test strategies using AI and coverage analysis.
    Makes intelligent decisions about which endpoints to test and how.
    """
    
    def __init__(self):
        self.db = get_db_session()
        self.openai_client = None
        self.plan_version = "v3"
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your_openai_api_key_here":
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("Autonomous planner initialized with OpenAI")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI: {str(e)}")
        
        if not self.openai_client:
            logger.info("Autonomous planner using deterministic algorithms")
    
    async def create_autonomous_test_plan(
        self,
        api_spec_id: int,
        token_budget: int = 150000,
        time_budget_minutes: int = 60,
        priority_endpoints: Optional[List[str]] = None
    ) -> TestPlan:
        """
        Create an autonomous test plan based on current system state.
        
        Args:
            api_spec_id: API specification to plan tests for
            token_budget: Maximum LLM tokens to use
            time_budget_minutes: Maximum execution time budget
            priority_endpoints: Optional list of high-priority endpoints
            
        Returns:
            TestPlan with optimal test selections and strategy
        """
        try:
            # Gather context for planning decisions
            context = await self._gather_planner_context(
                api_spec_id, token_budget, time_budget_minutes, priority_endpoints
            )
            
            # Generate test plan using AI or deterministic algorithms
            if self.openai_client:
                plan = await self._generate_plan_with_llm(context)
            else:
                plan = await self._generate_plan_deterministic(context)
            
            # Validate and optimize the plan
            validated_plan = await self._validate_and_optimize_plan(plan, context)
            
            logger.info(f"Generated autonomous test plan with {len(validated_plan.selections)} selections")
            return validated_plan
            
        except Exception as e:
            logger.error(f"Failed to create autonomous test plan: {str(e)}")
            raise ValueError(f"Planning failed: {str(e)}")
    
    async def _gather_planner_context(
        self,
        api_spec_id: int,
        token_budget: int,
        time_budget_minutes: int,
        priority_endpoints: Optional[List[str]]
    ) -> PlannerContext:
        """Gather all context needed for intelligent planning."""
        
        # Get coverage gaps
        coverage_gaps = await self._analyze_coverage_gaps(api_spec_id)
        
        # Get recent failures
        recent_failures = await self._analyze_recent_failures(api_spec_id)
        
        # Get execution history
        execution_history = await self._get_execution_history(api_spec_id, limit=10)
        
        # Identify flaky tests
        flaky_tests = await self._identify_flaky_tests(api_spec_id)
        
        return PlannerContext(
            api_spec_id=api_spec_id,
            coverage_gaps=coverage_gaps,
            recent_failures=recent_failures,
            execution_history=execution_history,
            token_budget=token_budget,
            time_budget_minutes=time_budget_minutes,
            priority_endpoints=priority_endpoints or [],
            flaky_tests=flaky_tests
        )
    
    async def _analyze_coverage_gaps(self, api_spec_id: int) -> Dict[str, Any]:
        """Analyze current coverage gaps to identify what needs testing."""
        
        try:
            # Get API specification
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == api_spec_id
            ).first()
            
            if not api_spec or not api_spec.parsed_endpoints:
                return {"endpoints": [], "methods": [], "status_codes": []}
            
            # Get all possible endpoints
            all_endpoints = []
            for path, methods in api_spec.parsed_endpoints.items():
                for method in methods.keys():
                    all_endpoints.append(f"{method.upper()} {path}")
            
            # Get covered endpoints from recent executions
            recent_sessions = self.db.query(ExecutionSession).filter(
                ExecutionSession.api_spec_id == api_spec_id
            ).order_by(ExecutionSession.created_at.desc()).limit(5).all()
            
            covered_endpoints = set()
            tested_status_codes = set()
            
            for session in recent_sessions:
                executions = self.db.query(TestExecution).filter(
                    TestExecution.session_id == session.id
                ).all()
                
                for execution in executions:
                    if execution.test_case:
                        covered_endpoints.add(f"{execution.test_case.method} {execution.test_case.endpoint}")
                    if execution.response_code:
                        tested_status_codes.add(execution.response_code)
            
            # Calculate gaps
            uncovered_endpoints = [ep for ep in all_endpoints if ep not in covered_endpoints]
            untested_status_codes = [200, 201, 400, 401, 403, 404, 422, 500] 
            gap_status_codes = [code for code in untested_status_codes if code not in tested_status_codes]
            
            # Analyze parameter coverage
            parameter_gaps = await self._analyze_parameter_coverage_gaps(api_spec, recent_sessions)
            
            return {
                "total_endpoints": len(all_endpoints),
                "covered_endpoints": len(covered_endpoints),
                "uncovered_endpoints": uncovered_endpoints,
                "coverage_percentage": len(covered_endpoints) / len(all_endpoints) * 100 if all_endpoints else 0,
                "untested_status_codes": gap_status_codes,
                "parameter_gaps": parameter_gaps,
                "priority_gaps": [ep for ep in uncovered_endpoints if any(priority in ep for priority in ["POST", "PUT", "DELETE"])]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze coverage gaps: {str(e)}")
            return {"endpoints": [], "methods": [], "status_codes": []}
    
    async def _analyze_recent_failures(self, api_spec_id: int) -> List[Dict[str, Any]]:
        """Analyze recent test failures to identify patterns."""
        
        try:
            # Get recent failed executions
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            failed_executions = self.db.query(TestExecution).join(
                ExecutionSession
            ).filter(
                ExecutionSession.api_spec_id == api_spec_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR]),
                TestExecution.created_at >= cutoff_date
            ).limit(50).all()
            
            failures = []
            for execution in failed_executions:
                if execution.test_case:
                    failure_fingerprint = self._generate_failure_fingerprint(execution)
                    failures.append({
                        "execution_id": execution.id,
                        "endpoint": execution.test_case.endpoint,
                        "method": execution.test_case.method,
                        "status_code": execution.response_code,
                        "error_message": execution.error_message,
                        "fingerprint": failure_fingerprint,
                        "test_type": execution.test_case.test_type.value,
                        "created_at": execution.created_at.isoformat()
                    })
            
            # Group failures by fingerprint to identify patterns
            failure_groups = {}
            for failure in failures:
                fp = failure["fingerprint"]
                if fp not in failure_groups:
                    failure_groups[fp] = []
                failure_groups[fp].append(failure)
            
            # Identify most common failure patterns
            common_patterns = sorted(
                failure_groups.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:10]
            
            return [
                {
                    "pattern": pattern,
                    "count": len(failures_list),
                    "examples": failures_list[:3],
                    "affected_endpoints": list(set(f["endpoint"] for f in failures_list))
                }
                for pattern, failures_list in common_patterns
            ]
            
        except Exception as e:
            logger.error(f"Failed to analyze recent failures: {str(e)}")
            return []
    
    def _generate_failure_fingerprint(self, execution: TestExecution) -> str:
        """Generate a stable fingerprint for failure deduplication."""
        
        try:
            endpoint = execution.test_case.endpoint if execution.test_case else "unknown"
            status_code = execution.response_code or 0
            error_msg = execution.error_message or ""
            
            # Normalize error message (remove timestamps, IDs, etc.)
            normalized_msg = self._normalize_error_message(error_msg)
            
            fingerprint_data = f"{endpoint}|{status_code}|{normalized_msg}"
            return hashlib.sha1(fingerprint_data.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Failed to generate failure fingerprint: {str(e)}")
            return "unknown"
    
    def _normalize_error_message(self, error_msg: str) -> str:
        """Normalize error message for fingerprinting."""
        
        import re
        
        # Remove timestamps
        error_msg = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', '', error_msg)
        
        # Remove UUIDs and IDs
        error_msg = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', error_msg)
        error_msg = re.sub(r'\b\d{3,}\b', 'ID', error_msg)
        
        # Remove request IDs and trace IDs
        error_msg = re.sub(r'(request|trace|correlation)[-_]?id[:\s]*\S+', '', error_msg, flags=re.IGNORECASE)
        
        # Normalize whitespace
        error_msg = ' '.join(error_msg.split())
        
        return error_msg.lower().strip()
    
    async def _analyze_parameter_coverage_gaps(
        self, 
        api_spec: APISpecification, 
        recent_sessions: List[ExecutionSession]
    ) -> Dict[str, Any]:
        """Analyze which parameters haven't been properly tested."""
        
        try:
            parameter_gaps = {}
            
            for path, methods in (api_spec.parsed_endpoints or {}).items():
                for method, method_info in methods.items():
                    parameters = method_info.get("parameters", [])
                    
                    if parameters:
                        endpoint_key = f"{method.upper()} {path}"
                        tested_params = set()
                        
                        # Find what parameters were tested in recent sessions
                        for session in recent_sessions:
                            executions = self.db.query(TestExecution).filter(
                                TestExecution.session_id == session.id
                            ).all()
                            
                            for execution in executions:
                                if execution.test_case and execution.test_case.endpoint == path:
                                    test_data = execution.test_case.test_data or {}
                                    tested_params.update(test_data.get("query_params", {}).keys())
                                    tested_params.update(test_data.get("path_params", {}).keys())
                                    tested_params.update(test_data.get("headers", {}).keys())
                        
                        # Find untested parameters
                        all_params = set(param.get("name", "") for param in parameters)
                        untested_params = all_params - tested_params
                        
                        if untested_params:
                            parameter_gaps[endpoint_key] = {
                                "untested_parameters": list(untested_params),
                                "total_parameters": len(all_params),
                                "coverage_percentage": len(tested_params) / len(all_params) * 100 if all_params else 0
                            }
            
            return parameter_gaps
            
        except Exception as e:
            logger.error(f"Failed to analyze parameter coverage: {str(e)}")
            return {}
    
    async def _get_execution_history(self, api_spec_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history for pattern analysis."""
        
        try:
            sessions = self.db.query(ExecutionSession).filter(
                ExecutionSession.api_spec_id == api_spec_id
            ).order_by(ExecutionSession.created_at.desc()).limit(limit).all()
            
            history = []
            for session in sessions:
                executions = self.db.query(TestExecution).filter(
                    TestExecution.session_id == session.id
                ).all()
                
                success_rate = session.passed_tests / session.total_tests if session.total_tests > 0 else 0
                avg_response_time = sum(ex.response_time_ms or 0 for ex in executions) / len(executions) if executions else 0
                
                history.append({
                    "session_id": session.id,
                    "total_tests": session.total_tests,
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "duration_seconds": session.duration_seconds,
                    "created_at": session.created_at.isoformat(),
                    "trigger": session.trigger
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get execution history: {str(e)}")
            return []
    
    async def _identify_flaky_tests(self, api_spec_id: int) -> List[int]:
        """Identify flaky tests that pass/fail inconsistently."""
        
        try:
            # Get test cases with multiple executions
            test_case_ids = self.db.query(TestCase.id).filter(
                TestCase.api_spec_id == api_spec_id
            ).all()
            
            flaky_tests = []
            
            for (test_case_id,) in test_case_ids:
                executions = self.db.query(TestExecution).filter(
                    TestExecution.test_case_id == test_case_id
                ).limit(10).all()  # Look at last 10 executions
                
                if len(executions) >= 3:  # Need at least 3 executions to detect flakiness
                    statuses = [ex.status for ex in executions]
                    passed_count = sum(1 for s in statuses if s == TestStatus.PASSED)
                    failed_count = sum(1 for s in statuses if s == TestStatus.FAILED)
                    
                    # Consider flaky if both passes and failures exist
                    if passed_count > 0 and failed_count > 0:
                        flakiness_ratio = min(passed_count, failed_count) / len(statuses)
                        if flakiness_ratio >= 0.2:  # At least 20% inconsistency
                            flaky_tests.append(test_case_id)
            
            return flaky_tests
            
        except Exception as e:
            logger.error(f"Failed to identify flaky tests: {str(e)}")
            return []
    
    async def _generate_plan_with_llm(self, context: PlannerContext) -> TestPlan:
        """Generate test plan using LLM intelligence."""
        
        try:
            prompt = self._build_planner_prompt(context)
            
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert API testing strategist. Generate optimal test plans in JSON format that maximize coverage and defect discovery while staying within budget constraints."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more deterministic planning
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            plan_data = self._parse_llm_plan_response(content, context)
            
            return TestPlan(
                plan_version=self.plan_version,
                budget_tokens=context.token_budget,
                selections=plan_data["selections"],
                rationale=plan_data["rationale"],
                estimated_coverage_gain=plan_data["estimated_coverage_gain"],
                estimated_defect_discovery=plan_data["estimated_defect_discovery"],
                risk_assessment=plan_data["risk_assessment"]
            )
            
        except Exception as e:
            logger.error(f"LLM plan generation failed: {str(e)}")
            # Fallback to deterministic planning
            return await self._generate_plan_deterministic(context)
    
    def _build_planner_prompt(self, context: PlannerContext) -> str:
        """Build the prompt for LLM-based planning."""
        
        return f"""
You are an autonomous API testing planner. Given the current coverage gaps and recent failures, 
select optimal endpoint+strategy pairs to maximize defect discovery and coverage within budget.

CURRENT CONTEXT:
- API Spec ID: {context.api_spec_id}
- Token Budget: {context.token_budget}
- Time Budget: {context.time_budget_minutes} minutes
- Coverage Gaps: {json.dumps(context.coverage_gaps, indent=2)}
- Recent Failures: {len(context.recent_failures)} patterns identified
- Flaky Tests: {len(context.flaky_tests)} tests identified
- Priority Endpoints: {context.priority_endpoints}

AVAILABLE STRATEGIES:
1. contract_crud - Basic CRUD operations testing
2. boundary_negative - Boundary values and negative test cases
3. auth_matrix - Authentication and authorization testing
4. stateful_flows - Multi-step workflow testing
5. chaos_injection - Fault injection and resilience testing
6. llm_fuzzing - AI-guided input fuzzing
7. concurrency_race - Race condition testing
8. property_based - Property invariant testing
9. delta_testing - Cross-version/environment testing

SELECTION CRITERIA:
- Prioritize uncovered endpoints and high-impact methods (POST, PUT, DELETE)
- Focus on endpoints with recent failures
- Balance between coverage gains and defect discovery potential
- Stay within token and time budgets
- Avoid flaky tests unless necessary
- Consider execution cost vs benefit

Generate a JSON response with this exact structure:
{{
  "plan_version": "v3",
  "budget_tokens": {context.token_budget},
  "selections": [
    {{
      "endpoint": "/api/endpoint",
      "method": "POST",
      "strategy": "boundary_negative",
      "batch_size": 25,
      "priority": "high",
      "rationale": "High-impact endpoint with coverage gaps"
    }}
  ],
  "rationale": "Overall planning rationale",
  "estimated_coverage_gain": 15.5,
  "estimated_defect_discovery": 8.2,
  "risk_assessment": "medium"
}}

Select 5-15 endpoint+strategy combinations that will provide maximum value.
"""
    
    def _parse_llm_plan_response(self, content: str, context: PlannerContext) -> Dict[str, Any]:
        """Parse and validate LLM planning response."""
        
        try:
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                plan_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["selections", "rationale", "estimated_coverage_gain", "estimated_defect_discovery"]
                for field in required_fields:
                    if field not in plan_data:
                        plan_data[field] = self._get_default_plan_field(field)
                
                # Validate selections
                validated_selections = []
                for selection in plan_data.get("selections", []):
                    if self._validate_selection(selection):
                        validated_selections.append(selection)
                
                plan_data["selections"] = validated_selections
                return plan_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM plan response: {str(e)}")
        
        # Return fallback plan
        return self._create_fallback_plan(context)
    
    def _validate_selection(self, selection: Dict[str, Any]) -> bool:
        """Validate a test selection from the LLM."""
        
        required_fields = ["endpoint", "method", "strategy", "batch_size"]
        return all(field in selection for field in required_fields)
    
    def _get_default_plan_field(self, field: str) -> Any:
        """Get default value for missing plan fields."""
        
        defaults = {
            "selections": [],
            "rationale": "Automated plan generation",
            "estimated_coverage_gain": 10.0,
            "estimated_defect_discovery": 5.0,
            "risk_assessment": "medium"
        }
        return defaults.get(field, "")
    
    async def _generate_plan_deterministic(self, context: PlannerContext) -> TestPlan:
        """Generate test plan using deterministic algorithms."""
        
        try:
            selections = []
            
            # Prioritize uncovered endpoints
            for endpoint in context.coverage_gaps.get("uncovered_endpoints", [])[:10]:
                method, path = endpoint.split(" ", 1)
                
                # Choose strategy based on method and endpoint characteristics
                strategy = self._choose_strategy_for_endpoint(method, path, context)
                batch_size = self._calculate_optimal_batch_size(method, strategy, context)
                
                selections.append({
                    "endpoint": path,
                    "method": method,
                    "strategy": strategy.value,
                    "batch_size": batch_size,
                    "priority": "high" if method in ["POST", "PUT", "DELETE"] else "medium",
                    "rationale": f"Coverage gap in {method} {path}"
                })
            
            # Add tests for endpoints with recent failures
            failure_endpoints = set()
            for failure_pattern in context.recent_failures[:5]:
                for endpoint in failure_pattern.get("affected_endpoints", []):
                    if endpoint not in failure_endpoints:
                        failure_endpoints.add(endpoint)
                        
                        # Parse endpoint
                        parts = endpoint.split("/")
                        method = "GET"  # Default, should be improved
                        path = endpoint
                        
                        selections.append({
                            "endpoint": path,
                            "method": method,
                            "strategy": TestStrategy.BOUNDARY_NEGATIVE.value,
                            "batch_size": 20,
                            "priority": "high",
                            "rationale": f"Recent failures detected in {endpoint}"
                        })
            
            # Limit selections to stay within budget
            selections = selections[:15]  # Maximum 15 selections
            
            return TestPlan(
                plan_version=self.plan_version,
                budget_tokens=context.token_budget,
                selections=selections,
                rationale="Deterministic plan focusing on coverage gaps and recent failures",
                estimated_coverage_gain=len(selections) * 2.5,
                estimated_defect_discovery=len(selections) * 1.5,
                risk_assessment="low"
            )
            
        except Exception as e:
            logger.error(f"Deterministic planning failed: {str(e)}")
            return self._create_emergency_plan(context)
    
    def _choose_strategy_for_endpoint(
        self, 
        method: str, 
        path: str, 
        context: PlannerContext
    ) -> TestStrategy:
        """Choose optimal strategy for an endpoint based on characteristics."""
        
        # Strategy selection logic
        if "auth" in path.lower() or "login" in path.lower():
            return TestStrategy.AUTH_MATRIX
        elif method in ["POST", "PUT", "PATCH"]:
            if "order" in path.lower() or "transaction" in path.lower():
                return TestStrategy.STATEFUL_FLOWS
            else:
                return TestStrategy.BOUNDARY_NEGATIVE
        elif method == "GET":
            if "{id}" in path or "/{" in path:
                return TestStrategy.PROPERTY_BASED
            else:
                return TestStrategy.CONTRACT_CRUD
        elif method == "DELETE":
            return TestStrategy.STATEFUL_FLOWS
        else:
            return TestStrategy.CONTRACT_CRUD
    
    def _calculate_optimal_batch_size(
        self, 
        method: str, 
        strategy: TestStrategy, 
        context: PlannerContext
    ) -> int:
        """Calculate optimal batch size for a strategy."""
        
        # Base batch sizes by strategy complexity
        base_sizes = {
            TestStrategy.CONTRACT_CRUD: 10,
            TestStrategy.BOUNDARY_NEGATIVE: 25,
            TestStrategy.AUTH_MATRIX: 15,
            TestStrategy.STATEFUL_FLOWS: 8,
            TestStrategy.CHAOS_INJECTION: 5,
            TestStrategy.LLM_FUZZING: 30,
            TestStrategy.CONCURRENCY_RACE: 12,
            TestStrategy.PROPERTY_BASED: 20,
            TestStrategy.DELTA_TESTING: 6
        }
        
        base_size = base_sizes.get(strategy, 15)
        
        # Adjust based on available budget
        budget_factor = min(context.token_budget / 100000, 2.0)  # Scale with budget
        time_factor = min(context.time_budget_minutes / 30, 2.0)  # Scale with time
        
        adjusted_size = int(base_size * min(budget_factor, time_factor))
        return max(5, min(adjusted_size, 50))  # Keep within reasonable bounds
    
    async def _validate_and_optimize_plan(self, plan: TestPlan, context: PlannerContext) -> TestPlan:
        """Validate and optimize the generated plan."""
        
        try:
            # Remove duplicates
            seen_combinations = set()
            unique_selections = []
            
            for selection in plan.selections:
                combo = f"{selection['method']} {selection['endpoint']} {selection['strategy']}"
                if combo not in seen_combinations:
                    seen_combinations.add(combo)
                    unique_selections.append(selection)
            
            plan.selections = unique_selections
            
            # Estimate total cost and adjust if needed
            total_estimated_tokens = sum(s.get("batch_size", 10) * 500 for s in plan.selections)
            
            if total_estimated_tokens > context.token_budget:
                # Scale down batch sizes proportionally
                scale_factor = context.token_budget / total_estimated_tokens
                for selection in plan.selections:
                    selection["batch_size"] = max(3, int(selection["batch_size"] * scale_factor))
            
            # Re-calculate estimates
            plan.estimated_coverage_gain = len(plan.selections) * 2.5
            plan.estimated_defect_discovery = sum(
                5 if s.get("priority") == "high" else 3 for s in plan.selections
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Plan validation failed: {str(e)}")
            return plan
    
    def _create_fallback_plan(self, context: PlannerContext) -> Dict[str, Any]:
        """Create a basic fallback plan."""
        
        return {
            "selections": [
                {
                    "endpoint": "/health",
                    "method": "GET",
                    "strategy": "contract_crud",
                    "batch_size": 5,
                    "priority": "medium",
                    "rationale": "Basic health check coverage"
                }
            ],
            "rationale": "Fallback plan due to parsing error",
            "estimated_coverage_gain": 5.0,
            "estimated_defect_discovery": 2.0,
            "risk_assessment": "low"
        }
    
    def _create_emergency_plan(self, context: PlannerContext) -> TestPlan:
        """Create emergency plan when all else fails."""
        
        return TestPlan(
            plan_version=self.plan_version,
            budget_tokens=context.token_budget,
            selections=[{
                "endpoint": "/status",
                "method": "GET", 
                "strategy": "contract_crud",
                "batch_size": 3,
                "priority": "low",
                "rationale": "Emergency fallback"
            }],
            rationale="Emergency plan - system recovery needed",
            estimated_coverage_gain=1.0,
            estimated_defect_discovery=0.5,
            risk_assessment="minimal"
        )
    
    async def execute_autonomous_plan(self, plan: TestPlan, api_spec_id: int) -> Dict[str, Any]:
        """Execute an autonomous test plan."""
        
        try:
            from ..ai.test_generator import AITestGenerator
            from ..execution.test_executor import APITestExecutor
            
            generator = AITestGenerator()
            executor = APITestExecutor()
            
            execution_results = []
            total_tests_generated = 0
            total_tests_executed = 0
            
            for selection in plan.selections:
                try:
                    # Generate test cases for this selection
                    test_cases = await generator.generate_test_cases(
                        api_spec_id=api_spec_id,
                        endpoint_path=selection["endpoint"],
                        method=selection["method"],
                        test_types=[TestType.FUNCTIONAL, TestType.EDGE_CASE],
                        count=selection["batch_size"],
                        custom_context=f"Strategy: {selection['strategy']}"
                    )
                    
                    total_tests_generated += len(test_cases)
                    
                    # Execute the generated tests
                    session = await executor.execute_test_session(
                        api_spec_id=api_spec_id,
                        test_case_ids=[tc.id for tc in test_cases],
                        session_name=f"Autonomous Plan: {selection['strategy']} on {selection['endpoint']}",
                        trigger="autonomous_planner"
                    )
                    
                    total_tests_executed += session.total_tests
                    
                    execution_results.append({
                        "selection": selection,
                        "session_id": session.id,
                        "tests_generated": len(test_cases),
                        "tests_executed": session.total_tests,
                        "success_rate": session.passed_tests / session.total_tests if session.total_tests > 0 else 0,
                        "coverage_gain": getattr(session, 'endpoint_coverage', {}).get('percentage', 0)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to execute selection {selection}: {str(e)}")
                    execution_results.append({
                        "selection": selection,
                        "error": str(e),
                        "tests_generated": 0,
                        "tests_executed": 0
                    })
            
            # Calculate overall results
            total_success_rate = sum(
                r.get("success_rate", 0) * r.get("tests_executed", 0) 
                for r in execution_results
            ) / total_tests_executed if total_tests_executed > 0 else 0
            
            total_coverage_gain = sum(r.get("coverage_gain", 0) for r in execution_results) / len(execution_results) if execution_results else 0
            
            return {
                "plan_executed": True,
                "execution_results": execution_results,
                "summary": {
                    "total_selections": len(plan.selections),
                    "successful_selections": len([r for r in execution_results if "error" not in r]),
                    "total_tests_generated": total_tests_generated,
                    "total_tests_executed": total_tests_executed,
                    "overall_success_rate": total_success_rate,
                    "coverage_gain_percentage": total_coverage_gain
                },
                "plan_performance": {
                    "estimated_vs_actual_coverage": {
                        "estimated": plan.estimated_coverage_gain,
                        "actual": total_coverage_gain,
                        "accuracy": abs(plan.estimated_coverage_gain - total_coverage_gain) < 5.0
                    },
                    "budget_utilization": {
                        "token_budget": plan.budget_tokens,
                        "estimated_tokens_used": total_tests_generated * 500  # Rough estimate
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute autonomous plan: {str(e)}")
            return {
                "plan_executed": False,
                "error": str(e),
                "execution_results": []
            }
    
    async def analyze_plan_performance(
        self, 
        executed_plan: Dict[str, Any], 
        original_plan: TestPlan
    ) -> Dict[str, Any]:
        """Analyze how well the executed plan performed vs estimates."""
        
        try:
            execution_results = executed_plan.get("execution_results", [])
            summary = executed_plan.get("summary", {})
            
            # Calculate accuracy metrics
            estimated_coverage = original_plan.estimated_coverage_gain
            actual_coverage = summary.get("coverage_gain_percentage", 0)
            coverage_accuracy = 100 - abs(estimated_coverage - actual_coverage)
            
            estimated_defects = original_plan.estimated_defect_discovery
            actual_defects = sum(
                r.get("tests_executed", 0) - (r.get("success_rate", 1) * r.get("tests_executed", 0))
                for r in execution_results
            )
            defect_accuracy = 100 - abs(estimated_defects - actual_defects) if estimated_defects > 0 else 100
            
            # Strategy effectiveness analysis
            strategy_performance = {}
            for result in execution_results:
                strategy = result.get("selection", {}).get("strategy", "unknown")
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "executions": 0,
                        "total_success_rate": 0,
                        "total_coverage": 0
                    }
                
                perf = strategy_performance[strategy]
                perf["executions"] += 1
                perf["total_success_rate"] += result.get("success_rate", 0)
                perf["total_coverage"] += result.get("coverage_gain", 0)
            
            # Calculate average performance per strategy
            for strategy, perf in strategy_performance.items():
                if perf["executions"] > 0:
                    perf["avg_success_rate"] = perf["total_success_rate"] / perf["executions"]
                    perf["avg_coverage"] = perf["total_coverage"] / perf["executions"]
            
            return {
                "plan_id": hash(str(original_plan.selections)),
                "accuracy_metrics": {
                    "coverage_accuracy": coverage_accuracy,
                    "defect_discovery_accuracy": defect_accuracy,
                    "overall_accuracy": (coverage_accuracy + defect_accuracy) / 2
                },
                "strategy_performance": strategy_performance,
                "learning_insights": {
                    "best_performing_strategy": max(
                        strategy_performance.items(), 
                        key=lambda x: x[1].get("avg_success_rate", 0)
                    )[0] if strategy_performance else "none",
                    "coverage_champions": [
                        strategy for strategy, perf in strategy_performance.items() 
                        if perf.get("avg_coverage", 0) > 10
                    ],
                    "improvement_opportunities": [
                        strategy for strategy, perf in strategy_performance.items()
                        if perf.get("avg_success_rate", 1) < 0.7
                    ]
                },
                "recommendations": self._generate_learning_recommendations(strategy_performance)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze plan performance: {str(e)}")
            return {"error": str(e)}
    
    def _generate_learning_recommendations(self, strategy_performance: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on strategy performance analysis."""
        
        recommendations = []
        
        for strategy, perf in strategy_performance.items():
            avg_success_rate = perf.get("avg_success_rate", 0)
            avg_coverage = perf.get("avg_coverage", 0)
            
            if avg_success_rate < 0.5:
                recommendations.append(f"Strategy '{strategy}' has low success rate ({avg_success_rate:.1%}) - consider tuning parameters")
            
            if avg_coverage > 15:
                recommendations.append(f"Strategy '{strategy}' provides excellent coverage ({avg_coverage:.1f}%) - increase usage")
            
            if perf.get("executions", 0) < 3:
                recommendations.append(f"Strategy '{strategy}' needs more data points for reliable analysis")
        
        if not recommendations:
            recommendations.append("All strategies performing within expected ranges")
        
        return recommendations
    
    async def get_planning_insights(self, api_spec_id: int) -> Dict[str, Any]:
        """Get insights about planning effectiveness over time."""
        
        try:
            # Get recent autonomous executions
            recent_sessions = self.db.query(ExecutionSession).filter(
                ExecutionSession.api_spec_id == api_spec_id,
                ExecutionSession.trigger == "autonomous_planner"
            ).order_by(ExecutionSession.created_at.desc()).limit(10).all()
            
            if not recent_sessions:
                return {
                    "message": "No autonomous planning data available yet",
                    "insights": []
                }
            
            # Analyze planning trends
            planning_metrics = []
            for session in recent_sessions:
                coverage_metrics = self.db.query(CoverageMetrics).filter(
                    CoverageMetrics.session_id == session.id
                ).first()
                
                if coverage_metrics:
                    planning_metrics.append({
                        "session_id": session.id,
                        "created_at": session.created_at.isoformat(),
                        "coverage_percentage": coverage_metrics.endpoint_coverage_pct,
                        "bugs_found": coverage_metrics.bugs_found,
                        "quality_score": coverage_metrics.quality_score,
                        "duration_minutes": session.duration_seconds / 60 if session.duration_seconds else 0
                    })
            
            # Calculate trends
            if len(planning_metrics) >= 2:
                coverage_trend = planning_metrics[0]["coverage_percentage"] - planning_metrics[-1]["coverage_percentage"]
                quality_trend = planning_metrics[0]["quality_score"] - planning_metrics[-1]["quality_score"]
            else:
                coverage_trend = 0
                quality_trend = 0
            
            insights = []
            
            if coverage_trend > 5:
                insights.append("Coverage is improving over time - planner learning effectively")
            elif coverage_trend < -5:
                insights.append("Coverage declining - planner may need retuning")
            
            if quality_trend > 10:
                insights.append("Test quality improving - good strategy selection")
            elif quality_trend < -10:
                insights.append("Test quality declining - review strategy effectiveness")
            
            avg_bugs_per_session = sum(m["bugs_found"] for m in planning_metrics) / len(planning_metrics)
            if avg_bugs_per_session > 5:
                insights.append(f"High defect discovery rate ({avg_bugs_per_session:.1f} bugs/session)")
            
            return {
                "planning_metrics": planning_metrics,
                "trends": {
                    "coverage_trend": coverage_trend,
                    "quality_trend": quality_trend,
                    "avg_bugs_per_session": avg_bugs_per_session
                },
                "insights": insights,
                "total_autonomous_sessions": len(recent_sessions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get planning insights: {str(e)}")
            return {"insights": [], "error": str(e)}
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
