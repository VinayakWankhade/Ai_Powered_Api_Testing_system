"""
Sandboxed test execution engine with comprehensive logging and coverage tracking.
"""

import asyncio
import time
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..database.models import (
    TestCase, ExecutionSession, TestExecution, TestStatus, 
    CoverageMetrics, APISpecification
)
from ..database.connection import get_db_session
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TestExecutionError(Exception):
    """Custom exception for test execution errors."""
    pass

class APITestExecutor:
    """
    Sandboxed test executor with coverage tracking and comprehensive result logging.
    """

    def __init__(self, max_concurrent_tests: int = 5, timeout_seconds: int = 300):
        self.db = get_db_session()
        self.max_concurrent_tests = max_concurrent_tests
        self.timeout_seconds = timeout_seconds
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Test executor initialized with max_concurrent_tests={max_concurrent_tests}, timeout={timeout_seconds}s")

    async def execute_test_session(
        self,
        api_spec_id: int,
        test_case_ids: Optional[List[int]] = None,
        session_name: Optional[str] = None,
        trigger: str = "manual"
    ) -> ExecutionSession:
        """
        Execute a test session with multiple test cases.
        
        Args:
            api_spec_id: API specification ID
            test_case_ids: Specific test cases to run (if None, runs all active tests)
            session_name: Optional name for the session
            trigger: What triggered this session (manual, scheduled, ci_cd, etc.)
            
        Returns:
            ExecutionSession with results
        """
        try:
            # Create execution session
            session = ExecutionSession(
                api_spec_id=api_spec_id,
                name=session_name or f"Test Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                trigger=trigger,
                started_at=datetime.utcnow()
            )
            
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            
            # Get test cases to execute
            if test_case_ids:
                test_cases = self.db.query(TestCase).filter(
                    TestCase.id.in_(test_case_ids),
                    TestCase.is_active == True
                ).all()
            else:
                test_cases = self.db.query(TestCase).filter(
                    TestCase.api_spec_id == api_spec_id,
                    TestCase.is_active == True
                ).all()
            
            if not test_cases:
                raise TestExecutionError("No test cases found for execution")
            
            session.total_tests = len(test_cases)
            self.db.commit()
            
            logger.info(f"Starting execution session {session.id} with {len(test_cases)} test cases")
            
            # Execute tests concurrently
            test_executions = await self._execute_tests_concurrently(session.id, test_cases)
            
            # Update session statistics
            await self._update_session_statistics(session, test_executions)
            
            # Calculate coverage metrics
            await self._calculate_coverage_metrics(session, test_executions)
            
            logger.info(f"Completed execution session {session.id}: {session.passed_tests} passed, {session.failed_tests} failed, {session.error_tests} errors")
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to execute test session: {str(e)}")
            if 'session' in locals():
                session.completed_at = datetime.utcnow()
                session.duration_seconds = (session.completed_at - session.started_at).total_seconds()
                self.db.commit()
            raise TestExecutionError(f"Session execution failed: {str(e)}")

    async def _execute_tests_concurrently(
        self,
        session_id: int,
        test_cases: List[TestCase]
    ) -> List[TestExecution]:
        """Execute test cases concurrently with controlled parallelism."""
        
        test_executions = []
        semaphore = asyncio.Semaphore(self.max_concurrent_tests)
        
        async def execute_single_test(test_case: TestCase) -> TestExecution:
            async with semaphore:
                return await self._execute_single_test(session_id, test_case)
        
        # Create tasks for all test cases
        tasks = [execute_single_test(test_case) for test_case in test_cases]
        
        # Execute with progress tracking
        for future in asyncio.as_completed(tasks):
            try:
                execution = await future
                test_executions.append(execution)
                logger.debug(f"Completed test execution {execution.id}: {execution.status.value}")
            except Exception as e:
                logger.error(f"Test execution failed: {str(e)}")
        
        return test_executions

    async def _execute_single_test(
        self,
        session_id: int,
        test_case: TestCase
    ) -> TestExecution:
        """Execute a single test case and record results."""
        
        # Create execution record
        execution = TestExecution(
            session_id=session_id,
            test_case_id=test_case.id,
            status=TestStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        
        try:
            logger.debug(f"Executing test case {test_case.id}: {test_case.name}")
            
            # Get API specification for base URL
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == test_case.api_spec_id
            ).first()
            
            if not api_spec:
                raise TestExecutionError(f"API specification not found for test case {test_case.id}")
            
            # Prepare request
            request_data = self._prepare_request(api_spec, test_case)
            
            # Execute the HTTP request
            start_time = time.time()
            response = await self._make_http_request(request_data)
            end_time = time.time()
            
            execution.response_time_ms = (end_time - start_time) * 1000
            execution.response_code = response.status_code
            execution.response_headers = dict(response.headers)
            
            # Try to parse response body as JSON, fallback to text
            try:
                execution.response_body = response.json()
            except ValueError:
                execution.response_body = {"text": response.text, "content_type": response.headers.get('Content-Type', 'unknown')}
            
            # Run assertions
            assertion_results = self._run_assertions(test_case, response)
            execution.assertion_results = assertion_results
            
            # Determine status based on assertions
            if all(result["passed"] for result in assertion_results.values()):
                execution.status = TestStatus.PASSED
            else:
                execution.status = TestStatus.FAILED
            
            # Calculate coverage contribution
            execution.coverage_contribution = self._calculate_coverage_contribution(test_case, response)
            
        except asyncio.TimeoutError:
            execution.status = TestStatus.ERROR
            execution.error_message = f"Test execution timeout after {self.timeout_seconds} seconds"
            logger.error(f"Test case {test_case.id} timed out")
            
        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.error_message = str(e)
            execution.error_traceback = traceback.format_exc()
            logger.error(f"Test case {test_case.id} failed with error: {str(e)}")
        
        finally:
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            # Update test case statistics
            await self._update_test_case_statistics(test_case, execution)
            
            self.db.commit()
            
        return execution

    def _prepare_request(
        self,
        api_spec: APISpecification,
        test_case: TestCase
    ) -> Dict[str, Any]:
        """Prepare HTTP request from test case data."""
        
        test_data = test_case.test_data or {}
        base_url = api_spec.base_url.rstrip('/')
        endpoint = test_case.endpoint
        
        # Construct full URL
        url = f"{base_url}{endpoint}"
        
        # Handle path parameters
        path_params = test_data.get("path_params", {})
        for param_name, param_value in path_params.items():
            url = url.replace(f"{{{param_name}}}", str(param_value))
        
        # Prepare request components
        request_data = {
            "url": url,
            "method": test_case.method.upper(),
            "headers": test_data.get("headers", {}),
            "params": test_data.get("query_params", {}),
            "timeout": self.timeout_seconds
        }
        
        # Add request body for applicable methods
        if test_case.method.upper() in ["POST", "PUT", "PATCH"]:
            body = test_data.get("body", {})
            if body:
                if isinstance(body, dict):
                    request_data["json"] = body
                    if "Content-Type" not in request_data["headers"]:
                        request_data["headers"]["Content-Type"] = "application/json"
                else:
                    request_data["data"] = body
        
        return request_data

    async def _make_http_request(self, request_data: Dict[str, Any]) -> requests.Response:
        """Make the actual HTTP request asynchronously."""
        
        loop = asyncio.get_event_loop()
        
        # Run the request in a thread pool to avoid blocking
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                self.session.request,
                method=request_data["method"],
                url=request_data["url"],
                headers=request_data.get("headers", {}),
                params=request_data.get("params", {}),
                json=request_data.get("json"),
                data=request_data.get("data"),
                timeout=request_data["timeout"]
            )
            
            response = await loop.run_in_executor(None, lambda: future.result())
            return response

    def _run_assertions(
        self,
        test_case: TestCase,
        response: requests.Response
    ) -> Dict[str, Any]:
        """Run test assertions and return results."""
        
        assertion_results = {}
        assertions = test_case.assertions or []
        
        # Prepare assertion context
        context = {
            "response": response,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "json": None,
            "text": response.text
        }
        
        # Try to parse JSON response
        try:
            context["json"] = response.json()
        except ValueError:
            pass
        
        # Run each assertion
        for i, assertion in enumerate(assertions):
            assertion_key = f"assertion_{i}"
            
            try:
                # Simple assertion evaluation (can be enhanced with more sophisticated parsing)
                result = self._evaluate_assertion(assertion, context)
                assertion_results[assertion_key] = {
                    "assertion": assertion,
                    "passed": result,
                    "error": None
                }
            except Exception as e:
                assertion_results[assertion_key] = {
                    "assertion": assertion,
                    "passed": False,
                    "error": str(e)
                }
        
        # Add default status code assertion if not present
        if not any("status_code" in assertion for assertion in assertions):
            expected_status = test_case.expected_response.get("status_code", 200) if test_case.expected_response else 200
            assertion_results["default_status"] = {
                "assertion": f"status_code == {expected_status}",
                "passed": response.status_code == expected_status,
                "error": None
            }
        
        return assertion_results

    def _evaluate_assertion(self, assertion: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single assertion safely."""
        
        # Replace common patterns for safety
        safe_assertion = assertion.replace("response.status_code", "status_code")
        safe_assertion = safe_assertion.replace("response.json()", "json")
        safe_assertion = safe_assertion.replace("response.text", "text")
        safe_assertion = safe_assertion.replace("response.headers", "headers")
        
        # Define safe evaluation environment
        safe_env = {
            "status_code": context["status_code"],
            "json": context["json"],
            "text": context["text"],
            "headers": context["headers"],
            # Add common utility functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool
        }
        
        # Evaluate the assertion
        try:
            return eval(safe_assertion, {"__builtins__": {}}, safe_env)
        except Exception as e:
            logger.error(f"Assertion evaluation failed: {assertion} - {str(e)}")
            return False

    def _calculate_coverage_contribution(
        self,
        test_case: TestCase,
        response: requests.Response
    ) -> Dict[str, Any]:
        """Calculate what this test execution contributed to coverage."""
        
        return {
            "endpoint": test_case.endpoint,
            "method": test_case.method,
            "status_code": response.status_code,
            "response_time_category": self._categorize_response_time(response.elapsed.total_seconds() * 1000),
            "headers_tested": list(test_case.test_data.get("headers", {}).keys()) if test_case.test_data else [],
            "query_params_tested": list(test_case.test_data.get("query_params", {}).keys()) if test_case.test_data else [],
            "body_fields_tested": self._extract_body_fields(test_case.test_data.get("body", {})) if test_case.test_data else []
        }

    def _categorize_response_time(self, response_time_ms: float) -> str:
        """Categorize response time for coverage analysis."""
        if response_time_ms < 100:
            return "fast"
        elif response_time_ms < 500:
            return "medium"
        elif response_time_ms < 2000:
            return "slow"
        else:
            return "very_slow"

    def _extract_body_fields(self, body: Any) -> List[str]:
        """Extract field names from request body for coverage tracking."""
        if isinstance(body, dict):
            return list(body.keys())
        elif isinstance(body, str):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    return list(parsed.keys())
            except json.JSONDecodeError:
                pass
        return []

    async def _update_test_case_statistics(
        self,
        test_case: TestCase,
        execution: TestExecution
    ):
        """Update test case statistics based on execution results."""
        
        try:
            test_case.selection_count += 1
            
            # Update success rate
            total_executions = self.db.query(TestExecution).filter(
                TestExecution.test_case_id == test_case.id
            ).count()
            
            passed_executions = self.db.query(TestExecution).filter(
                TestExecution.test_case_id == test_case.id,
                TestExecution.status == TestStatus.PASSED
            ).count()
            
            test_case.success_rate = passed_executions / total_executions if total_executions > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to update test case statistics: {str(e)}")

    async def _update_session_statistics(
        self,
        session: ExecutionSession,
        test_executions: List[TestExecution]
    ):
        """Update session statistics based on execution results."""
        
        try:
            session.passed_tests = sum(1 for ex in test_executions if ex.status == TestStatus.PASSED)
            session.failed_tests = sum(1 for ex in test_executions if ex.status == TestStatus.FAILED)
            session.error_tests = sum(1 for ex in test_executions if ex.status == TestStatus.ERROR)
            session.skipped_tests = sum(1 for ex in test_executions if ex.status == TestStatus.SKIPPED)
            
            session.completed_at = datetime.utcnow()
            session.duration_seconds = (session.completed_at - session.started_at).total_seconds()
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session statistics: {str(e)}")

    async def _calculate_coverage_metrics(
        self,
        session: ExecutionSession,
        test_executions: List[TestExecution]
    ):
        """Calculate and store coverage metrics for the session."""
        
        try:
            # Get API specification
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == session.api_spec_id
            ).first()
            
            if not api_spec or not api_spec.parsed_endpoints:
                logger.warning(f"Cannot calculate coverage - API spec or endpoints not found")
                return
            
            # Calculate endpoint coverage
            all_endpoints = set()
            tested_endpoints = set()
            
            for path, methods in api_spec.parsed_endpoints.items():
                for method in methods.keys():
                    all_endpoints.add(f"{method.upper()} {path}")
            
            for execution in test_executions:
                if execution.status in [TestStatus.PASSED, TestStatus.FAILED]:
                    tested_endpoints.add(f"{execution.test_case.method} {execution.test_case.endpoint}")
            
            endpoint_coverage_pct = len(tested_endpoints) / len(all_endpoints) * 100 if all_endpoints else 0
            
            # Calculate response code coverage
            tested_status_codes = set(ex.response_code for ex in test_executions if ex.response_code)
            expected_status_codes = {200, 201, 400, 401, 403, 404, 422, 500}  # Common codes
            
            status_code_coverage_pct = len(tested_status_codes & expected_status_codes) / len(expected_status_codes) * 100
            
            # Detect bugs (failures and errors)
            bugs_found = sum(1 for ex in test_executions if ex.status in [TestStatus.FAILED, TestStatus.ERROR])
            
            # Create coverage metrics record
            coverage_metrics = CoverageMetrics(
                session_id=session.id,
                endpoint_coverage_pct=endpoint_coverage_pct,
                method_coverage_pct=endpoint_coverage_pct,  # Same as endpoint for now
                response_code_coverage_pct=status_code_coverage_pct,
                parameter_coverage_pct=0,  # To be implemented
                covered_endpoints=list(tested_endpoints),
                missed_endpoints=list(all_endpoints - tested_endpoints),
                bugs_found=bugs_found,
                new_bugs_found=bugs_found,  # All bugs are "new" for now
                quality_score=endpoint_coverage_pct * 0.4 + status_code_coverage_pct * 0.3 + (100 - bugs_found) * 0.3
            )
            
            self.db.add(coverage_metrics)
            
            # Update session coverage information
            session.endpoint_coverage = {
                "total": len(all_endpoints),
                "covered": len(tested_endpoints),
                "percentage": endpoint_coverage_pct
            }
            
            session.response_code_coverage = {
                "tested_codes": list(tested_status_codes),
                "percentage": status_code_coverage_pct
            }
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to calculate coverage metrics: {str(e)}")

    async def execute_single_test_case(
        self,
        test_case_id: int,
        session_name: Optional[str] = None
    ) -> TestExecution:
        """Execute a single test case and return the result."""
        
        try:
            test_case = self.db.query(TestCase).filter(TestCase.id == test_case_id).first()
            if not test_case:
                raise TestExecutionError(f"Test case with ID {test_case_id} not found")
            
            # Create a temporary session for this single test
            session = await self.execute_test_session(
                api_spec_id=test_case.api_spec_id,
                test_case_ids=[test_case_id],
                session_name=session_name or f"Single Test: {test_case.name}",
                trigger="manual"
            )
            
            # Return the test execution
            execution = self.db.query(TestExecution).filter(
                TestExecution.session_id == session.id,
                TestExecution.test_case_id == test_case_id
            ).first()
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute single test case: {str(e)}")
            raise TestExecutionError(f"Single test execution failed: {str(e)}")

    def get_execution_history(
        self,
        api_spec_id: Optional[int] = None,
        limit: int = 50
    ) -> List[ExecutionSession]:
        """Get execution session history."""
        
        try:
            query = self.db.query(ExecutionSession)
            if api_spec_id:
                query = query.filter(ExecutionSession.api_spec_id == api_spec_id)
            
            return query.order_by(ExecutionSession.created_at.desc()).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Failed to get execution history: {str(e)}")
            return []

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'db'):
            self.db.close()
        if hasattr(self, 'session'):
            self.session.close()
