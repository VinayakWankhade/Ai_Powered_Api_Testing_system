"""
Self-healing mechanism for automatic failure analysis and test repair.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from ..database.models import (
    TestCase, TestExecution, TestStatus, ExecutionSession,
    APISpecification, TestType
)
from ..database.connection import get_db_session
from ..ai.rag_system import RAGSystem
from ..execution.test_executor import APITestExecutor
from ..utils.logger import get_logger

logger = get_logger(__name__)

class HealingStrategy(Enum):
    """Different strategies for healing failed tests."""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    ASSERTION_RELAXATION = "assertion_relaxation"
    REQUEST_MODIFICATION = "request_modification"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"
    AUTHENTICATION_FIX = "authentication_fix"
    DATA_FORMAT_FIX = "data_format_fix"

class TestHealingError(Exception):
    """Custom exception for test healing errors."""
    pass

class AITestHealer:
    """
    AI-powered self-healing mechanism for automatic test failure analysis and repair.
    """

    def __init__(self):
        self.db = get_db_session()
        self.rag_system = RAGSystem()
        self.test_executor = APITestExecutor()
        
        # Initialize LLM for healing analysis
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.3,  # Lower temperature for more consistent healing
            max_tokens=2000
        )
        
        logger.info("AI Test Healer initialized")

    async def heal_failed_tests(
        self,
        session_id: int,
        max_healing_attempts: int = 3,
        auto_revalidate: bool = True
    ) -> Dict[str, Any]:
        """
        Heal all failed tests in a session.
        
        Args:
            session_id: Execution session ID
            max_healing_attempts: Maximum healing attempts per test
            auto_revalidate: Whether to automatically re-execute healed tests
            
        Returns:
            Healing results summary
        """
        try:
            # Get failed test executions from the session
            failed_executions = self.db.query(TestExecution).filter(
                TestExecution.session_id == session_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR])
            ).all()
            
            if not failed_executions:
                return {"message": "No failed tests found in session", "healed_count": 0}
            
            logger.info(f"Starting healing process for {len(failed_executions)} failed tests")
            
            healing_results = {
                "total_failed_tests": len(failed_executions),
                "healing_attempts": 0,
                "successfully_healed": 0,
                "healing_failed": 0,
                "revalidation_results": [],
                "detailed_results": []
            }
            
            # Heal each failed test
            for execution in failed_executions:
                try:
                    test_healing_result = await self._heal_single_test(
                        execution, max_healing_attempts
                    )
                    
                    healing_results["healing_attempts"] += test_healing_result["attempts"]
                    
                    if test_healing_result["success"]:
                        healing_results["successfully_healed"] += 1
                        
                        # Re-validate if requested
                        if auto_revalidate:
                            revalidation_result = await self._revalidate_healed_test(
                                test_healing_result["healed_test"]
                            )
                            healing_results["revalidation_results"].append(revalidation_result)
                    else:
                        healing_results["healing_failed"] += 1
                    
                    healing_results["detailed_results"].append(test_healing_result)
                    
                except Exception as e:
                    logger.error(f"Failed to heal test {execution.test_case_id}: {str(e)}")
                    healing_results["healing_failed"] += 1
            
            logger.info(f"Healing complete: {healing_results['successfully_healed']}/{len(failed_executions)} tests healed")
            return healing_results
            
        except Exception as e:
            logger.error(f"Healing process failed: {str(e)}")
            raise TestHealingError(f"Healing failed: {str(e)}")

    async def _heal_single_test(
        self,
        failed_execution: TestExecution,
        max_attempts: int
    ) -> Dict[str, Any]:
        """Heal a single failed test case."""
        
        healing_result = {
            "test_case_id": failed_execution.test_case_id,
            "original_failure": {
                "status": failed_execution.status.value,
                "error_message": failed_execution.error_message,
                "response_code": failed_execution.response_code
            },
            "healing_strategies_tried": [],
            "attempts": 0,
            "success": False,
            "healed_test": None,
            "healing_log": []
        }
        
        test_case = failed_execution.test_case
        
        for attempt in range(max_attempts):
            healing_result["attempts"] += 1
            
            try:
                # Analyze the failure
                failure_analysis = await self._analyze_failure(failed_execution)
                healing_result["healing_log"].append({
                    "attempt": attempt + 1,
                    "analysis": failure_analysis
                })
                
                # Determine healing strategy
                strategy = self._determine_healing_strategy(failure_analysis)
                healing_result["healing_strategies_tried"].append(strategy.value)
                
                # Apply healing strategy
                healed_test_case = await self._apply_healing_strategy(
                    test_case, failed_execution, strategy, failure_analysis
                )
                
                if healed_test_case:
                    # Update healing metadata
                    healed_test_case.required_healing = True
                    healed_test_case.healing_attempts += 1
                    healed_test_case.healing_log = healed_test_case.healing_log or []
                    healed_test_case.healing_log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "strategy": strategy.value,
                        "original_failure": failed_execution.error_message,
                        "changes_made": failure_analysis.get("suggested_changes", [])
                    })
                    
                    self.db.commit()
                    
                    healing_result["success"] = True
                    healing_result["healed_test"] = healed_test_case
                    healing_result["healing_log"].append({
                        "attempt": attempt + 1,
                        "result": "SUCCESS",
                        "strategy": strategy.value
                    })
                    
                    logger.info(f"Successfully healed test {test_case.id} using {strategy.value}")
                    break
                
            except Exception as e:
                healing_result["healing_log"].append({
                    "attempt": attempt + 1,
                    "result": "FAILED",
                    "error": str(e)
                })
                logger.error(f"Healing attempt {attempt + 1} failed for test {test_case.id}: {str(e)}")
        
        if not healing_result["success"]:
            # Mark test as unhealable
            test_case.healed_successfully = False
            self.db.commit()
        
        return healing_result

    async def _analyze_failure(self, failed_execution: TestExecution) -> Dict[str, Any]:
        """Analyze the failure to understand root cause and suggest fixes."""
        
        test_case = failed_execution.test_case
        
        # Retrieve relevant documentation and error patterns
        error_context = await self._get_error_context(
            failed_execution.error_message or "Unknown error",
            test_case.api_spec_id
        )
        
        # Create analysis prompt
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert API test analyst. Analyze the failed test execution and provide detailed insights about the failure cause and potential fixes.

Your analysis should include:
1. Root cause identification
2. Failure category (authentication, validation, timeout, server error, etc.)
3. Specific suggested changes to fix the test
4. Confidence level in your analysis (1-10)
5. Alternative approaches if primary fix doesn't work

Be precise and actionable in your suggestions."""),
            
            HumanMessage(content=f"""
Test Case Details:
- Name: {test_case.name}
- Endpoint: {test_case.method} {test_case.endpoint}
- Test Type: {test_case.test_type.value}
- Test Data: {json.dumps(test_case.test_data, indent=2)}
- Expected Response: {json.dumps(test_case.expected_response, indent=2)}
- Assertions: {test_case.assertions}

Failure Details:
- Status: {failed_execution.status.value}
- Error Message: {failed_execution.error_message or 'No error message'}
- Response Code: {failed_execution.response_code or 'No response code'}
- Response Body: {json.dumps(failed_execution.response_body, indent=2) if failed_execution.response_body else 'No response body'}
- Response Headers: {json.dumps(failed_execution.response_headers, indent=2) if failed_execution.response_headers else 'No response headers'}

Related Documentation:
{error_context}

Please provide a comprehensive analysis and actionable healing strategy.""")
        ])
        
        # Get LLM analysis
        messages = analysis_prompt.format_messages()
        response = await self.llm.agenerate([messages])
        analysis_text = response.generations[0][0].text
        
        # Parse the analysis (simplified - in production, use structured output)
        analysis = {
            "raw_analysis": analysis_text,
            "failure_category": self._extract_failure_category(analysis_text),
            "confidence_level": self._extract_confidence_level(analysis_text),
            "suggested_changes": self._extract_suggested_changes(analysis_text),
            "root_cause": self._extract_root_cause(analysis_text)
        }
        
        return analysis

    async def _get_error_context(self, error_message: str, api_spec_id: int) -> str:
        """Get relevant context for error analysis using RAG."""
        
        try:
            # Search for error-related documentation
            error_docs = self.rag_system.retrieve_error_handling_docs(
                error_type=self._categorize_error(error_message),
                api_spec_id=api_spec_id
            )
            
            # Search for similar test patterns
            similar_patterns = self.rag_system.retrieve_similar_test_patterns(
                test_description=f"failed test {error_message}",
                api_spec_id=api_spec_id
            )
            
            context_parts = []
            
            if error_docs:
                context_parts.append("Error Handling Documentation:")
                for doc in error_docs[:2]:  # Top 2 most relevant
                    context_parts.append(f"- {doc['content'][:300]}...")
            
            if similar_patterns:
                context_parts.append("\nSimilar Test Patterns:")
                for pattern in similar_patterns[:2]:
                    context_parts.append(f"- {pattern['content'][:300]}...")
            
            return "\n".join(context_parts) if context_parts else "No relevant context found."
            
        except Exception as e:
            logger.error(f"Failed to retrieve error context: {str(e)}")
            return "Context retrieval failed."

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error type from message."""
        error_message_lower = error_message.lower()
        
        if any(word in error_message_lower for word in ['auth', 'unauthorized', 'forbidden', 'token']):
            return "authentication"
        elif any(word in error_message_lower for word in ['timeout', 'time out']):
            return "timeout"
        elif any(word in error_message_lower for word in ['validation', 'invalid', 'bad request', '400']):
            return "validation"
        elif any(word in error_message_lower for word in ['500', 'internal', 'server error']):
            return "server_error"
        elif any(word in error_message_lower for word in ['not found', '404']):
            return "not_found"
        else:
            return "general"

    def _determine_healing_strategy(self, failure_analysis: Dict[str, Any]) -> HealingStrategy:
        """Determine the best healing strategy based on failure analysis."""
        
        failure_category = failure_analysis.get("failure_category", "").lower()
        
        if "authentication" in failure_category or "auth" in failure_category:
            return HealingStrategy.AUTHENTICATION_FIX
        elif "timeout" in failure_category:
            return HealingStrategy.TIMEOUT_ADJUSTMENT
        elif "validation" in failure_category or "format" in failure_category:
            return HealingStrategy.DATA_FORMAT_FIX
        elif "parameter" in failure_category:
            return HealingStrategy.PARAMETER_ADJUSTMENT
        elif "assertion" in failure_category:
            return HealingStrategy.ASSERTION_RELAXATION
        else:
            return HealingStrategy.REQUEST_MODIFICATION

    async def _apply_healing_strategy(
        self,
        test_case: TestCase,
        failed_execution: TestExecution,
        strategy: HealingStrategy,
        failure_analysis: Dict[str, Any]
    ) -> Optional[TestCase]:
        """Apply the healing strategy to create a fixed test case."""
        
        try:
            if strategy == HealingStrategy.AUTHENTICATION_FIX:
                return await self._heal_authentication_issue(test_case, failure_analysis)
            
            elif strategy == HealingStrategy.TIMEOUT_ADJUSTMENT:
                return await self._heal_timeout_issue(test_case, failure_analysis)
            
            elif strategy == HealingStrategy.DATA_FORMAT_FIX:
                return await self._heal_data_format_issue(test_case, failed_execution, failure_analysis)
            
            elif strategy == HealingStrategy.PARAMETER_ADJUSTMENT:
                return await self._heal_parameter_issue(test_case, failed_execution, failure_analysis)
            
            elif strategy == HealingStrategy.ASSERTION_RELAXATION:
                return await self._heal_assertion_issue(test_case, failed_execution, failure_analysis)
            
            elif strategy == HealingStrategy.REQUEST_MODIFICATION:
                return await self._heal_request_issue(test_case, failed_execution, failure_analysis)
            
            else:
                logger.warning(f"Unknown healing strategy: {strategy}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to apply healing strategy {strategy}: {str(e)}")
            return None

    async def _heal_authentication_issue(
        self,
        test_case: TestCase,
        failure_analysis: Dict[str, Any]
    ) -> TestCase:
        """Heal authentication-related issues."""
        
        # Clone the test case
        healed_test = self._clone_test_case(test_case)
        healed_test.name = f"{test_case.name} (Healed - Auth)"
        
        # Add or update authentication headers
        test_data = healed_test.test_data or {}
        headers = test_data.get("headers", {})
        
        # Add common authentication headers if missing
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer <token>"
        
        if "X-API-Key" not in headers and "Api-Key" not in headers:
            headers["X-API-Key"] = "<api_key>"
        
        test_data["headers"] = headers
        healed_test.test_data = test_data
        
        self.db.add(healed_test)
        return healed_test

    async def _heal_timeout_issue(
        self,
        test_case: TestCase,
        failure_analysis: Dict[str, Any]
    ) -> TestCase:
        """Heal timeout-related issues."""
        
        healed_test = self._clone_test_case(test_case)
        healed_test.name = f"{test_case.name} (Healed - Timeout)"
        
        # Increase timeout expectations
        expected_response = healed_test.expected_response or {}
        expected_response["max_response_time"] = 30000  # 30 seconds
        healed_test.expected_response = expected_response
        
        # Relax time-related assertions
        if healed_test.assertions:
            relaxed_assertions = []
            for assertion in healed_test.assertions:
                if "response_time" in assertion or "elapsed" in assertion:
                    # Make timeout assertions more lenient
                    relaxed_assertion = assertion.replace("< 1000", "< 30000").replace("< 5000", "< 30000")
                    relaxed_assertions.append(relaxed_assertion)
                else:
                    relaxed_assertions.append(assertion)
            healed_test.assertions = relaxed_assertions
        
        self.db.add(healed_test)
        return healed_test

    async def _heal_data_format_issue(
        self,
        test_case: TestCase,
        failed_execution: TestExecution,
        failure_analysis: Dict[str, Any]
    ) -> TestCase:
        """Heal data format-related issues."""
        
        healed_test = self._clone_test_case(test_case)
        healed_test.name = f"{test_case.name} (Healed - Format)"
        
        # Try to fix common data format issues
        test_data = healed_test.test_data or {}
        
        if "body" in test_data and test_data["body"]:
            body = test_data["body"]
            
            # Convert string numbers to actual numbers
            if isinstance(body, dict):
                for key, value in body.items():
                    if isinstance(value, str) and value.isdigit():
                        body[key] = int(value)
                    elif isinstance(value, str):
                        try:
                            body[key] = float(value)
                        except ValueError:
                            pass
            
            test_data["body"] = body
        
        # Ensure proper content-type header
        headers = test_data.get("headers", {})
        if "Content-Type" not in headers and test_case.method.upper() in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"
        test_data["headers"] = headers
        
        healed_test.test_data = test_data
        self.db.add(healed_test)
        return healed_test

    async def _heal_parameter_issue(
        self,
        test_case: TestCase,
        failed_execution: TestExecution,
        failure_analysis: Dict[str, Any]
    ) -> TestCase:
        """Heal parameter-related issues."""
        
        healed_test = self._clone_test_case(test_case)
        healed_test.name = f"{test_case.name} (Healed - Params)"
        
        # Remove potentially problematic parameters or provide defaults
        test_data = healed_test.test_data or {}
        
        # Fix query parameters
        if "query_params" in test_data:
            query_params = test_data["query_params"]
            # Remove empty or null parameters
            cleaned_params = {k: v for k, v in query_params.items() if v is not None and v != ""}
            test_data["query_params"] = cleaned_params
        
        # Fix body parameters
        if "body" in test_data and isinstance(test_data["body"], dict):
            body = test_data["body"]
            # Remove null/empty fields that might cause validation errors
            cleaned_body = {k: v for k, v in body.items() if v is not None and v != ""}
            test_data["body"] = cleaned_body
        
        healed_test.test_data = test_data
        self.db.add(healed_test)
        return healed_test

    async def _heal_assertion_issue(
        self,
        test_case: TestCase,
        failed_execution: TestExecution,
        failure_analysis: Dict[str, Any]
    ) -> TestCase:
        """Heal assertion-related issues."""
        
        healed_test = self._clone_test_case(test_case)
        healed_test.name = f"{test_case.name} (Healed - Assertions)"
        
        # Relax or modify assertions based on actual response
        if healed_test.assertions and failed_execution.response_body:
            relaxed_assertions = []
            
            for assertion in healed_test.assertions:
                # Make assertions more flexible
                if "==" in assertion:
                    # Change exact matches to type checks or existence checks
                    relaxed_assertion = assertion.replace("==", "is not None and")
                    relaxed_assertions.append(relaxed_assertion)
                elif "status_code" in assertion:
                    # Accept any 2xx status code for success
                    relaxed_assertions.append("200 <= status_code < 300")
                else:
                    relaxed_assertions.append(assertion)
            
            healed_test.assertions = relaxed_assertions
        
        self.db.add(healed_test)
        return healed_test

    async def _heal_request_issue(
        self,
        test_case: TestCase,
        failed_execution: TestExecution,
        failure_analysis: Dict[str, Any]
    ) -> TestCase:
        """Heal general request-related issues."""
        
        healed_test = self._clone_test_case(test_case)
        healed_test.name = f"{test_case.name} (Healed - Request)"
        
        # Apply general fixes
        test_data = healed_test.test_data or {}
        
        # Ensure proper headers
        headers = test_data.get("headers", {})
        headers.update({
            "Accept": "application/json",
            "User-Agent": "AI-API-Testing-Framework/1.0"
        })
        
        if test_case.method.upper() in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"
        
        test_data["headers"] = headers
        healed_test.test_data = test_data
        
        self.db.add(healed_test)
        return healed_test

    def _clone_test_case(self, original: TestCase) -> TestCase:
        """Create a copy of a test case for healing."""
        
        return TestCase(
            api_spec_id=original.api_spec_id,
            name=original.name,
            description=f"Healed version of: {original.description}",
            test_type=original.test_type,
            endpoint=original.endpoint,
            method=original.method,
            test_data=original.test_data.copy() if original.test_data else {},
            expected_response=original.expected_response.copy() if original.expected_response else {},
            assertions=original.assertions.copy() if original.assertions else [],
            generated_by_llm=True,
            generation_context=original.generation_context,
            required_healing=False,
            healing_attempts=0,
            healed_successfully=False
        )

    async def _revalidate_healed_test(self, healed_test: TestCase) -> Dict[str, Any]:
        """Re-execute healed test to validate the fix."""
        
        try:
            execution = await self.test_executor.execute_single_test_case(
                healed_test.id,
                session_name=f"Healing Validation: {healed_test.name}"
            )
            
            success = execution.status == TestStatus.PASSED
            if success:
                healed_test.healed_successfully = True
                self.db.commit()
            
            return {
                "test_case_id": healed_test.id,
                "revalidation_success": success,
                "execution_status": execution.status.value,
                "response_code": execution.response_code,
                "error_message": execution.error_message
            }
            
        except Exception as e:
            logger.error(f"Revalidation failed for healed test {healed_test.id}: {str(e)}")
            return {
                "test_case_id": healed_test.id,
                "revalidation_success": False,
                "error": str(e)
            }

    def _extract_failure_category(self, analysis_text: str) -> str:
        """Extract failure category from analysis text."""
        analysis_lower = analysis_text.lower()
        
        categories = [
            "authentication", "timeout", "validation", "parameter",
            "assertion", "server_error", "format", "connection"
        ]
        
        for category in categories:
            if category in analysis_lower:
                return category
        
        return "general"

    def _extract_confidence_level(self, analysis_text: str) -> float:
        """Extract confidence level from analysis text."""
        import re
        
        # Look for confidence patterns
        patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*[/]?\s*10\s*confidence",
            r"(\d+)%\s*confident"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis_text.lower())
            if match:
                value = float(match.group(1))
                return value if value <= 10 else value / 10
        
        return 5.0  # Default moderate confidence

    def _extract_suggested_changes(self, analysis_text: str) -> List[str]:
        """Extract suggested changes from analysis text."""
        # This is a simplified extraction - in production, use more sophisticated parsing
        lines = analysis_text.split('\n')
        suggestions = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['suggest', 'fix', 'change', 'modify', 'update']):
                suggestions.append(line)
        
        return suggestions[:5]  # Return top 5 suggestions

    def _extract_root_cause(self, analysis_text: str) -> str:
        """Extract root cause from analysis text."""
        lines = analysis_text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['root cause', 'cause:', 'reason:']):
                return line.strip()
        
        return "Root cause analysis not available"

    def get_healing_statistics(self, api_spec_id: Optional[int] = None) -> Dict[str, Any]:
        """Get healing statistics and insights."""
        
        try:
            query = self.db.query(TestCase).filter(TestCase.required_healing == True)
            if api_spec_id:
                query = query.filter(TestCase.api_spec_id == api_spec_id)
            
            healed_tests = query.all()
            
            if not healed_tests:
                return {"message": "No healing data available"}
            
            total_healed = len(healed_tests)
            successfully_healed = len([t for t in healed_tests if t.healed_successfully])
            
            # Analyze healing patterns
            healing_strategies = {}
            for test in healed_tests:
                if test.healing_log:
                    for log_entry in test.healing_log:
                        strategy = log_entry.get("strategy", "unknown")
                        healing_strategies[strategy] = healing_strategies.get(strategy, 0) + 1
            
            return {
                "total_tests_healed": total_healed,
                "successfully_healed": successfully_healed,
                "healing_success_rate": (successfully_healed / total_healed * 100) if total_healed > 0 else 0,
                "healing_strategies_used": healing_strategies,
                "average_healing_attempts": sum(t.healing_attempts for t in healed_tests) / total_healed if total_healed > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get healing statistics: {str(e)}")
            return {"error": str(e)}

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'db'):
            self.db.close()
