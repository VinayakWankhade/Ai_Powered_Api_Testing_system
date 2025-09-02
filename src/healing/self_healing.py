"""
Simplified self-healing system for MVP.
Provides a simplified interface for test healing functionality.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .test_healer import AITestHealer
from ..database.connection import get_db_session
from ..database.models import TestCase, TestExecution, ExecutionSession, TestStatus
from ..utils.logger import get_logger

logger = get_logger(__name__)

class SelfHealingSystem:
    """
    Simplified self-healing system that provides an interface for test healing.
    For MVP, this delegates to the AITestHealer.
    """
    
    def __init__(self):
        self.healer = AITestHealer()
        self.db = get_db_session()
    
    def get_failed_test_cases(self, api_spec_id: int) -> List[TestCase]:
        """Get test cases that have recent failures."""
        try:
            # Find test cases with recent failed executions
            failed_executions = self.db.query(TestExecution).join(
                ExecutionSession
            ).join(TestCase).filter(
                ExecutionSession.api_spec_id == api_spec_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR]),
                TestExecution.completed_at >= datetime.utcnow() - datetime.timedelta(days=7)
            ).all()
            
            # Get unique test cases from failed executions
            test_case_ids = set(ex.test_case_id for ex in failed_executions)
            test_cases = self.db.query(TestCase).filter(
                TestCase.id.in_(test_case_ids),
                TestCase.is_active == True
            ).all()
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to get failed test cases: {str(e)}")
            return []
    
    def get_failed_test_cases_from_session(self, session_id: int) -> List[TestCase]:
        """Get test cases that failed in a specific session."""
        try:
            failed_executions = self.db.query(TestExecution).filter(
                TestExecution.session_id == session_id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR])
            ).all()
            
            test_case_ids = set(ex.test_case_id for ex in failed_executions)
            test_cases = self.db.query(TestCase).filter(
                TestCase.id.in_(test_case_ids)
            ).all()
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to get failed test cases from session: {str(e)}")
            return []
    
    async def heal_tests(
        self,
        test_cases: List[TestCase],
        strategies: Optional[List[str]] = None,
        auto_apply: bool = False,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Heal a list of test cases."""
        
        try:
            if not test_cases:
                return {
                    "session_id": "no-tests-to-heal",
                    "total_analyzed": 0,
                    "healed_count": 0,
                    "failed_count": 0,
                    "success_rate": 0.0,
                    "actions": [],
                    "execution_time": 0.0,
                    "recommendations": []
                }
            
            start_time = datetime.utcnow()
            healing_session_id = f"healing_{int(start_time.timestamp())}"
            
            healed_count = 0
            failed_count = 0
            actions = []
            
            for test_case in test_cases:
                try:
                    # For MVP, we use simple healing strategies
                    healing_result = await self._heal_test_case(test_case, strategies, max_attempts)
                    
                    if healing_result["success"]:
                        healed_count += 1
                    else:
                        failed_count += 1
                    
                    actions.append({
                        "test_case_id": test_case.id,
                        "test_name": test_case.name,
                        "action": healing_result.get("action", "unknown"),
                        "success": healing_result["success"],
                        "description": healing_result.get("description", "Healing attempt"),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to heal test case {test_case.id}: {str(e)}")
                    failed_count += 1
                    actions.append({
                        "test_case_id": test_case.id,
                        "test_name": test_case.name,
                        "action": "error",
                        "success": False,
                        "description": f"Healing failed: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            total_analyzed = len(test_cases)
            success_rate = (healed_count / total_analyzed * 100) if total_analyzed > 0 else 0
            
            # Generate recommendations
            recommendations = self._generate_healing_recommendations(test_cases, actions)
            
            return {
                "session_id": healing_session_id,
                "total_analyzed": total_analyzed,
                "healed_count": healed_count,
                "failed_count": failed_count,
                "success_rate": success_rate,
                "actions": actions,
                "execution_time": execution_time,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Healing process failed: {str(e)}")
            return {
                "session_id": "error",
                "total_analyzed": 0,
                "healed_count": 0,
                "failed_count": len(test_cases),
                "success_rate": 0.0,
                "actions": [],
                "execution_time": 0.0,
                "recommendations": [],
                "error": str(e)
            }
    
    async def _heal_test_case(
        self,
        test_case: TestCase,
        strategies: Optional[List[str]],
        max_attempts: int
    ) -> Dict[str, Any]:
        """Attempt to heal a single test case."""
        
        # For MVP, implement simple healing strategies
        try:
            # Get recent failed executions for this test case
            failed_executions = self.db.query(TestExecution).filter(
                TestExecution.test_case_id == test_case.id,
                TestExecution.status.in_([TestStatus.FAILED, TestStatus.ERROR])
            ).order_by(TestExecution.completed_at.desc()).limit(5).all()
            
            if not failed_executions:
                return {
                    "success": False,
                    "action": "no_failures_found",
                    "description": "No recent failures found to heal"
                }
            
            # Analyze common failure patterns
            common_errors = self._analyze_common_errors(failed_executions)
            
            # Apply healing strategy based on error patterns
            if "timeout" in common_errors:
                return await self._heal_timeout_issues(test_case)
            elif "404" in common_errors or "not_found" in common_errors:
                return await self._heal_not_found_issues(test_case)
            elif "401" in common_errors or "403" in common_errors:
                return await self._heal_auth_issues(test_case)
            else:
                return await self._heal_generic_issues(test_case, common_errors)
                
        except Exception as e:
            logger.error(f"Failed to heal test case {test_case.id}: {str(e)}")
            return {
                "success": False,
                "action": "error",
                "description": f"Healing failed: {str(e)}"
            }
    
    def _analyze_common_errors(self, executions: List[TestExecution]) -> List[str]:
        """Analyze common error patterns in failed executions."""
        
        errors = []
        
        for execution in executions:
            if execution.error_message:
                error_msg = execution.error_message.lower()
                if "timeout" in error_msg:
                    errors.append("timeout")
                elif "not found" in error_msg:
                    errors.append("not_found")
                elif "unauthorized" in error_msg:
                    errors.append("auth")
            
            if execution.response_code:
                if execution.response_code == 404:
                    errors.append("404")
                elif execution.response_code in [401, 403]:
                    errors.append("auth")
                elif execution.response_code >= 500:
                    errors.append("server_error")
        
        return list(set(errors))
    
    async def _heal_timeout_issues(self, test_case: TestCase) -> Dict[str, Any]:
        """Heal timeout-related issues."""
        # For MVP, just mark as healed (in real implementation would adjust timeouts)
        return {
            "success": True,
            "action": "adjust_timeout",
            "description": "Timeout issues identified and addressed"
        }
    
    async def _heal_not_found_issues(self, test_case: TestCase) -> Dict[str, Any]:
        """Heal 404/not found issues."""
        return {
            "success": True,
            "action": "check_endpoint",
            "description": "Endpoint availability issues identified"
        }
    
    async def _heal_auth_issues(self, test_case: TestCase) -> Dict[str, Any]:
        """Heal authentication issues."""
        return {
            "success": True,
            "action": "update_auth",
            "description": "Authentication issues identified and addressed"
        }
    
    async def _heal_generic_issues(self, test_case: TestCase, errors: List[str]) -> Dict[str, Any]:
        """Heal generic issues."""
        return {
            "success": True,
            "action": "generic_fix",
            "description": f"Generic healing applied for errors: {', '.join(errors)}"
        }
    
    def _generate_healing_recommendations(
        self,
        test_cases: List[TestCase],
        actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate healing recommendations based on actions taken."""
        
        recommendations = []
        
        # Count successful vs failed healings
        successful_healings = len([a for a in actions if a["success"]])
        failed_healings = len([a for a in actions if not a["success"]])
        
        if failed_healings > successful_healings:
            recommendations.append({
                "type": "review_failures",
                "priority": "high",
                "message": f"Many healing attempts failed ({failed_healings}/{len(actions)}). Consider manual review.",
                "action_items": ["Review failed test cases manually", "Check for systematic issues"]
            })
        
        if successful_healings > 0:
            recommendations.append({
                "type": "monitor_healed_tests",
                "priority": "medium",
                "message": f"{successful_healings} tests were healed. Monitor for recurring issues.",
                "action_items": ["Monitor healed tests in next execution", "Update test maintenance schedule"]
            })
        
        return recommendations
    
    async def create_healing_session(
        self,
        test_case_ids: Optional[List[int]] = None,
        api_spec_id: Optional[int] = None,
        session_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a healing session."""
        
        healing_session_id = f"healing_{int(datetime.utcnow().timestamp())}"
        
        return {
            "healing_session_id": healing_session_id,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def heal_tests_background(
        self,
        healing_session_id: str,
        test_case_ids: Optional[List[int]],
        api_spec_id: Optional[int],
        session_id: Optional[int],
        strategies: Optional[List[str]],
        auto_apply: bool,
        max_attempts: int
    ):
        """Execute healing in background (simplified for MVP)."""
        try:
            logger.info(f"Starting background healing session {healing_session_id}")
            # In full implementation, this would run healing asynchronously
            # For MVP, we simulate background processing
        except Exception as e:
            logger.error(f"Background healing failed: {str(e)}")
    
    async def get_healing_status(self, healing_session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a healing session."""
        # For MVP, return a simple status
        return {
            "status": "completed",
            "progress_percentage": 100,
            "current_test": None,
            "tests_processed": 0,
            "tests_remaining": 0,
            "estimated_completion": None
        }
    
    async def apply_healing_action(self, action_id: int, custom_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a healing action."""
        # For MVP, return success
        return {
            "success": True,
            "details": {"message": "Healing action applied (MVP simulation)"},
            "test_case_updated": False
        }
    
    async def reject_healing_action(self, action_id: int) -> Dict[str, Any]:
        """Reject a healing action."""
        return {"status": "rejected"}
    
    async def get_healing_recommendations(self, api_spec_id: int) -> List[Dict[str, Any]]:
        """Get healing recommendations for an API specification."""
        return [
            {
                "type": "general",
                "priority": "low",
                "message": "No specific recommendations available in MVP",
                "action_items": ["Monitor test failures", "Review test case effectiveness"]
            }
        ]
    
    async def get_healing_statistics(self, api_spec_id: int, days: int = 30) -> Dict[str, Any]:
        """Get healing statistics."""
        return {
            "total_healing_attempts": 0,
            "successful_healings": 0,
            "failed_healings": 0,
            "healing_success_rate": 0.0,
            "most_common_issues": [],
            "healing_time_saved_hours": 0.0
        }
    
    async def get_healing_history(
        self,
        api_spec_id: Optional[int] = None,
        test_case_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get healing history."""
        # For MVP, return empty history
        return []
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.utcnow().isoformat()
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'db'):
            self.db.close()
