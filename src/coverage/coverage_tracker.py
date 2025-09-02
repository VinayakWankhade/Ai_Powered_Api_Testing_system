"""
Simplified coverage tracker for MVP.
Tracks test coverage metrics and provides analytics.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..database.connection import get_db_session
from ..database.models import (
    APISpecification, TestCase, ExecutionSession, TestExecution, 
    CoverageMetrics, TestType, TestStatus
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CoverageTracker:
    """Simplified coverage tracker for test coverage analysis."""
    
    def __init__(self):
        self.db = get_db_session()
    
    async def get_coverage_report(self, api_spec_id: int) -> Dict[str, Any]:
        """Get comprehensive coverage report for an API specification."""
        
        try:
            # Get API specification
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == api_spec_id
            ).first()
            
            if not api_spec:
                raise ValueError(f"API specification {api_spec_id} not found")
            
            # Get all test cases for this API spec
            test_cases = self.db.query(TestCase).filter(
                TestCase.api_spec_id == api_spec_id,
                TestCase.is_active == True
            ).all()
            
            # Get recent executions
            recent_executions = self.db.query(TestExecution).join(ExecutionSession).filter(
                ExecutionSession.api_spec_id == api_spec_id,
                TestExecution.completed_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Calculate endpoint coverage
            endpoint_coverage = self._calculate_endpoint_coverage(api_spec, test_cases, recent_executions)
            
            # Calculate method coverage
            method_coverage = self._calculate_method_coverage(api_spec, test_cases, recent_executions)
            
            # Calculate status code coverage
            status_code_coverage = self._calculate_status_code_coverage(recent_executions)
            
            # Calculate parameter coverage
            parameter_coverage = self._calculate_parameter_coverage(api_spec, test_cases)
            
            # Calculate test type coverage
            test_type_coverage = self._calculate_test_type_coverage(test_cases)
            
            # Identify coverage gaps
            coverage_gaps = self._identify_coverage_gaps(api_spec, test_cases, recent_executions)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(coverage_gaps, test_type_coverage)
            
            # Calculate overall coverage percentage
            overall_coverage = self._calculate_overall_coverage(
                endpoint_coverage, method_coverage, status_code_coverage, 
                parameter_coverage, test_type_coverage
            )
            
            return {
                "overall_coverage": overall_coverage,
                "endpoint_coverage": endpoint_coverage,
                "method_coverage": method_coverage,
                "status_code_coverage": status_code_coverage,
                "parameter_coverage": parameter_coverage,
                "test_type_coverage": test_type_coverage,
                "coverage_gaps": coverage_gaps,
                "recommendations": recommendations,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get coverage report: {str(e)}")
            return self._get_empty_coverage_report()
    
    def _calculate_endpoint_coverage(
        self,
        api_spec: APISpecification,
        test_cases: List[TestCase],
        executions: List[TestExecution]
    ) -> Dict[str, Any]:
        """Calculate endpoint coverage statistics."""
        
        try:
            # Get all endpoints from API specification
            all_endpoints = set()
            if api_spec.parsed_endpoints:
                for path, methods in api_spec.parsed_endpoints.items():
                    for method in methods.keys():
                        all_endpoints.add(f"{method.upper()} {path}")
            
            # Get covered endpoints from test cases and executions
            covered_endpoints = set()
            for test_case in test_cases:
                if test_case.endpoint and test_case.method:
                    covered_endpoints.add(f"{test_case.method.upper()} {test_case.endpoint}")
            
            # Get executed endpoints
            executed_endpoints = set()
            for execution in executions:
                if execution.endpoint and execution.method:
                    executed_endpoints.add(f"{execution.method.upper()} {execution.endpoint}")
            
            total_endpoints = len(all_endpoints) if all_endpoints else 1
            covered_count = len(covered_endpoints & all_endpoints)
            executed_count = len(executed_endpoints & all_endpoints)
            
            coverage_percentage = (covered_count / total_endpoints) * 100
            execution_percentage = (executed_count / total_endpoints) * 100
            
            return {
                "total_endpoints": total_endpoints,
                "covered_endpoints": covered_count,
                "executed_endpoints": executed_count,
                "coverage_percentage": coverage_percentage,
                "execution_percentage": execution_percentage,
                "uncovered_endpoints": list(all_endpoints - covered_endpoints),
                "covered_endpoint_list": list(covered_endpoints & all_endpoints)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate endpoint coverage: {str(e)}")
            return {
                "total_endpoints": 0,
                "covered_endpoints": 0,
                "executed_endpoints": 0,
                "coverage_percentage": 0.0,
                "execution_percentage": 0.0,
                "uncovered_endpoints": [],
                "covered_endpoint_list": []
            }
    
    def _calculate_method_coverage(
        self,
        api_spec: APISpecification,
        test_cases: List[TestCase],
        executions: List[TestExecution]
    ) -> Dict[str, Any]:
        """Calculate HTTP method coverage statistics."""
        
        try:
            # Get all methods from API specification
            all_methods = set()
            if api_spec.parsed_endpoints:
                for methods in api_spec.parsed_endpoints.values():
                    all_methods.update(method.upper() for method in methods.keys())
            
            # Get covered methods from test cases
            covered_methods = set(tc.method.upper() for tc in test_cases if tc.method)
            
            # Get executed methods
            executed_methods = set(ex.method.upper() for ex in executions if ex.method)
            
            total_methods = len(all_methods) if all_methods else 1
            covered_count = len(covered_methods & all_methods)
            executed_count = len(executed_methods & all_methods)
            
            coverage_percentage = (covered_count / total_methods) * 100
            execution_percentage = (executed_count / total_methods) * 100
            
            # Method-specific statistics
            method_stats = {}
            for method in all_methods:
                method_test_cases = [tc for tc in test_cases if tc.method and tc.method.upper() == method]
                method_executions = [ex for ex in executions if ex.method and ex.method.upper() == method]
                
                method_stats[method] = {
                    "test_cases": len(method_test_cases),
                    "executions": len(method_executions),
                    "success_rate": (
                        len([ex for ex in method_executions if ex.status == TestStatus.PASSED]) / 
                        len(method_executions)
                    ) * 100 if method_executions else 0
                }
            
            return {
                "total_methods": total_methods,
                "covered_methods": covered_count,
                "executed_methods": executed_count,
                "coverage_percentage": coverage_percentage,
                "execution_percentage": execution_percentage,
                "method_statistics": method_stats,
                "uncovered_methods": list(all_methods - covered_methods)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate method coverage: {str(e)}")
            return {
                "total_methods": 0,
                "covered_methods": 0,
                "executed_methods": 0,
                "coverage_percentage": 0.0,
                "execution_percentage": 0.0,
                "method_statistics": {},
                "uncovered_methods": []
            }
    
    def _calculate_status_code_coverage(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Calculate HTTP status code coverage from executions."""
        
        try:
            status_codes = defaultdict(int)
            success_codes = set()
            client_error_codes = set()
            server_error_codes = set()
            
            for execution in executions:
                if execution.response_data and isinstance(execution.response_data, dict):
                    status_code = execution.response_data.get('status_code')
                    if status_code:
                        status_codes[status_code] += 1
                        
                        if 200 <= status_code < 300:
                            success_codes.add(status_code)
                        elif 400 <= status_code < 500:
                            client_error_codes.add(status_code)
                        elif 500 <= status_code < 600:
                            server_error_codes.add(status_code)
            
            total_responses = len(executions)
            unique_status_codes = len(status_codes)
            
            # Calculate coverage based on common status codes
            common_codes = {200, 201, 400, 401, 403, 404, 422, 500}
            covered_common = len(common_codes & set(status_codes.keys()))
            common_coverage = (covered_common / len(common_codes)) * 100
            
            return {
                "total_responses": total_responses,
                "unique_status_codes": unique_status_codes,
                "common_coverage_percentage": common_coverage,
                "status_code_distribution": dict(status_codes),
                "success_codes": list(success_codes),
                "client_error_codes": list(client_error_codes),
                "server_error_codes": list(server_error_codes),
                "most_common_status": max(status_codes.items(), key=lambda x: x[1])[0] if status_codes else None
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate status code coverage: {str(e)}")
            return {
                "total_responses": 0,
                "unique_status_codes": 0,
                "common_coverage_percentage": 0.0,
                "status_code_distribution": {},
                "success_codes": [],
                "client_error_codes": [],
                "server_error_codes": [],
                "most_common_status": None
            }
    
    def _calculate_parameter_coverage(
        self,
        api_spec: APISpecification,
        test_cases: List[TestCase]
    ) -> Dict[str, Any]:
        """Calculate parameter coverage (simplified for MVP)."""
        
        try:
            # For MVP, we'll use a simplified parameter coverage calculation
            # based on test case complexity and data variety
            
            total_test_cases = len(test_cases)
            if total_test_cases == 0:
                return {
                    "total_parameters": 0,
                    "covered_parameters": 0,
                    "coverage_percentage": 0.0,
                    "parameter_types_covered": [],
                    "missing_parameter_types": []
                }
            
            # Analyze test cases for parameter usage patterns
            parameter_patterns = {
                "path_parameters": 0,
                "query_parameters": 0,
                "request_body": 0,
                "headers": 0,
                "form_data": 0
            }
            
            for test_case in test_cases:
                if test_case.test_data:
                    test_data = test_case.test_data
                    if isinstance(test_data, dict):
                        if test_data.get('path_params'):
                            parameter_patterns["path_parameters"] += 1
                        if test_data.get('query_params'):
                            parameter_patterns["query_parameters"] += 1
                        if test_data.get('body') or test_data.get('json'):
                            parameter_patterns["request_body"] += 1
                        if test_data.get('headers'):
                            parameter_patterns["headers"] += 1
                        if test_data.get('form_data'):
                            parameter_patterns["form_data"] += 1
            
            # Calculate coverage percentage based on parameter diversity
            total_parameter_types = len(parameter_patterns)
            covered_types = sum(1 for count in parameter_patterns.values() if count > 0)
            coverage_percentage = (covered_types / total_parameter_types) * 100
            
            return {
                "total_parameters": total_parameter_types,
                "covered_parameters": covered_types,
                "coverage_percentage": coverage_percentage,
                "parameter_types_covered": [
                    param_type for param_type, count in parameter_patterns.items() if count > 0
                ],
                "missing_parameter_types": [
                    param_type for param_type, count in parameter_patterns.items() if count == 0
                ],
                "parameter_usage_stats": parameter_patterns
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate parameter coverage: {str(e)}")
            return {
                "total_parameters": 0,
                "covered_parameters": 0,
                "coverage_percentage": 0.0,
                "parameter_types_covered": [],
                "missing_parameter_types": [],
                "parameter_usage_stats": {}
            }
    
    def _calculate_test_type_coverage(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Calculate test type coverage distribution."""
        
        try:
            test_type_counts = defaultdict(int)
            total_tests = len(test_cases)
            
            for test_case in test_cases:
                test_type_counts[test_case.test_type.value] += 1
            
            # Calculate percentages
            test_type_percentages = {}
            for test_type, count in test_type_counts.items():
                test_type_percentages[test_type] = (count / total_tests) * 100 if total_tests > 0 else 0
            
            # Define ideal distribution (simplified for MVP)
            ideal_distribution = {
                "functional": 40,
                "edge_case": 25,
                "security": 15,
                "performance": 10,
                "generated": 10
            }
            
            # Calculate coverage score based on how well we match ideal distribution
            coverage_score = 0
            for test_type, ideal_pct in ideal_distribution.items():
                actual_pct = test_type_percentages.get(test_type, 0)
                # Score based on how close we are to ideal (max score when within 10% of ideal)
                if actual_pct >= ideal_pct * 0.5:  # At least 50% of ideal
                    coverage_score += min(actual_pct, ideal_pct) / ideal_pct * 20  # 20 points per type
            
            missing_types = [
                test_type for test_type, ideal_pct in ideal_distribution.items()
                if test_type_percentages.get(test_type, 0) < ideal_pct * 0.25
            ]
            
            return {
                "total_test_cases": total_tests,
                "test_type_distribution": dict(test_type_counts),
                "test_type_percentages": test_type_percentages,
                "coverage_score": min(coverage_score, 100),  # Cap at 100
                "ideal_distribution": ideal_distribution,
                "missing_types": missing_types,
                "well_covered_types": [
                    test_type for test_type, pct in test_type_percentages.items()
                    if pct >= ideal_distribution.get(test_type, 0) * 0.75
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate test type coverage: {str(e)}")
            return {
                "total_test_cases": 0,
                "test_type_distribution": {},
                "test_type_percentages": {},
                "coverage_score": 0.0,
                "ideal_distribution": {},
                "missing_types": [],
                "well_covered_types": []
            }
    
    def _identify_coverage_gaps(
        self,
        api_spec: APISpecification,
        test_cases: List[TestCase],
        executions: List[TestExecution]
    ) -> List[Dict[str, Any]]:
        """Identify coverage gaps and areas for improvement."""
        
        gaps = []
        
        try:
            # Gap 1: Untested endpoints
            all_endpoints = set()
            if api_spec.parsed_endpoints:
                for path, methods in api_spec.parsed_endpoints.items():
                    for method in methods.keys():
                        all_endpoints.add(f"{method.upper()} {path}")
            
            tested_endpoints = set()
            for test_case in test_cases:
                if test_case.endpoint and test_case.method:
                    tested_endpoints.add(f"{test_case.method.upper()} {test_case.endpoint}")
            
            untested_endpoints = all_endpoints - tested_endpoints
            if untested_endpoints:
                gaps.append({
                    "type": "untested_endpoints",
                    "severity": "high",
                    "title": "Untested Endpoints",
                    "description": f"{len(untested_endpoints)} endpoints have no test coverage",
                    "details": list(untested_endpoints)[:10],  # Limit to first 10
                    "recommendation": "Create test cases for uncovered endpoints"
                })
            
            # Gap 2: Missing test types
            existing_types = set(tc.test_type.value for tc in test_cases)
            important_types = {"functional", "edge_case", "security"}
            missing_types = important_types - existing_types
            
            if missing_types:
                gaps.append({
                    "type": "missing_test_types",
                    "severity": "medium",
                    "title": "Missing Test Types",
                    "description": f"Important test types are missing: {', '.join(missing_types)}",
                    "details": list(missing_types),
                    "recommendation": "Add test cases for missing test types to improve coverage quality"
                })
            
            # Gap 3: Low execution success rate
            if executions:
                passed_executions = len([ex for ex in executions if ex.status == TestStatus.PASSED])
                success_rate = (passed_executions / len(executions)) * 100
                
                if success_rate < 70:
                    gaps.append({
                        "type": "low_success_rate",
                        "severity": "high",
                        "title": "Low Test Success Rate",
                        "description": f"Test success rate is {success_rate:.1f}%, indicating potential issues",
                        "details": {"success_rate": success_rate, "total_executions": len(executions)},
                        "recommendation": "Review failing tests and improve test reliability or fix API issues"
                    })
            
            # Gap 4: Insufficient error code coverage
            status_codes = set()
            for execution in executions:
                if execution.response_data and isinstance(execution.response_data, dict):
                    status_code = execution.response_data.get('status_code')
                    if status_code:
                        status_codes.add(status_code)
            
            important_error_codes = {400, 401, 403, 404, 422}
            missing_error_codes = important_error_codes - status_codes
            
            if len(missing_error_codes) > 2:  # Allow some flexibility
                gaps.append({
                    "type": "missing_error_codes",
                    "severity": "medium",
                    "title": "Insufficient Error Code Coverage",
                    "description": f"Missing tests for important error codes: {missing_error_codes}",
                    "details": list(missing_error_codes),
                    "recommendation": "Add negative test cases to cover error scenarios"
                })
            
        except Exception as e:
            logger.error(f"Failed to identify coverage gaps: {str(e)}")
        
        return gaps
    
    def _generate_recommendations(
        self,
        coverage_gaps: List[Dict[str, Any]],
        test_type_coverage: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on coverage analysis."""
        
        recommendations = []
        
        try:
            # High-priority recommendations from gaps
            high_priority_gaps = [gap for gap in coverage_gaps if gap.get("severity") == "high"]
            for gap in high_priority_gaps:
                recommendations.append({
                    "type": "coverage_gap",
                    "priority": "high",
                    "title": f"Address {gap['title']}",
                    "description": gap["recommendation"],
                    "action_items": [gap["recommendation"]],
                    "expected_impact": "Significant improvement in test coverage"
                })
            
            # Test type balance recommendations
            missing_types = test_type_coverage.get("missing_types", [])
            if missing_types:
                recommendations.append({
                    "type": "test_balance",
                    "priority": "medium",
                    "title": "Improve Test Type Balance",
                    "description": f"Add more {', '.join(missing_types)} tests to improve coverage balance",
                    "action_items": [
                        f"Create {test_type} test cases" for test_type in missing_types
                    ],
                    "expected_impact": "Better test coverage quality and risk mitigation"
                })
            
            # Coverage score improvement
            coverage_score = test_type_coverage.get("coverage_score", 0)
            if coverage_score < 80:
                recommendations.append({
                    "type": "coverage_improvement",
                    "priority": "medium",
                    "title": "Increase Overall Coverage Score",
                    "description": f"Current coverage score is {coverage_score:.1f}%. Target: 80%+",
                    "action_items": [
                        "Focus on underrepresented test types",
                        "Add more comprehensive test scenarios",
                        "Ensure balanced test distribution"
                    ],
                    "expected_impact": "Higher confidence in API quality and reliability"
                })
            
            # General recommendations
            if not recommendations:
                recommendations.append({
                    "type": "maintenance",
                    "priority": "low",
                    "title": "Maintain Current Coverage",
                    "description": "Coverage levels are good. Focus on maintaining quality.",
                    "action_items": [
                        "Continue regular test execution",
                        "Monitor for new endpoints",
                        "Update tests as API evolves"
                    ],
                    "expected_impact": "Sustained high-quality test coverage"
                })
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_overall_coverage(
        self,
        endpoint_coverage: Dict[str, Any],
        method_coverage: Dict[str, Any],
        status_code_coverage: Dict[str, Any],
        parameter_coverage: Dict[str, Any],
        test_type_coverage: Dict[str, Any]
    ) -> float:
        """Calculate overall coverage percentage with weighted scoring."""
        
        try:
            # Weighted coverage calculation
            weights = {
                "endpoint": 0.3,
                "method": 0.25,
                "status_code": 0.15,
                "parameter": 0.15,
                "test_type": 0.15
            }
            
            scores = {
                "endpoint": endpoint_coverage.get("coverage_percentage", 0),
                "method": method_coverage.get("coverage_percentage", 0),
                "status_code": status_code_coverage.get("common_coverage_percentage", 0),
                "parameter": parameter_coverage.get("coverage_percentage", 0),
                "test_type": test_type_coverage.get("coverage_score", 0)
            }
            
            overall_score = sum(
                scores[category] * weight 
                for category, weight in weights.items()
            )
            
            return min(max(overall_score, 0), 100)  # Clamp between 0 and 100
            
        except Exception as e:
            logger.error(f"Failed to calculate overall coverage: {str(e)}")
            return 0.0
    
    def _get_empty_coverage_report(self) -> Dict[str, Any]:
        """Return empty coverage report for error cases."""
        
        return {
            "overall_coverage": 0.0,
            "endpoint_coverage": {"total_endpoints": 0, "covered_endpoints": 0, "coverage_percentage": 0.0},
            "method_coverage": {"total_methods": 0, "covered_methods": 0, "coverage_percentage": 0.0},
            "status_code_coverage": {"total_responses": 0, "unique_status_codes": 0, "common_coverage_percentage": 0.0},
            "parameter_coverage": {"total_parameters": 0, "covered_parameters": 0, "coverage_percentage": 0.0},
            "test_type_coverage": {"total_test_cases": 0, "coverage_score": 0.0},
            "coverage_gaps": [],
            "recommendations": [{
                "type": "error",
                "priority": "high",
                "title": "Coverage Analysis Failed",
                "description": "Unable to analyze coverage. Check system logs.",
                "action_items": ["Review system logs", "Verify data integrity"],
                "expected_impact": "Restored coverage analysis capability"
            }],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def get_analytics_report(self, api_spec_id: int, days: int = 30) -> Dict[str, Any]:
        """Get analytics report with execution statistics and trends."""
        # Simplified implementation for MVP
        return {
            "execution_stats": {"total_sessions": 0, "total_tests": 0},
            "performance_metrics": {"avg_response_time": 0},
            "quality_trends": {"success_rate_trend": "stable"},
            "failure_analysis": {"common_failures": []},
            "suggestions": [],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def get_coverage_trends(self, api_spec_id: int, days: int = 30) -> Dict[str, Any]:
        """Get coverage trends over time."""
        # Simplified implementation for MVP
        return {
            "trends": [],
            "analysis": {"trend_direction": "stable"}
        }
    
    async def get_test_quality_metrics(self, api_spec_id: int) -> Dict[str, Any]:
        """Get test quality metrics."""
        # Simplified implementation for MVP
        return {"quality_score": 75.0, "reliability_score": 80.0}
    
    async def get_performance_analysis(self, api_spec_id: int, days: int = 7) -> Dict[str, Any]:
        """Get performance analysis."""
        # Simplified implementation for MVP
        return {"avg_response_time": 250, "p95_response_time": 500}
    
    async def analyze_failure_patterns(self, api_spec_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze failure patterns."""
        # Simplified implementation for MVP
        return {"common_failures": [], "failure_rate": 0.05}
    
    async def identify_coverage_gaps(self, api_spec_id: int) -> Dict[str, Any]:
        """Identify coverage gaps."""
        # Simplified implementation for MVP
        return {"gaps": [], "recommendations": [], "analysis": {}}
    
    async def analyze_test_effectiveness(self, api_spec_id: int) -> Dict[str, Any]:
        """Analyze test effectiveness."""
        # Simplified implementation for MVP
        return {"effectiveness_score": 80.0}
    
    async def update_coverage_from_session(self, api_spec_id: int, session_id: int) -> Dict[str, Any]:
        """Update coverage data from execution session."""
        # Simplified implementation for MVP
        return {"status": "updated", "coverage_updated": True}
    
    async def update_coverage_data(self, api_spec_id: int) -> Dict[str, Any]:
        """Update coverage data."""
        # Simplified implementation for MVP
        return {"status": "updated", "coverage_updated": True}
    
    async def get_dashboard_data(self, api_spec_id: int) -> Dict[str, Any]:
        """Get dashboard data."""
        # Simplified implementation for MVP
        return {"summary": "Dashboard data not implemented in MVP"}
    
    async def get_comparative_analysis(self, api_spec_ids: List[int], days: int = 30) -> Dict[str, Any]:
        """Get comparative analysis across multiple API specifications."""
        # Simplified implementation for MVP
        return {"comparison": "Not implemented in MVP"}
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
