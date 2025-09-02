"""
Advanced Oracle System for comprehensive API test validation.
Implements contract oracles, invariant checking, diff detection, and business rule validation.
"""

import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    import deepdiff
    DEEPDIFF_AVAILABLE = True
except ImportError:
    DEEPDIFF_AVAILABLE = False

from ..database.connection import get_db_session
from ..database.models import TestExecution, TestCase, APISpecification
from ..utils.logger import get_logger

logger = get_logger(__name__)

class OracleType(Enum):
    """Types of oracles for different validation scenarios."""
    CONTRACT = "contract"
    INVARIANT = "invariant"
    DIFF = "diff"
    BUSINESS_RULE = "business_rule"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SCHEMA = "schema"

class InvariantType(Enum):
    """Types of API invariants to check."""
    IDEMPOTENCY = "idempotency"
    AUTH_REQUIRED = "auth_required"
    PAGINATION_MONOTONIC = "pagination_monotonic"
    LIFECYCLE_CONSISTENCY = "lifecycle_consistency"
    RATE_LIMIT_RESPECT = "rate_limit_respect"
    DATA_INTEGRITY = "data_integrity"

@dataclass
class OracleResult:
    """Result of an oracle evaluation."""
    oracle_type: OracleType
    passed: bool
    details: Dict[str, Any]
    severity: str  # "low", "medium", "high", "critical"
    message: str
    evidence: Optional[Dict[str, Any]] = None

@dataclass
class InvariantViolation:
    """Details of an invariant violation."""
    invariant_type: InvariantType
    endpoint: str
    method: str
    violation_details: Dict[str, Any]
    test_data: Dict[str, Any]
    response_data: Dict[str, Any]
    severity: str

class AdvancedOracleSystem:
    """
    Advanced oracle system that validates API responses against multiple criteria:
    - Contract compliance (status codes, schemas, required fields)
    - API invariants (idempotency, auth, pagination, etc.)
    - Response diff detection (schema drift, breaking changes)
    - Business rule validation (configurable predicates)
    - Security validation (no sensitive data leakage)
    """
    
    def __init__(self):
        self.db = get_db_session()
        self.baseline_responses = {}  # Cache for diff detection
        
    async def evaluate_all_oracles(
        self, 
        test_execution: TestExecution,
        previous_responses: Optional[List[Dict[str, Any]]] = None,
        business_rules: Optional[List[Dict[str, Any]]] = None
    ) -> List[OracleResult]:
        """
        Run all applicable oracles on a test execution.
        
        Args:
            test_execution: The test execution to validate
            previous_responses: Previous responses for diff/invariant checking
            business_rules: Custom business rules to validate
            
        Returns:
            List of oracle results
        """
        try:
            oracle_results = []
            
            # Contract Oracle
            contract_result = await self._evaluate_contract_oracle(test_execution)
            oracle_results.append(contract_result)
            
            # Schema Oracle (if schema validation available)
            if JSONSCHEMA_AVAILABLE:
                schema_result = await self._evaluate_schema_oracle(test_execution)
                oracle_results.append(schema_result)
            
            # Invariant Oracles
            invariant_results = await self._evaluate_invariant_oracles(test_execution, previous_responses)
            oracle_results.extend(invariant_results)
            
            # Diff Oracle (if previous responses available)
            if previous_responses:
                diff_result = await self._evaluate_diff_oracle(test_execution, previous_responses)
                oracle_results.append(diff_result)
            
            # Business Rule Oracles
            if business_rules:
                business_results = await self._evaluate_business_rule_oracles(test_execution, business_rules)
                oracle_results.extend(business_results)
            
            # Security Oracle
            security_result = await self._evaluate_security_oracle(test_execution)
            oracle_results.append(security_result)
            
            # Performance Oracle
            performance_result = await self._evaluate_performance_oracle(test_execution)
            oracle_results.append(performance_result)
            
            logger.debug(f"Evaluated {len(oracle_results)} oracles for execution {test_execution.id}")
            return oracle_results
            
        except Exception as e:
            logger.error(f"Oracle evaluation failed: {str(e)}")
            return [OracleResult(
                oracle_type=OracleType.CONTRACT,
                passed=False,
                details={"error": str(e)},
                severity="high",
                message=f"Oracle system error: {str(e)}"
            )]
    
    async def _evaluate_contract_oracle(self, test_execution: TestExecution) -> OracleResult:
        """Evaluate contract compliance (status codes, required fields, etc.)."""
        
        try:
            test_case = test_execution.test_case
            if not test_case:
                return OracleResult(
                    oracle_type=OracleType.CONTRACT,
                    passed=False,
                    details={"error": "No test case associated"},
                    severity="high",
                    message="Contract validation failed: missing test case"
                )
            
            violations = []
            
            # Check status code
            expected_status = test_case.expected_response.get("status_code") if test_case.expected_response else 200
            actual_status = test_execution.response_code
            
            if isinstance(expected_status, list):
                status_valid = actual_status in expected_status
            else:
                status_valid = actual_status == expected_status
            
            if not status_valid:
                violations.append(f"Status code mismatch: expected {expected_status}, got {actual_status}")
            
            # Check required response fields
            if test_execution.response_body and isinstance(test_execution.response_body, dict):
                required_fields = test_case.expected_response.get("required_fields", []) if test_case.expected_response else []
                for field in required_fields:
                    if field not in test_execution.response_body:
                        violations.append(f"Missing required field: {field}")
            
            # Check response structure consistency
            content_type = test_execution.response_headers.get("Content-Type", "") if test_execution.response_headers else ""
            if "application/json" in content_type and not isinstance(test_execution.response_body, dict):
                violations.append("Content-Type indicates JSON but response is not JSON")
            
            # Check for error response structure
            if actual_status >= 400 and isinstance(test_execution.response_body, dict):
                if not any(field in test_execution.response_body for field in ["error", "message", "detail", "errors"]):
                    violations.append("Error response missing error information")
            
            passed = len(violations) == 0
            severity = "high" if not passed and actual_status >= 500 else "medium" if not passed else "low"
            
            return OracleResult(
                oracle_type=OracleType.CONTRACT,
                passed=passed,
                details={
                    "violations": violations,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "response_structure_valid": isinstance(test_execution.response_body, dict)
                },
                severity=severity,
                message=f"Contract validation {'passed' if passed else 'failed'}: {len(violations)} violations"
            )
            
        except Exception as e:
            logger.error(f"Contract oracle evaluation failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.CONTRACT,
                passed=False,
                details={"error": str(e)},
                severity="high",
                message=f"Contract validation error: {str(e)}"
            )
    
    async def _evaluate_schema_oracle(self, test_execution: TestExecution) -> OracleResult:
        """Evaluate JSON schema compliance."""
        
        try:
            test_case = test_execution.test_case
            if not test_case or not test_execution.response_body:
                return OracleResult(
                    oracle_type=OracleType.SCHEMA,
                    passed=True,
                    details={"skipped": "No response body to validate"},
                    severity="low",
                    message="Schema validation skipped"
                )
            
            # Get expected schema from API spec
            api_spec = self.db.query(APISpecification).filter(
                APISpecification.id == test_case.api_spec_id
            ).first()
            
            if not api_spec or not api_spec.parsed_endpoints:
                return OracleResult(
                    oracle_type=OracleType.SCHEMA,
                    passed=True,
                    details={"skipped": "No schema available"},
                    severity="low",
                    message="Schema validation skipped: no schema available"
                )
            
            # Extract schema for this endpoint
            endpoint_methods = api_spec.parsed_endpoints.get(test_case.endpoint, {})
            method_info = endpoint_methods.get(test_case.method.lower(), {})
            responses = method_info.get("responses", {})
            
            status_code_str = str(test_execution.response_code)
            response_schema = responses.get(status_code_str, {}).get("schema")
            
            if not response_schema:
                return OracleResult(
                    oracle_type=OracleType.SCHEMA,
                    passed=True,
                    details={"skipped": f"No schema for status {status_code_str}"},
                    severity="low",
                    message="Schema validation skipped: no schema defined"
                )
            
            # Validate against schema
            try:
                jsonschema.validate(test_execution.response_body, response_schema)
                return OracleResult(
                    oracle_type=OracleType.SCHEMA,
                    passed=True,
                    details={"schema_valid": True},
                    severity="low",
                    message="Schema validation passed"
                )
            except jsonschema.ValidationError as e:
                return OracleResult(
                    oracle_type=OracleType.SCHEMA,
                    passed=False,
                    details={
                        "schema_violations": [str(e)],
                        "schema": response_schema,
                        "response": test_execution.response_body
                    },
                    severity="medium",
                    message=f"Schema validation failed: {str(e)}"
                )
            
        except Exception as e:
            logger.error(f"Schema oracle evaluation failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.SCHEMA,
                passed=False,
                details={"error": str(e)},
                severity="medium",
                message=f"Schema validation error: {str(e)}"
            )
    
    async def _evaluate_invariant_oracles(
        self, 
        test_execution: TestExecution,
        previous_responses: Optional[List[Dict[str, Any]]] = None
    ) -> List[OracleResult]:
        """Evaluate API invariants."""
        
        invariant_results = []
        
        # Idempotency check for GET requests
        if test_execution.test_case and test_execution.test_case.method.upper() == "GET":
            idempotency_result = await self._check_idempotency_invariant(test_execution, previous_responses)
            invariant_results.append(idempotency_result)
        
        # Auth requirement check
        auth_result = await self._check_auth_invariant(test_execution)
        invariant_results.append(auth_result)
        
        # Pagination consistency (for paginated endpoints)
        if self._is_paginated_endpoint(test_execution):
            pagination_result = await self._check_pagination_invariant(test_execution)
            invariant_results.append(pagination_result)
        
        # Data integrity checks
        integrity_result = await self._check_data_integrity_invariant(test_execution)
        invariant_results.append(integrity_result)
        
        return invariant_results
    
    async def _check_idempotency_invariant(
        self, 
        test_execution: TestExecution,
        previous_responses: Optional[List[Dict[str, Any]]] = None
    ) -> OracleResult:
        """Check that GET requests are idempotent (don't cause mutations)."""
        
        try:
            # For GET requests, subsequent calls should return same data
            if test_execution.test_case.method.upper() != "GET":
                return OracleResult(
                    oracle_type=OracleType.INVARIANT,
                    passed=True,
                    details={"skipped": "Not a GET request"},
                    severity="low",
                    message="Idempotency check skipped for non-GET request"
                )
            
            # Check if we have previous responses to compare
            if not previous_responses:
                return OracleResult(
                    oracle_type=OracleType.INVARIANT,
                    passed=True,
                    details={"skipped": "No previous responses for comparison"},
                    severity="low",
                    message="Idempotency check skipped: first execution"
                )
            
            # Compare current response with previous responses
            current_response = test_execution.response_body
            violations = []
            
            for i, prev_response in enumerate(previous_responses[-3:]):  # Check last 3 responses
                if prev_response.get("status_code") == test_execution.response_code:
                    # Compare response bodies (ignore timestamps and dynamic fields)
                    normalized_current = self._normalize_response_for_comparison(current_response)
                    normalized_previous = self._normalize_response_for_comparison(prev_response.get("body"))
                    
                    if normalized_current != normalized_previous:
                        violations.append(f"Response differs from execution {i+1} ago")
            
            passed = len(violations) == 0
            
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=passed,
                details={
                    "invariant": "idempotency",
                    "violations": violations,
                    "comparisons_made": len(previous_responses)
                },
                severity="medium" if not passed else "low",
                message=f"Idempotency check {'passed' if passed else 'failed'}: {len(violations)} violations"
            )
            
        except Exception as e:
            logger.error(f"Idempotency invariant check failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=False,
                details={"error": str(e)},
                severity="medium",
                message=f"Idempotency check error: {str(e)}"
            )
    
    async def _check_auth_invariant(self, test_execution: TestExecution) -> OracleResult:
        """Check authentication and authorization invariants."""
        
        try:
            violations = []
            
            # Check that protected endpoints require authentication
            if self._is_protected_endpoint(test_execution.test_case):
                auth_header = None
                if test_execution.test_case.test_data:
                    headers = test_execution.test_case.test_data.get("headers", {})
                    auth_header = headers.get("Authorization") or headers.get("authentication")
                
                if not auth_header and test_execution.response_code not in [401, 403]:
                    violations.append("Protected endpoint should require authentication")
            
            # Check that auth failures return proper status codes
            if test_execution.response_code in [401, 403]:
                response_body = test_execution.response_body or {}
                if isinstance(response_body, dict):
                    has_error_info = any(
                        key in response_body for key in ["error", "message", "detail", "errors"]
                    )
                    if not has_error_info:
                        violations.append("Auth failure should include error information")
            
            # Check for auth token leakage in responses
            if test_execution.response_body:
                auth_leakage = self._check_for_auth_leakage(test_execution.response_body)
                if auth_leakage:
                    violations.extend(auth_leakage)
            
            passed = len(violations) == 0
            
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=passed,
                details={
                    "invariant": "authentication",
                    "violations": violations,
                    "protected_endpoint": self._is_protected_endpoint(test_execution.test_case)
                },
                severity="high" if not passed else "low",
                message=f"Auth invariant {'passed' if passed else 'failed'}: {len(violations)} violations"
            )
            
        except Exception as e:
            logger.error(f"Auth invariant check failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=False,
                details={"error": str(e)},
                severity="high",
                message=f"Auth check error: {str(e)}"
            )
    
    async def _check_pagination_invariant(self, test_execution: TestExecution) -> OracleResult:
        """Check pagination consistency invariants."""
        
        try:
            response_body = test_execution.response_body
            if not response_body or not isinstance(response_body, dict):
                return OracleResult(
                    oracle_type=OracleType.INVARIANT,
                    passed=True,
                    details={"skipped": "No JSON response body"},
                    severity="low",
                    message="Pagination check skipped"
                )
            
            violations = []
            
            # Check for common pagination fields
            pagination_fields = ["total", "count", "page", "per_page", "offset", "limit", "next", "previous"]
            has_pagination = any(field in response_body for field in pagination_fields)
            
            if has_pagination:
                # Validate pagination consistency
                total = response_body.get("total")
                count = response_body.get("count")
                page = response_body.get("page")
                per_page = response_body.get("per_page") or response_body.get("limit")
                
                # Check that count doesn't exceed total
                if total is not None and count is not None:
                    if count > total:
                        violations.append(f"Count ({count}) exceeds total ({total})")
                
                # Check that page numbers are valid
                if page is not None and page < 1:
                    violations.append(f"Invalid page number: {page}")
                
                # Check that per_page is reasonable
                if per_page is not None and (per_page < 1 or per_page > 1000):
                    violations.append(f"Unreasonable per_page value: {per_page}")
                
                # Check that data array exists and has reasonable size
                data_field = None
                for field in ["data", "items", "results", "records"]:
                    if field in response_body:
                        data_field = field
                        break
                
                if data_field:
                    data_array = response_body[data_field]
                    if isinstance(data_array, list):
                        if count is not None and len(data_array) != count:
                            violations.append(f"Data array length ({len(data_array)}) doesn't match count ({count})")
                        if per_page is not None and len(data_array) > per_page:
                            violations.append(f"Data array length ({len(data_array)}) exceeds per_page ({per_page})")
            
            passed = len(violations) == 0
            
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=passed,
                details={
                    "invariant": "pagination",
                    "violations": violations,
                    "has_pagination": has_pagination,
                    "pagination_fields_found": [f for f in pagination_fields if f in response_body]
                },
                severity="medium" if not passed else "low",
                message=f"Pagination invariant {'passed' if passed else 'failed'}: {len(violations)} violations"
            )
            
        except Exception as e:
            logger.error(f"Pagination invariant check failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=False,
                details={"error": str(e)},
                severity="medium",
                message=f"Pagination check error: {str(e)}"
            )
    
    async def _check_data_integrity_invariant(self, test_execution: TestExecution) -> OracleResult:
        """Check data integrity invariants."""
        
        try:
            response_body = test_execution.response_body
            if not response_body or not isinstance(response_body, dict):
                return OracleResult(
                    oracle_type=OracleType.INVARIANT,
                    passed=True,
                    details={"skipped": "No JSON response body"},
                    severity="low",
                    message="Data integrity check skipped"
                )
            
            violations = []
            
            # Check for common data integrity issues
            violations.extend(self._check_numeric_fields_integrity(response_body))
            violations.extend(self._check_date_fields_integrity(response_body))
            violations.extend(self._check_email_fields_integrity(response_body))
            violations.extend(self._check_url_fields_integrity(response_body))
            violations.extend(self._check_business_logic_integrity(response_body))
            
            passed = len(violations) == 0
            
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=passed,
                details={
                    "invariant": "data_integrity",
                    "violations": violations,
                    "checks_performed": ["numeric", "date", "email", "url", "business_logic"]
                },
                severity="medium" if not passed else "low",
                message=f"Data integrity {'passed' if passed else 'failed'}: {len(violations)} violations"
            )
            
        except Exception as e:
            logger.error(f"Data integrity check failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.INVARIANT,
                passed=False,
                details={"error": str(e)},
                severity="medium",
                message=f"Data integrity error: {str(e)}"
            )
    
    def _check_numeric_fields_integrity(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check integrity of numeric fields."""
        
        violations = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                violations.extend(self._check_numeric_fields_integrity(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(self._check_numeric_fields_integrity(item, f"{current_path}[{i}]"))
            elif isinstance(value, (int, float)):
                # Check for common numeric integrity issues
                if key.lower() in ["price", "amount", "cost", "total", "sum"] and value < 0:
                    violations.append(f"Negative value for {current_path}: {value}")
                
                if key.lower() in ["count", "quantity", "number"] and value < 0:
                    violations.append(f"Negative count for {current_path}: {value}")
                
                if key.lower() in ["percentage", "percent"] and (value < 0 or value > 100):
                    violations.append(f"Invalid percentage for {current_path}: {value}")
        
        return violations
    
    def _check_date_fields_integrity(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check integrity of date fields."""
        
        violations = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                violations.extend(self._check_date_fields_integrity(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(self._check_date_fields_integrity(item, f"{current_path}[{i}]"))
            elif isinstance(value, str) and self._looks_like_date(key, value):
                # Validate date format and reasonableness
                if not self._is_valid_date_format(value):
                    violations.append(f"Invalid date format for {current_path}: {value}")
                elif not self._is_reasonable_date(value):
                    violations.append(f"Unreasonable date for {current_path}: {value}")
        
        # Check date ordering (start_date < end_date, created_at <= updated_at)
        violations.extend(self._check_date_ordering(data, path))
        
        return violations
    
    def _check_email_fields_integrity(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check integrity of email fields."""
        
        violations = []
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                violations.extend(self._check_email_fields_integrity(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(self._check_email_fields_integrity(item, f"{current_path}[{i}]"))
            elif isinstance(value, str) and "email" in key.lower():
                if not email_pattern.match(value):
                    violations.append(f"Invalid email format for {current_path}: {value}")
        
        return violations
    
    def _check_url_fields_integrity(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check integrity of URL fields."""
        
        violations = []
        url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                violations.extend(self._check_url_fields_integrity(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(self._check_url_fields_integrity(item, f"{current_path}[{i}]"))
            elif isinstance(value, str) and ("url" in key.lower() or "link" in key.lower() or "href" in key.lower()):
                if not url_pattern.match(value):
                    violations.append(f"Invalid URL format for {current_path}: {value}")
        
        return violations
    
    def _check_business_logic_integrity(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check business logic integrity."""
        
        violations = []
        
        # Common business logic checks
        if isinstance(data, dict):
            # Check that end dates are after start dates
            start_fields = [k for k in data.keys() if "start" in k.lower() and "date" in k.lower()]
            end_fields = [k for k in data.keys() if "end" in k.lower() and "date" in k.lower()]
            
            for start_field in start_fields:
                for end_field in end_fields:
                    start_date = data.get(start_field)
                    end_date = data.get(end_field)
                    
                    if start_date and end_date and isinstance(start_date, str) and isinstance(end_date, str):
                        if start_date > end_date:  # Simple string comparison for ISO dates
                            violations.append(f"Start date ({start_field}={start_date}) after end date ({end_field}={end_date})")
            
            # Check quantity vs total consistency
            if "quantity" in data and "unit_price" in data and "total" in data:
                expected_total = data["quantity"] * data["unit_price"]
                actual_total = data["total"]
                if abs(expected_total - actual_total) > 0.01:  # Allow for rounding
                    violations.append(f"Total ({actual_total}) doesn't match quantity*unit_price ({expected_total})")
        
        return violations
    
    async def _evaluate_diff_oracle(
        self, 
        test_execution: TestExecution,
        previous_responses: List[Dict[str, Any]]
    ) -> OracleResult:
        """Evaluate response differences for schema drift detection."""
        
        try:
            if not DEEPDIFF_AVAILABLE:
                return OracleResult(
                    oracle_type=OracleType.DIFF,
                    passed=True,
                    details={"skipped": "deepdiff not available"},
                    severity="low",
                    message="Diff check skipped: deepdiff not installed"
                )
            
            current_response = test_execution.response_body
            if not current_response:
                return OracleResult(
                    oracle_type=OracleType.DIFF,
                    passed=True,
                    details={"skipped": "No response body"},
                    severity="low",
                    message="Diff check skipped: no response body"
                )
            
            # Find baseline response with same status code
            baseline_response = None
            for prev_resp in previous_responses:
                if prev_resp.get("status_code") == test_execution.response_code:
                    baseline_response = prev_resp.get("body")
                    break
            
            if not baseline_response:
                return OracleResult(
                    oracle_type=OracleType.DIFF,
                    passed=True,
                    details={"skipped": "No baseline response found"},
                    severity="low",
                    message="Diff check skipped: no baseline"
                )
            
            # Calculate differences
            from deepdiff import DeepDiff
            diff = DeepDiff(
                baseline_response, 
                current_response, 
                ignore_order=True,
                exclude_paths=["root['timestamp']", "root['request_id']", "root['trace_id']"]
            )
            
            # Analyze differences for schema drift
            significant_changes = []
            
            if 'type_changes' in diff:
                significant_changes.append("Type changes detected (possible schema drift)")
            
            if 'dictionary_item_removed' in diff:
                removed_fields = list(diff['dictionary_item_removed'].keys())
                significant_changes.append(f"Fields removed: {removed_fields}")
            
            if 'dictionary_item_added' in diff:
                added_fields = list(diff['dictionary_item_added'].keys())
                significant_changes.append(f"Fields added: {added_fields}")
            
            passed = len(significant_changes) == 0
            severity = "high" if "type_changes" in diff else "medium" if significant_changes else "low"
            
            return OracleResult(
                oracle_type=OracleType.DIFF,
                passed=passed,
                details={
                    "significant_changes": significant_changes,
                    "full_diff": str(diff) if diff else "No differences",
                    "baseline_available": True
                },
                severity=severity,
                message=f"Diff check {'passed' if passed else 'failed'}: {len(significant_changes)} significant changes"
            )
            
        except Exception as e:
            logger.error(f"Diff oracle evaluation failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.DIFF,
                passed=False,
                details={"error": str(e)},
                severity="medium",
                message=f"Diff check error: {str(e)}"
            )
    
    async def _evaluate_business_rule_oracles(
        self, 
        test_execution: TestExecution,
        business_rules: List[Dict[str, Any]]
    ) -> List[OracleResult]:
        """Evaluate custom business rules."""
        
        business_results = []
        
        for rule in business_rules:
            try:
                rule_result = await self._evaluate_single_business_rule(test_execution, rule)
                business_results.append(rule_result)
            except Exception as e:
                logger.error(f"Business rule evaluation failed: {str(e)}")
                business_results.append(OracleResult(
                    oracle_type=OracleType.BUSINESS_RULE,
                    passed=False,
                    details={"error": str(e), "rule": rule},
                    severity="medium",
                    message=f"Business rule error: {str(e)}"
                ))
        
        return business_results
    
    async def _evaluate_single_business_rule(
        self, 
        test_execution: TestExecution, 
        rule: Dict[str, Any]
    ) -> OracleResult:
        """Evaluate a single business rule."""
        
        rule_name = rule.get("name", "unnamed_rule")
        rule_expression = rule.get("expression", "")
        rule_severity = rule.get("severity", "medium")
        
        try:
            # Prepare evaluation context
            context = {
                "response": test_execution.response_body,
                "status_code": test_execution.response_code,
                "headers": test_execution.response_headers,
                "response_time_ms": test_execution.response_time_ms
            }
            
            # Evaluate the rule expression safely
            result = self._safe_evaluate_expression(rule_expression, context)
            
            return OracleResult(
                oracle_type=OracleType.BUSINESS_RULE,
                passed=bool(result),
                details={
                    "rule_name": rule_name,
                    "expression": rule_expression,
                    "evaluation_result": result
                },
                severity=rule_severity,
                message=f"Business rule '{rule_name}' {'passed' if result else 'failed'}"
            )
            
        except Exception as e:
            return OracleResult(
                oracle_type=OracleType.BUSINESS_RULE,
                passed=False,
                details={"error": str(e), "rule_name": rule_name},
                severity=rule_severity,
                message=f"Business rule '{rule_name}' evaluation error: {str(e)}"
            )
    
    async def _evaluate_security_oracle(self, test_execution: TestExecution) -> OracleResult:
        """Evaluate security aspects of the response."""
        
        try:
            violations = []
            
            # Check for sensitive data leakage
            if test_execution.response_body:
                leakage_issues = self._check_for_sensitive_data_leakage(test_execution.response_body)
                violations.extend(leakage_issues)
            
            # Check security headers
            if test_execution.response_headers:
                header_issues = self._check_security_headers(test_execution.response_headers)
                violations.extend(header_issues)
            
            # Check for error information leakage
            if test_execution.response_code >= 500:
                error_leakage = self._check_for_error_information_leakage(test_execution.response_body)
                violations.extend(error_leakage)
            
            passed = len(violations) == 0
            severity = "critical" if any("password" in v.lower() or "token" in v.lower() for v in violations) else "high" if violations else "low"
            
            return OracleResult(
                oracle_type=OracleType.SECURITY,
                passed=passed,
                details={
                    "violations": violations,
                    "checks_performed": ["sensitive_data", "security_headers", "error_leakage"]
                },
                severity=severity,
                message=f"Security validation {'passed' if passed else 'failed'}: {len(violations)} violations"
            )
            
        except Exception as e:
            logger.error(f"Security oracle evaluation failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.SECURITY,
                passed=False,
                details={"error": str(e)},
                severity="high",
                message=f"Security check error: {str(e)}"
            )
    
    async def _evaluate_performance_oracle(self, test_execution: TestExecution) -> OracleResult:
        """Evaluate performance characteristics."""
        
        try:
            violations = []
            response_time = test_execution.response_time_ms or 0
            
            # Performance thresholds (configurable)
            thresholds = {
                "GET": 1000,      # 1 second for GET requests
                "POST": 3000,     # 3 seconds for POST requests
                "PUT": 3000,      # 3 seconds for PUT requests
                "PATCH": 2000,    # 2 seconds for PATCH requests
                "DELETE": 2000    # 2 seconds for DELETE requests
            }
            
            method = test_execution.test_case.method.upper() if test_execution.test_case else "GET"
            threshold = thresholds.get(method, 2000)
            
            if response_time > threshold:
                violations.append(f"Response time ({response_time:.1f}ms) exceeds threshold ({threshold}ms)")
            
            # Check for performance regression indicators
            if response_time > 10000:  # 10 seconds
                violations.append(f"Extremely slow response time: {response_time:.1f}ms")
            
            # Check response size (large responses may indicate performance issues)
            if test_execution.response_body:
                response_size = len(str(test_execution.response_body))
                if response_size > 1000000:  # 1MB
                    violations.append(f"Large response size: {response_size} bytes")
            
            passed = len(violations) == 0
            severity = "high" if response_time > threshold * 3 else "medium" if violations else "low"
            
            return OracleResult(
                oracle_type=OracleType.PERFORMANCE,
                passed=passed,
                details={
                    "response_time_ms": response_time,
                    "threshold_ms": threshold,
                    "violations": violations,
                    "method": method
                },
                severity=severity,
                message=f"Performance check {'passed' if passed else 'failed'}: {response_time:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Performance oracle evaluation failed: {str(e)}")
            return OracleResult(
                oracle_type=OracleType.PERFORMANCE,
                passed=False,
                details={"error": str(e)},
                severity="medium",
                message=f"Performance check error: {str(e)}"
            )
    
    # Helper methods
    
    def _is_protected_endpoint(self, test_case: TestCase) -> bool:
        """Determine if an endpoint should require authentication."""
        
        if not test_case:
            return False
        
        # Heuristics for protected endpoints
        protected_patterns = [
            "/admin/", "/user/", "/account/", "/profile/", "/orders/",
            "/transactions/", "/payments/", "/billing/"
        ]
        
        public_patterns = [
            "/health", "/status", "/ping", "/public/", "/docs", "/openapi"
        ]
        
        endpoint = test_case.endpoint.lower()
        
        # Check if explicitly public
        if any(pattern in endpoint for pattern in public_patterns):
            return False
        
        # Check if explicitly protected
        if any(pattern in endpoint for pattern in protected_patterns):
            return True
        
        # Default: POST/PUT/DELETE are protected, GET is not
        return test_case.method.upper() in ["POST", "PUT", "DELETE", "PATCH"]
    
    def _is_paginated_endpoint(self, test_execution: TestExecution) -> bool:
        """Determine if an endpoint supports pagination."""
        
        if not test_execution.test_case:
            return False
        
        endpoint = test_execution.test_case.endpoint.lower()
        method = test_execution.test_case.method.upper()
        
        # Only GET requests can be paginated
        if method != "GET":
            return False
        
        # Check for pagination indicators in endpoint path
        pagination_indicators = [
            "/list", "/search", "/query", "/all", "/collection"
        ]
        
        return any(indicator in endpoint for indicator in pagination_indicators)
    
    def _check_for_auth_leakage(self, response_body: Any) -> List[str]:
        """Check for authentication token leakage in responses."""
        
        violations = []
        
        if isinstance(response_body, dict):
            violations.extend(self._check_dict_for_auth_leakage(response_body))
        elif isinstance(response_body, str):
            if self._contains_auth_tokens(response_body):
                violations.append("Response contains potential auth tokens")
        
        return violations
    
    def _check_dict_for_auth_leakage(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Recursively check dictionary for auth token leakage."""
        
        violations = []
        sensitive_keys = ["password", "token", "secret", "key", "auth", "jwt", "bearer"]
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check key names
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                violations.append(f"Sensitive field exposed: {current_path}")
            
            # Recursively check nested objects
            if isinstance(value, dict):
                violations.extend(self._check_dict_for_auth_leakage(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(self._check_dict_for_auth_leakage(item, f"{current_path}[{i}]"))
            elif isinstance(value, str) and self._contains_auth_tokens(value):
                violations.append(f"Potential token in field {current_path}")
        
        return violations
    
    def _contains_auth_tokens(self, text: str) -> bool:
        """Check if text contains potential authentication tokens."""
        
        # JWT pattern
        jwt_pattern = re.compile(r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*')
        
        # Bearer token pattern  
        bearer_pattern = re.compile(r'Bearer\s+[A-Za-z0-9-_=+/]+')
        
        # API key pattern
        api_key_pattern = re.compile(r'[Aa]pi[_-]?[Kk]ey[:\s]*[A-Za-z0-9-_=+/]{20,}')
        
        return bool(jwt_pattern.search(text) or bearer_pattern.search(text) or api_key_pattern.search(text))
    
    def _check_for_sensitive_data_leakage(self, response_body: Any) -> List[str]:
        """Check for various types of sensitive data leakage."""
        
        violations = []
        
        if isinstance(response_body, dict):
            # Check for common sensitive data patterns
            violations.extend(self._check_for_pii_leakage(response_body))
            violations.extend(self._check_for_system_info_leakage(response_body))
        elif isinstance(response_body, str):
            # Check string content for patterns
            if self._contains_stack_trace(response_body):
                violations.append("Response contains stack trace information")
            if self._contains_file_paths(response_body):
                violations.append("Response contains file system paths")
        
        return violations
    
    def _check_for_pii_leakage(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check for personally identifiable information leakage."""
        
        violations = []
        pii_patterns = {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "phone": re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b')
        }
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                violations.extend(self._check_for_pii_leakage(value, current_path))
            elif isinstance(value, str):
                for pii_type, pattern in pii_patterns.items():
                    if pattern.search(value):
                        violations.append(f"Potential {pii_type} in {current_path}")
        
        return violations
    
    def _check_for_system_info_leakage(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check for system information leakage."""
        
        violations = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                violations.extend(self._check_for_system_info_leakage(value, current_path))
            elif isinstance(value, str):
                # Check for database connection strings
                if "://" in value and any(db in value for db in ["mysql", "postgres", "mongodb"]):
                    violations.append(f"Database connection string in {current_path}")
                
                # Check for file paths
                if value.startswith("/") or (len(value) > 3 and value[1] == ":"):
                    violations.append(f"File system path in {current_path}")
        
        return violations
    
    def _check_security_headers(self, headers: Dict[str, Any]) -> List[str]:
        """Check for missing security headers."""
        
        violations = []
        
        recommended_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": None,  # Any value is good
            "Content-Security-Policy": None
        }
        
        for header, expected_values in recommended_headers.items():
            header_value = headers.get(header)
            
            if not header_value:
                violations.append(f"Missing security header: {header}")
            elif expected_values and isinstance(expected_values, list):
                if header_value not in expected_values:
                    violations.append(f"Invalid {header} value: {header_value}")
            elif expected_values and isinstance(expected_values, str):
                if header_value != expected_values:
                    violations.append(f"Invalid {header} value: {header_value}")
        
        return violations
    
    def _check_for_error_information_leakage(self, response_body: Any) -> List[str]:
        """Check for sensitive error information leakage in error responses."""
        
        violations = []
        
        if isinstance(response_body, dict):
            # Check for stack traces
            for key, value in response_body.items():
                if isinstance(value, str):
                    if self._contains_stack_trace(value):
                        violations.append(f"Stack trace in error field: {key}")
                    if self._contains_file_paths(value):
                        violations.append(f"File paths in error field: {key}")
                    if self._contains_sql_info(value):
                        violations.append(f"SQL information in error field: {key}")
        elif isinstance(response_body, str):
            if self._contains_stack_trace(response_body):
                violations.append("Stack trace in error response")
            if self._contains_file_paths(response_body):
                violations.append("File paths in error response")
        
        return violations
    
    def _contains_stack_trace(self, text: str) -> bool:
        """Check if text contains stack trace information."""
        
        stack_indicators = [
            "Traceback", "at line", "Exception in", "raised:", 
            ".py:", ".java:", ".cs:", "stackTrace"
        ]
        
        return any(indicator in text for indicator in stack_indicators)
    
    def _contains_file_paths(self, text: str) -> bool:
        """Check if text contains file system paths."""
        
        # Unix paths
        unix_path_pattern = re.compile(r'/[a-zA-Z0-9_.-]+(/[a-zA-Z0-9_.-]+)+')
        
        # Windows paths
        windows_path_pattern = re.compile(r'[A-Za-z]:\\[^<>:"|?*\n\r]+')
        
        return bool(unix_path_pattern.search(text) or windows_path_pattern.search(text))
    
    def _contains_sql_info(self, text: str) -> bool:
        """Check if text contains SQL-related information."""
        
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "JOIN"]
        return any(keyword in text.upper() for keyword in sql_keywords)
    
    def _normalize_response_for_comparison(self, response: Any) -> Any:
        """Normalize response data for comparison (remove dynamic fields)."""
        
        if isinstance(response, dict):
            normalized = {}
            for key, value in response.items():
                # Skip dynamic fields
                if key.lower() in ["timestamp", "request_id", "trace_id", "correlation_id", "generated_at"]:
                    continue
                normalized[key] = self._normalize_response_for_comparison(value)
            return normalized
        elif isinstance(response, list):
            return [self._normalize_response_for_comparison(item) for item in response]
        else:
            return response
    
    def _looks_like_date(self, key: str, value: str) -> bool:
        """Check if a field looks like it should contain a date."""
        
        date_keywords = ["date", "time", "created", "updated", "modified", "expires", "timestamp"]
        return any(keyword in key.lower() for keyword in date_keywords)
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if string is a valid date format."""
        
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO 8601
            r'^\d{4}-\d{2}-\d{2}$',                    # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',                    # MM/DD/YYYY
            r'^\d{10}$',                               # Unix timestamp
            r'^\d{13}$'                                # Unix timestamp (ms)
        ]
        
        return any(re.match(pattern, date_str) for pattern in date_patterns)
    
    def _is_reasonable_date(self, date_str: str) -> bool:
        """Check if date is within reasonable bounds."""
        
        try:
            # Simple check for obviously wrong dates
            if re.match(r'^\d{4}', date_str):
                year = int(date_str[:4])
                return 1900 <= year <= 2100
            return True
        except:
            return True
    
    def _check_date_ordering(self, data: Dict[str, Any], path: str = "") -> List[str]:
        """Check that start dates come before end dates."""
        
        violations = []
        
        # Find date pairs to validate
        date_pairs = [
            ("start_date", "end_date"),
            ("created_at", "updated_at"),
            ("valid_from", "valid_to"),
            ("begin_date", "expire_date")
        ]
        
        for start_field, end_field in date_pairs:
            if start_field in data and end_field in data:
                start_date = data[start_field]
                end_date = data[end_field]
                
                if isinstance(start_date, str) and isinstance(end_date, str):
                    # Simple string comparison for ISO dates
                    if start_date > end_date:
                        current_path = f"{path}." if path else ""
                        violations.append(f"Date ordering violation: {current_path}{start_field} > {current_path}{end_field}")
        
        return violations
    
    def _safe_evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a business rule expression."""
        
        # Define safe evaluation environment
        safe_env = {
            "response": context.get("response"),
            "status_code": context.get("status_code"),
            "headers": context.get("headers"),
            "response_time_ms": context.get("response_time_ms"),
            # Safe functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max
        }
        
        try:
            # Simple expression evaluation with restricted environment
            return eval(expression, {"__builtins__": {}}, safe_env)
        except Exception as e:
            logger.warning(f"Business rule evaluation failed: {expression} - {str(e)}")
            return False
    
    async def detect_invariant_violations(
        self,
        test_executions: List[TestExecution],
        api_spec_id: int
    ) -> List[InvariantViolation]:
        """Detect invariant violations across multiple test executions."""
        
        violations = []
        
        try:
            # Group executions by endpoint for invariant checking
            endpoint_groups = {}
            for execution in test_executions:
                if execution.test_case:
                    key = f"{execution.test_case.method} {execution.test_case.endpoint}"
                    if key not in endpoint_groups:
                        endpoint_groups[key] = []
                    endpoint_groups[key].append(execution)
            
            # Check invariants for each endpoint group
            for endpoint_key, executions in endpoint_groups.items():
                # Idempotency violations
                idempotency_violations = await self._detect_idempotency_violations(executions)
                violations.extend(idempotency_violations)
                
                # Auth consistency violations
                auth_violations = await self._detect_auth_consistency_violations(executions)
                violations.extend(auth_violations)
                
                # Data integrity violations
                integrity_violations = await self._detect_data_integrity_violations(executions)
                violations.extend(integrity_violations)
            
            return violations
            
        except Exception as e:
            logger.error(f"Invariant violation detection failed: {str(e)}")
            return []
    
    async def _detect_idempotency_violations(self, executions: List[TestExecution]) -> List[InvariantViolation]:
        """Detect idempotency violations in GET requests."""
        
        violations = []
        
        # Filter GET requests
        get_executions = [ex for ex in executions if ex.test_case and ex.test_case.method.upper() == "GET"]
        
        if len(get_executions) < 2:
            return violations
        
        # Compare responses for consistency
        base_execution = get_executions[0]
        base_response = self._normalize_response_for_comparison(base_execution.response_body)
        
        for execution in get_executions[1:]:
            normalized_response = self._normalize_response_for_comparison(execution.response_body)
            
            if base_response != normalized_response and execution.response_code == base_execution.response_code:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.IDEMPOTENCY,
                    endpoint=execution.test_case.endpoint,
                    method=execution.test_case.method,
                    violation_details={
                        "message": "GET request responses are not consistent",
                        "base_execution_id": base_execution.id,
                        "violating_execution_id": execution.id
                    },
                    test_data=execution.test_case.test_data or {},
                    response_data=execution.response_body or {},
                    severity="medium"
                ))
        
        return violations
    
    async def _detect_auth_consistency_violations(self, executions: List[TestExecution]) -> List[InvariantViolation]:
        """Detect authentication consistency violations."""
        
        violations = []
        
        # Group by auth status
        auth_executions = []
        no_auth_executions = []
        
        for execution in executions:
            if execution.test_case and execution.test_case.test_data:
                headers = execution.test_case.test_data.get("headers", {})
                has_auth = "Authorization" in headers or "authentication" in headers
                
                if has_auth:
                    auth_executions.append(execution)
                else:
                    no_auth_executions.append(execution)
        
        # Check that auth requirements are consistent
        auth_statuses = set(ex.response_code for ex in auth_executions)
        no_auth_statuses = set(ex.response_code for ex in no_auth_executions)
        
        # If both auth and no-auth requests succeed, there might be an issue
        if (200 in auth_statuses or 201 in auth_statuses) and (200 in no_auth_statuses or 201 in no_auth_statuses):
            # Check if this is actually a public endpoint
            if len(no_auth_executions) > 0:
                test_case = no_auth_executions[0].test_case
                if self._is_protected_endpoint(test_case):
                    violations.append(InvariantViolation(
                        invariant_type=InvariantType.AUTH_REQUIRED,
                        endpoint=test_case.endpoint,
                        method=test_case.method,
                        violation_details={
                            "message": "Protected endpoint accessible without authentication",
                            "auth_success_codes": list(auth_statuses),
                            "no_auth_success_codes": list(no_auth_statuses)
                        },
                        test_data=test_case.test_data or {},
                        response_data={},
                        severity="high"
                    ))
        
        return violations
    
    async def _detect_data_integrity_violations(self, executions: List[TestExecution]) -> List[InvariantViolation]:
        """Detect data integrity violations."""
        
        violations = []
        
        for execution in executions:
            if execution.response_body and isinstance(execution.response_body, dict):
                # Check for data consistency issues
                integrity_issues = self._check_business_logic_integrity(execution.response_body)
                
                for issue in integrity_issues:
                    violations.append(InvariantViolation(
                        invariant_type=InvariantType.DATA_INTEGRITY,
                        endpoint=execution.test_case.endpoint if execution.test_case else "unknown",
                        method=execution.test_case.method if execution.test_case else "unknown",
                        violation_details={
                            "message": issue,
                            "execution_id": execution.id
                        },
                        test_data=execution.test_case.test_data if execution.test_case else {},
                        response_data=execution.response_body,
                        severity="medium"
                    ))
        
        return violations
    
    async def create_oracle_summary(
        self, 
        oracle_results: List[OracleResult]
    ) -> Dict[str, Any]:
        """Create a summary of oracle evaluation results."""
        
        try:
            total_oracles = len(oracle_results)
            passed_oracles = sum(1 for r in oracle_results if r.passed)
            
            # Group by oracle type
            by_type = {}
            for result in oracle_results:
                oracle_type = result.oracle_type.value
                if oracle_type not in by_type:
                    by_type[oracle_type] = {"passed": 0, "failed": 0, "total": 0}
                
                by_type[oracle_type]["total"] += 1
                if result.passed:
                    by_type[oracle_type]["passed"] += 1
                else:
                    by_type[oracle_type]["failed"] += 1
            
            # Group by severity
            by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            for result in oracle_results:
                if not result.passed:
                    by_severity[result.severity] += 1
            
            # Extract all violations
            all_violations = []
            for result in oracle_results:
                if not result.passed and result.details.get("violations"):
                    all_violations.extend(result.details["violations"])
            
            # Calculate overall quality score
            quality_score = (passed_oracles / total_oracles * 100) if total_oracles > 0 else 100
            
            # Adjust quality score based on severity
            severity_penalties = {"critical": 25, "high": 15, "medium": 5, "low": 1}
            penalty = sum(severity_penalties.get(severity, 0) * count for severity, count in by_severity.items())
            quality_score = max(0, quality_score - penalty)
            
            return {
                "overall": {
                    "total_oracles": total_oracles,
                    "passed_oracles": passed_oracles,
                    "failed_oracles": total_oracles - passed_oracles,
                    "pass_rate": passed_oracles / total_oracles if total_oracles > 0 else 0,
                    "quality_score": quality_score
                },
                "by_type": by_type,
                "by_severity": by_severity,
                "violations": all_violations,
                "critical_issues": [
                    r.message for r in oracle_results 
                    if not r.passed and r.severity == "critical"
                ],
                "recommendations": self._generate_oracle_recommendations(oracle_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to create oracle summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_oracle_recommendations(self, oracle_results: List[OracleResult]) -> List[str]:
        """Generate recommendations based on oracle results."""
        
        recommendations = []
        
        # Analyze patterns in failed oracles
        failed_oracles = [r for r in oracle_results if not r.passed]
        
        if not failed_oracles:
            recommendations.append("All oracle checks passed - excellent API quality")
            return recommendations
        
        # Group failures by type
        failure_types = {}
        for oracle in failed_oracles:
            oracle_type = oracle.oracle_type.value
            if oracle_type not in failure_types:
                failure_types[oracle_type] = 0
            failure_types[oracle_type] += 1
        
        # Generate specific recommendations
        if failure_types.get("contract", 0) > 2:
            recommendations.append("Multiple contract violations detected - review API specification compliance")
        
        if failure_types.get("security", 0) > 0:
            recommendations.append("Security issues found - implement security headers and data sanitization")
        
        if failure_types.get("invariant", 0) > 1:
            recommendations.append("API invariant violations detected - review consistency of API behavior")
        
        if failure_types.get("performance", 0) > 0:
            recommendations.append("Performance issues detected - optimize response times")
        
        # Severity-based recommendations
        critical_count = sum(1 for r in failed_oracles if r.severity == "critical")
        if critical_count > 0:
            recommendations.append(f"{critical_count} critical issues require immediate attention")
        
        high_count = sum(1 for r in failed_oracles if r.severity == "high")
        if high_count > 2:
            recommendations.append(f"{high_count} high-severity issues should be prioritized")
        
        return recommendations
    
    def __del__(self):
        """Clean up database session."""
        if hasattr(self, 'db'):
            self.db.close()
