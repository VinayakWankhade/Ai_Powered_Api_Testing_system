"""
Tests for database models and relationships.
"""

import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from src.database.models import APISpec, TestCase, TestExecution, CoverageReport, HealingSuggestion


class TestAPISpecModel:
    """Test APISpec model."""

    def test_create_api_spec(self, db_session):
        """Test creating an API specification."""
        spec = APISpec(
            name="Test API",
            description="A test API specification",
            spec_content={"openapi": "3.0.0", "info": {"title": "Test"}},
            base_url="https://api.example.com"
        )
        
        db_session.add(spec)
        db_session.commit()
        
        assert spec.id is not None
        assert spec.name == "Test API"
        assert spec.created_at is not None
        assert spec.updated_at is not None

    def test_api_spec_with_test_cases(self, db_session):
        """Test API spec with related test cases."""
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        # Add test cases
        test_case1 = TestCase(
            api_spec_id=spec.id,
            name="Test Case 1",
            method="GET",
            path="/users",
            expected_status=200
        )
        test_case2 = TestCase(
            api_spec_id=spec.id,
            name="Test Case 2", 
            method="POST",
            path="/users",
            expected_status=201
        )
        
        db_session.add_all([test_case1, test_case2])
        db_session.commit()
        
        # Verify relationship
        db_session.refresh(spec)
        assert len(spec.test_cases) == 2
        assert test_case1.api_spec == spec
        assert test_case2.api_spec == spec

    def test_api_spec_name_required(self, db_session):
        """Test that API spec name is required."""
        spec = APISpec(
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        
        db_session.add(spec)
        
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestTestCaseModel:
    """Test TestCase model."""

    def test_create_test_case(self, db_session):
        """Test creating a test case."""
        # Create API spec first
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        # Create test case
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test GET /users",
            method="GET",
            path="/users",
            headers={"Content-Type": "application/json"},
            query_params={"limit": "10"},
            body=None,
            expected_status=200,
            assertions=[
                {"type": "status_code", "expected": 200},
                {"type": "response_time", "max": 1000}
            ]
        )
        
        db_session.add(test_case)
        db_session.commit()
        
        assert test_case.id is not None
        assert test_case.name == "Test GET /users"
        assert test_case.method == "GET"
        assert test_case.path == "/users"
        assert test_case.expected_status == 200
        assert len(test_case.assertions) == 2

    def test_test_case_with_executions(self, db_session):
        """Test test case with related executions."""
        # Create API spec and test case
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        # Add executions
        execution1 = TestExecution(
            test_case_id=test_case.id,
            status="passed",
            response_time=150.0,
            status_code=200,
            response_body={"users": []},
            assertions_passed=1,
            assertions_failed=0
        )
        execution2 = TestExecution(
            test_case_id=test_case.id,
            status="failed",
            response_time=300.0,
            status_code=500,
            response_body={"error": "Server error"},
            assertions_passed=0,
            assertions_failed=1,
            error_message="Internal server error"
        )
        
        db_session.add_all([execution1, execution2])
        db_session.commit()
        
        # Verify relationship
        db_session.refresh(test_case)
        assert len(test_case.executions) == 2
        assert execution1.test_case == test_case
        assert execution2.test_case == test_case


class TestTestExecutionModel:
    """Test TestExecution model."""

    def test_create_test_execution(self, db_session):
        """Test creating a test execution."""
        # Create prerequisites
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        # Create execution
        execution = TestExecution(
            test_case_id=test_case.id,
            status="passed",
            response_time=225.5,
            status_code=200,
            response_body={"data": "test"},
            response_headers={"Content-Type": "application/json"},
            assertions_passed=2,
            assertions_failed=0,
            detailed_results={
                "assertions": [
                    {"type": "status_code", "passed": True},
                    {"type": "response_time", "passed": True}
                ]
            }
        )
        
        db_session.add(execution)
        db_session.commit()
        
        assert execution.id is not None
        assert execution.status == "passed"
        assert execution.response_time == 225.5
        assert execution.status_code == 200
        assert execution.assertions_passed == 2
        assert execution.assertions_failed == 0
        assert execution.executed_at is not None

    def test_execution_status_validation(self, db_session):
        """Test execution status validation."""
        # Create prerequisites
        spec = APISpec(
            name="Test API", 
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET", 
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        # Valid statuses should work
        valid_statuses = ["passed", "failed", "skipped", "error"]
        
        for status in valid_statuses:
            execution = TestExecution(
                test_case_id=test_case.id,
                status=status,
                response_time=100.0,
                status_code=200 if status == "passed" else 500,
                assertions_passed=1 if status == "passed" else 0,
                assertions_failed=0 if status == "passed" else 1
            )
            db_session.add(execution)
            db_session.commit()
            db_session.delete(execution)  # Clean up for next iteration


class TestCoverageReportModel:
    """Test CoverageReport model."""

    def test_create_coverage_report(self, db_session):
        """Test creating a coverage report."""
        # Create API spec
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        # Create coverage report
        report = CoverageReport(
            api_spec_id=spec.id,
            total_endpoints=10,
            covered_endpoints=7,
            coverage_percentage=70.0,
            uncovered_endpoints=[
                {"path": "/users/{id}", "method": "DELETE"},
                {"path": "/products", "method": "POST"},
                {"path": "/orders", "method": "PUT"}
            ],
            recommendations=[
                {
                    "endpoint": "/users/{id}",
                    "method": "DELETE", 
                    "priority": "high",
                    "reason": "Critical user management operation"
                }
            ]
        )
        
        db_session.add(report)
        db_session.commit()
        
        assert report.id is not None
        assert report.api_spec_id == spec.id
        assert report.total_endpoints == 10
        assert report.covered_endpoints == 7
        assert report.coverage_percentage == 70.0
        assert len(report.uncovered_endpoints) == 3
        assert len(report.recommendations) == 1
        assert report.generated_at is not None


class TestHealingSuggestionModel:
    """Test HealingSuggestion model."""

    def test_create_healing_suggestion(self, db_session):
        """Test creating a healing suggestion."""
        # Create prerequisites
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users", 
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        execution = TestExecution(
            test_case_id=test_case.id,
            status="failed",
            response_time=5000.0,
            status_code=500,
            assertions_passed=0,
            assertions_failed=1,
            error_message="Connection timeout"
        )
        db_session.add(execution)
        db_session.commit()
        
        # Create healing suggestion
        suggestion = HealingSuggestion(
            test_execution_id=execution.id,
            suggestion_type="timeout_adjustment",
            description="Increase request timeout to handle slow responses",
            action_data={
                "current_timeout": 5000,
                "suggested_timeout": 10000,
                "reasoning": "Multiple timeout failures detected"
            },
            confidence_score=0.85,
            priority="high"
        )
        
        db_session.add(suggestion)
        db_session.commit()
        
        assert suggestion.id is not None
        assert suggestion.test_execution_id == execution.id
        assert suggestion.suggestion_type == "timeout_adjustment"
        assert suggestion.confidence_score == 0.85
        assert suggestion.priority == "high"
        assert suggestion.status == "pending"  # Default status
        assert suggestion.created_at is not None

    def test_apply_healing_suggestion(self, db_session):
        """Test applying a healing suggestion."""
        # Create healing suggestion (simplified setup)
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        execution = TestExecution(
            test_case_id=test_case.id,
            status="failed",
            response_time=5000.0,
            status_code=500,
            assertions_passed=0,
            assertions_failed=1
        )
        db_session.add(execution)
        db_session.commit()
        
        suggestion = HealingSuggestion(
            test_execution_id=execution.id,
            suggestion_type="retry_mechanism",
            description="Add retry logic for transient failures",
            confidence_score=0.9
        )
        db_session.add(suggestion)
        db_session.commit()
        
        # Apply suggestion
        suggestion.status = "applied"
        suggestion.applied_at = datetime.utcnow()
        suggestion.application_result = {
            "changes_made": ["Added exponential backoff retry"],
            "success": True
        }
        
        db_session.commit()
        
        assert suggestion.status == "applied"
        assert suggestion.applied_at is not None
        assert suggestion.application_result["success"] is True


class TestModelRelationships:
    """Test relationships between models."""

    def test_cascade_delete_api_spec(self, db_session):
        """Test that deleting API spec cascades to related entities."""
        # Create full hierarchy
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        execution = TestExecution(
            test_case_id=test_case.id,
            status="passed",
            response_time=100.0,
            status_code=200,
            assertions_passed=1,
            assertions_failed=0
        )
        db_session.add(execution)
        db_session.commit()
        
        coverage_report = CoverageReport(
            api_spec_id=spec.id,
            total_endpoints=5,
            covered_endpoints=3,
            coverage_percentage=60.0
        )
        db_session.add(coverage_report)
        db_session.commit()
        
        # Store IDs for verification
        spec_id = spec.id
        test_case_id = test_case.id
        execution_id = execution.id
        coverage_id = coverage_report.id
        
        # Delete API spec
        db_session.delete(spec)
        db_session.commit()
        
        # Verify cascade delete
        assert db_session.get(APISpec, spec_id) is None
        assert db_session.get(TestCase, test_case_id) is None
        assert db_session.get(TestExecution, execution_id) is None
        assert db_session.get(CoverageReport, coverage_id) is None

    def test_test_case_execution_relationship(self, db_session):
        """Test relationship between test case and executions."""
        # Create test case
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        # Create multiple executions
        executions = []
        for i in range(3):
            execution = TestExecution(
                test_case_id=test_case.id,
                status="passed" if i % 2 == 0 else "failed",
                response_time=100.0 + i * 50,
                status_code=200 if i % 2 == 0 else 500,
                assertions_passed=1 if i % 2 == 0 else 0,
                assertions_failed=0 if i % 2 == 0 else 1
            )
            executions.append(execution)
        
        db_session.add_all(executions)
        db_session.commit()
        
        # Verify relationships
        db_session.refresh(test_case)
        assert len(test_case.executions) == 3
        
        # Verify back-references
        for execution in executions:
            db_session.refresh(execution)
            assert execution.test_case == test_case

    def test_execution_healing_relationship(self, db_session):
        """Test relationship between execution and healing suggestions."""
        # Create execution
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        execution = TestExecution(
            test_case_id=test_case.id,
            status="failed",
            response_time=5000.0,
            status_code=500,
            assertions_passed=0,
            assertions_failed=1,
            error_message="Timeout error"
        )
        db_session.add(execution)
        db_session.commit()
        
        # Create healing suggestions
        suggestions = []
        suggestion_types = ["timeout_adjustment", "retry_mechanism", "endpoint_modification"]
        
        for suggestion_type in suggestion_types:
            suggestion = HealingSuggestion(
                test_execution_id=execution.id,
                suggestion_type=suggestion_type,
                description=f"Apply {suggestion_type} to fix the issue",
                confidence_score=0.8
            )
            suggestions.append(suggestion)
        
        db_session.add_all(suggestions)
        db_session.commit()
        
        # Verify relationships
        db_session.refresh(execution)
        assert len(execution.healing_suggestions) == 3
        
        for suggestion in suggestions:
            db_session.refresh(suggestion)
            assert suggestion.test_execution == execution


class TestModelValidation:
    """Test model validation and constraints."""

    def test_response_time_positive(self, db_session):
        """Test that response time must be positive."""
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        # Negative response time should be invalid
        execution = TestExecution(
            test_case_id=test_case.id,
            status="failed",
            response_time=-100.0,  # Invalid negative time
            status_code=500,
            assertions_passed=0,
            assertions_failed=1
        )
        
        db_session.add(execution)
        
        # This should raise an integrity error or validation error
        with pytest.raises((IntegrityError, ValueError)):
            db_session.commit()

    def test_confidence_score_range(self, db_session):
        """Test that confidence score is within valid range."""
        # Create prerequisites
        spec = APISpec(
            name="Test API",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        execution = TestExecution(
            test_case_id=test_case.id,
            status="failed",
            response_time=100.0,
            status_code=500,
            assertions_passed=0,
            assertions_failed=1
        )
        db_session.add(execution)
        db_session.commit()
        
        # Valid confidence scores (0.0 to 1.0)
        valid_scores = [0.0, 0.5, 1.0]
        
        for score in valid_scores:
            suggestion = HealingSuggestion(
                test_execution_id=execution.id,
                suggestion_type="test_fix",
                description="Fix the test",
                confidence_score=score
            )
            db_session.add(suggestion)
            db_session.commit()
            db_session.delete(suggestion)  # Clean up
