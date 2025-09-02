"""
Tests for API endpoints and FastAPI application.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.database.models import APISpec, TestCase, TestExecution


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestAPISpecEndpoints:
    """Test API specification endpoints."""

    def test_create_api_spec(self, client, db_session, sample_openapi_spec):
        """Test creating a new API specification."""
        response = client.post(
            "/api/specs",
            json={
                "name": "Test API",
                "description": "Test API for testing",
                "spec_content": sample_openapi_spec
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test API"
        assert data["description"] == "Test API for testing"
        assert "id" in data

    def test_get_api_specs(self, client, db_session):
        """Test retrieving API specifications."""
        # Create a test spec first
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        response = client.get("/api/specs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["name"] == "Test API"

    def test_get_api_spec_by_id(self, client, db_session):
        """Test retrieving a specific API specification."""
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        response = client.get(f"/api/specs/{spec.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test API"
        assert data["id"] == spec.id

    def test_get_nonexistent_api_spec(self, client):
        """Test retrieving a non-existent API specification."""
        response = client.get("/api/specs/999")
        assert response.status_code == 404

    def test_update_api_spec(self, client, db_session):
        """Test updating an API specification."""
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        response = client.put(
            f"/api/specs/{spec.id}",
            json={
                "name": "Updated API",
                "description": "Updated description"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated API"
        assert data["description"] == "Updated description"

    def test_delete_api_spec(self, client, db_session):
        """Test deleting an API specification."""
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        spec_id = spec.id

        response = client.delete(f"/api/specs/{spec_id}")
        assert response.status_code == 204

        # Verify spec is deleted
        response = client.get(f"/api/specs/{spec_id}")
        assert response.status_code == 404


class TestTestCaseEndpoints:
    """Test test case endpoints."""

    @patch('src.services.ai_test_generator.AITestGenerator.generate_tests')
    def test_generate_test_cases(self, mock_generate, client, db_session, sample_openapi_spec):
        """Test generating test cases from API spec."""
        # Create API spec first
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content=sample_openapi_spec,
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        # Mock the AI generator
        mock_generate.return_value = [
            {
                "name": "Test GET /users",
                "method": "GET",
                "path": "/users",
                "expected_status": 200
            }
        ]

        response = client.post(f"/api/specs/{spec.id}/generate-tests")
        assert response.status_code == 201
        data = response.json()
        assert len(data) >= 1
        assert data[0]["name"] == "Test GET /users"

    def test_get_test_cases(self, client, db_session):
        """Test retrieving test cases."""
        # Create API spec and test case
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case 1",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()

        response = client.get(f"/api/specs/{spec.id}/test-cases")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["name"] == "Test Case 1"


class TestExecutionEndpoints:
    """Test execution endpoints."""

    @patch('src.services.test_executor.TestExecutor.execute_test')
    def test_execute_test_case(self, mock_execute, client, db_session):
        """Test executing a single test case."""
        # Create API spec and test case
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test Case 1",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()

        # Mock execution result
        mock_execute.return_value = {
            "status": "passed",
            "response_time": 150.0,
            "status_code": 200,
            "response_body": {"users": []},
            "assertions": [{"type": "status_code", "passed": True}]
        }

        response = client.post(f"/api/test-cases/{test_case.id}/execute")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        assert data["response_time"] == 150.0

    def test_get_test_executions(self, client, db_session):
        """Test retrieving test executions."""
        # Create test execution
        execution = TestExecution(
            test_case_id=1,
            status="passed",
            response_time=200.0,
            status_code=200,
            response_body={"result": "success"},
            assertions_passed=1,
            assertions_failed=0
        )
        db_session.add(execution)
        db_session.commit()

        response = client.get("/api/executions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["status"] == "passed"


class TestCoverageEndpoints:
    """Test coverage endpoints."""

    def test_get_coverage_report(self, client, db_session):
        """Test retrieving coverage report."""
        # Create some test data
        spec = APISpec(
            name="Test API",
            description="Test description",
            spec_content={"openapi": "3.0.0"},
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()

        response = client.get(f"/api/specs/{spec.id}/coverage")
        assert response.status_code == 200
        data = response.json()
        assert "endpoint_coverage" in data
        assert "method_coverage" in data
        assert "overall_coverage" in data


class TestMLEndpoints:
    """Test machine learning endpoints."""

    def test_get_rl_metrics(self, client):
        """Test retrieving RL optimization metrics."""
        response = client.get("/api/rl/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_episodes" in data
        assert "avg_reward" in data
        assert "exploration_rate" in data

    @patch('src.services.rl_optimizer.RLOptimizer.train_episode')
    def test_trigger_rl_training(self, mock_train, client):
        """Test triggering RL training."""
        mock_train.return_value = {"reward": 0.85, "actions_taken": 5}
        
        response = client.post("/api/rl/train")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training_completed"


class TestHealingEndpoints:
    """Test self-healing endpoints."""

    def test_get_healing_suggestions(self, client, db_session):
        """Test retrieving healing suggestions."""
        # Create a failed execution
        execution = TestExecution(
            test_case_id=1,
            status="failed",
            response_time=5000.0,
            status_code=500,
            response_body={"error": "Internal server error"},
            assertions_passed=0,
            assertions_failed=1,
            error_message="Request timeout"
        )
        db_session.add(execution)
        db_session.commit()

        response = client.get(f"/api/executions/{execution.id}/healing-suggestions")
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) > 0

    @patch('src.services.self_healing.SelfHealingSystem.apply_healing_suggestion')
    def test_apply_healing_suggestion(self, mock_apply, client, db_session):
        """Test applying a healing suggestion."""
        execution = TestExecution(
            test_case_id=1,
            status="failed",
            response_time=5000.0,
            status_code=500,
            response_body={"error": "Internal server error"},
            assertions_passed=0,
            assertions_failed=1,
            error_message="Request timeout"
        )
        db_session.add(execution)
        db_session.commit()

        mock_apply.return_value = {"status": "applied", "changes": ["Increased timeout to 10s"]}

        response = client.post(
            f"/api/executions/{execution.id}/apply-healing",
            json={"suggestion_id": "timeout_fix"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "applied"
