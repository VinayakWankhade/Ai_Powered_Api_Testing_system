"""
Tests for core services and business logic.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.services.ai_test_generator import AITestGenerator
from src.services.rag_system import RAGSystem
from src.services.test_executor import TestExecutor
from src.services.coverage_analyzer import CoverageAnalyzer
from src.services.self_healing import SelfHealingSystem
from src.services.rl_optimizer import RLOptimizer


class TestAITestGenerator:
    """Test AI-powered test generation."""

    @pytest.fixture
    def ai_generator(self, mock_openai, mock_rag_system):
        """Create AI test generator with mocked dependencies."""
        return AITestGenerator(openai_client=mock_openai, rag_system=mock_rag_system)

    def test_generate_tests_from_spec(self, ai_generator, sample_openapi_spec):
        """Test generating tests from OpenAPI spec."""
        tests = ai_generator.generate_tests(sample_openapi_spec)
        
        assert len(tests) > 0
        test_case = tests[0]
        assert "name" in test_case
        assert "method" in test_case
        assert "path" in test_case
        assert "expected_status" in test_case

    def test_generate_tests_with_rag_context(self, ai_generator, sample_openapi_spec):
        """Test generating tests with RAG context."""
        # Mock RAG system response
        ai_generator.rag_system.query_similar_specs.return_value = [
            {"content": "Similar API pattern", "score": 0.85}
        ]
        
        tests = ai_generator.generate_tests(sample_openapi_spec, use_rag=True)
        
        assert len(tests) > 0
        # Verify RAG system was called
        ai_generator.rag_system.query_similar_specs.assert_called_once()

    def test_generate_edge_case_tests(self, ai_generator, sample_openapi_spec):
        """Test generating edge case tests."""
        tests = ai_generator.generate_edge_case_tests(sample_openapi_spec)
        
        assert len(tests) > 0
        # Edge case tests should include various scenarios
        test_names = [test["name"] for test in tests]
        assert any("edge" in name.lower() or "boundary" in name.lower() for name in test_names)


class TestRAGSystem:
    """Test RAG (Retrieval-Augmented Generation) system."""

    @pytest.fixture
    def rag_system(self):
        """Create RAG system for testing."""
        with patch('chromadb.Client') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            rag_system = RAGSystem()
            return rag_system

    def test_add_api_spec(self, rag_system, sample_openapi_spec):
        """Test adding API spec to RAG system."""
        rag_system.add_api_spec("test_spec", sample_openapi_spec)
        
        # Verify collection.add was called
        rag_system.collection.add.assert_called_once()

    def test_query_similar_specs(self, rag_system):
        """Test querying similar specs."""
        # Mock query response
        rag_system.collection.query.return_value = {
            'documents': [['Sample API documentation']],
            'distances': [[0.1]],
            'metadatas': [[{'spec_id': 'test_spec'}]]
        }
        
        results = rag_system.query_similar_specs("GET /users endpoint", n_results=1)
        
        assert len(results) == 1
        assert results[0]['content'] == 'Sample API documentation'
        assert results[0]['score'] > 0.8  # High similarity


class TestTestExecutor:
    """Test test execution engine."""

    @pytest.fixture
    def executor(self):
        """Create test executor."""
        return TestExecutor()

    @pytest.mark.asyncio
    async def test_execute_single_test(self, executor):
        """Test executing a single test case."""
        test_case = {
            "method": "GET",
            "url": "https://httpbin.org/get",
            "headers": {"Content-Type": "application/json"},
            "query_params": {},
            "body": None,
            "assertions": [
                {"type": "status_code", "expected": 200}
            ]
        }
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.elapsed.total_seconds.return_value = 0.15
            mock_request.return_value = mock_response
            
            result = await executor.execute_test(test_case)
            
            assert result["status"] == "passed"
            assert result["status_code"] == 200
            assert result["response_time"] > 0

    @pytest.mark.asyncio
    async def test_execute_failed_test(self, executor):
        """Test executing a test that fails."""
        test_case = {
            "method": "GET",
            "url": "https://httpbin.org/status/500",
            "assertions": [
                {"type": "status_code", "expected": 200}
            ]
        }
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Internal Server Error"}
            mock_response.elapsed.total_seconds.return_value = 0.5
            mock_request.return_value = mock_response
            
            result = await executor.execute_test(test_case)
            
            assert result["status"] == "failed"
            assert result["status_code"] == 500

    @pytest.mark.asyncio
    async def test_execute_concurrent_tests(self, executor):
        """Test executing multiple tests concurrently."""
        test_cases = [
            {
                "method": "GET",
                "url": "https://httpbin.org/get",
                "assertions": [{"type": "status_code", "expected": 200}]
            },
            {
                "method": "GET", 
                "url": "https://httpbin.org/delay/1",
                "assertions": [{"type": "status_code", "expected": 200}]
            }
        ]
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_request.return_value = mock_response
            
            results = await executor.execute_concurrent_tests(test_cases)
            
            assert len(results) == 2
            assert all(result["status"] == "passed" for result in results)


class TestCoverageAnalyzer:
    """Test coverage analysis system."""

    @pytest.fixture
    def coverage_analyzer(self, db_session):
        """Create coverage analyzer."""
        return CoverageAnalyzer(db_session)

    def test_calculate_endpoint_coverage(self, coverage_analyzer, sample_openapi_spec):
        """Test calculating endpoint coverage."""
        # Mock some test executions
        with patch.object(coverage_analyzer, '_get_executed_endpoints') as mock_get:
            mock_get.return_value = [("/users", "GET")]
            
            coverage = coverage_analyzer.calculate_endpoint_coverage(1, sample_openapi_spec)
            
            assert "endpoint_coverage" in coverage
            assert "total_endpoints" in coverage
            assert "covered_endpoints" in coverage
            assert coverage["total_endpoints"] == 2  # /users and /users/{user_id}
            assert coverage["covered_endpoints"] == 1

    def test_identify_coverage_gaps(self, coverage_analyzer, sample_openapi_spec):
        """Test identifying coverage gaps."""
        with patch.object(coverage_analyzer, '_get_executed_endpoints') as mock_get:
            mock_get.return_value = [("/users", "GET")]
            
            gaps = coverage_analyzer.identify_coverage_gaps(1, sample_openapi_spec)
            
            assert len(gaps) > 0
            # Should identify the missing /users/{user_id} endpoint
            assert any("/users/{user_id}" in gap["endpoint"] for gap in gaps)

    def test_get_coverage_recommendations(self, coverage_analyzer, sample_openapi_spec):
        """Test getting coverage improvement recommendations."""
        with patch.object(coverage_analyzer, '_get_executed_endpoints') as mock_get:
            mock_get.return_value = []  # No coverage
            
            recommendations = coverage_analyzer.get_coverage_recommendations(1, sample_openapi_spec)
            
            assert len(recommendations) > 0
            assert all("priority" in rec for rec in recommendations)
            assert all("reason" in rec for rec in recommendations)


class TestSelfHealingSystem:
    """Test self-healing system."""

    @pytest.fixture
    def healing_system(self, db_session, mock_openai):
        """Create self-healing system."""
        return SelfHealingSystem(db_session, mock_openai)

    def test_analyze_failure_patterns(self, healing_system):
        """Test analyzing failure patterns."""
        # Mock failed executions
        with patch.object(healing_system, '_get_recent_failures') as mock_get:
            mock_get.return_value = [
                {"error_message": "Timeout", "status_code": 500, "response_time": 5000},
                {"error_message": "Timeout", "status_code": 500, "response_time": 4500},
            ]
            
            patterns = healing_system.analyze_failure_patterns()
            
            assert len(patterns) > 0
            assert "timeout" in patterns[0]["pattern_type"].lower()

    def test_generate_healing_suggestions(self, healing_system):
        """Test generating healing suggestions."""
        failure_data = {
            "error_message": "Connection timeout",
            "status_code": 500,
            "response_time": 5000,
            "endpoint": "/users"
        }
        
        suggestions = healing_system.generate_healing_suggestions(failure_data)
        
        assert len(suggestions) > 0
        assert all("action" in suggestion for suggestion in suggestions)
        assert all("confidence" in suggestion for suggestion in suggestions)

    def test_apply_healing_suggestion(self, healing_system):
        """Test applying a healing suggestion."""
        suggestion = {
            "id": "timeout_fix",
            "action": "increase_timeout",
            "parameters": {"timeout": 10000}
        }
        
        result = healing_system.apply_healing_suggestion(1, suggestion)
        
        assert result["status"] == "applied"
        assert "changes" in result


class TestRLOptimizer:
    """Test reinforcement learning optimizer."""

    @pytest.fixture
    def rl_optimizer(self):
        """Create RL optimizer."""
        return RLOptimizer()

    def test_q_learning_update(self, rl_optimizer):
        """Test Q-learning state update."""
        state = [0.8, 0.2, 0.5]  # [pass_rate, avg_response_time, coverage]
        action = 1  # Select high-priority tests
        reward = 0.9
        next_state = [0.85, 0.18, 0.6]
        
        rl_optimizer.update_q_learning(state, action, reward, next_state)
        
        # Verify Q-table is updated (basic check)
        assert len(rl_optimizer.q_table) > 0

    def test_select_optimal_tests(self, rl_optimizer):
        """Test selecting optimal tests using RL."""
        test_candidates = [
            {"id": 1, "priority": 0.8, "execution_time": 100},
            {"id": 2, "priority": 0.6, "execution_time": 200},
            {"id": 3, "priority": 0.9, "execution_time": 150}
        ]
        
        selected = rl_optimizer.select_optimal_tests(test_candidates, budget=300)
        
        assert len(selected) > 0
        assert len(selected) <= len(test_candidates)
        # Should prefer high-priority tests within budget
        assert any(test["priority"] >= 0.8 for test in selected)

    def test_train_episode(self, rl_optimizer):
        """Test training a single RL episode."""
        execution_results = [
            {"status": "passed", "response_time": 150, "test_id": 1},
            {"status": "failed", "response_time": 300, "test_id": 2},
            {"status": "passed", "response_time": 100, "test_id": 3}
        ]
        
        result = rl_optimizer.train_episode(execution_results)
        
        assert "reward" in result
        assert "actions_taken" in result
        assert result["reward"] >= 0


class TestSpecIngestion:
    """Test API specification ingestion service."""

    @pytest.mark.asyncio
    async def test_parse_openapi_spec(self, sample_openapi_spec):
        """Test parsing OpenAPI specification."""
        from src.services.spec_ingestion import SpecIngestionService
        
        service = SpecIngestionService()
        parsed = await service.parse_openapi_spec(sample_openapi_spec)
        
        assert parsed["info"]["title"] == "Test API"
        assert len(parsed["paths"]) == 2
        assert "/users" in parsed["paths"]

    @pytest.mark.asyncio
    async def test_validate_spec(self, sample_openapi_spec):
        """Test validating OpenAPI specification."""
        from src.services.spec_ingestion import SpecIngestionService
        
        service = SpecIngestionService()
        is_valid, errors = await service.validate_spec(sample_openapi_spec)
        
        assert is_valid
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_extract_endpoints(self, sample_openapi_spec):
        """Test extracting endpoints from spec."""
        from src.services.spec_ingestion import SpecIngestionService
        
        service = SpecIngestionService()
        endpoints = await service.extract_endpoints(sample_openapi_spec)
        
        assert len(endpoints) == 2
        assert any(ep["path"] == "/users" and ep["method"] == "GET" for ep in endpoints)
        assert any(ep["path"] == "/users/{user_id}" and ep["method"] == "GET" for ep in endpoints)


class TestIntegration:
    """Integration tests for service interactions."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, db_session, mock_openai, sample_openapi_spec):
        """Test complete workflow from spec ingestion to test execution."""
        # 1. Ingest API spec
        from src.services.spec_ingestion import SpecIngestionService
        from src.database.models import APISpec
        
        ingestion_service = SpecIngestionService()
        parsed_spec = await ingestion_service.parse_openapi_spec(sample_openapi_spec)
        
        # Save to database
        spec = APISpec(
            name="Integration Test API",
            description="Test for integration",
            spec_content=parsed_spec,
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        # 2. Generate tests using AI
        ai_generator = AITestGenerator(openai_client=mock_openai)
        generated_tests = ai_generator.generate_tests(parsed_spec)
        
        assert len(generated_tests) > 0
        
        # 3. Execute tests
        executor = TestExecutor()
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"users": []}
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_request.return_value = mock_response
            
            # Execute first generated test
            test_case = generated_tests[0]
            result = await executor.execute_test({
                "method": test_case["method"],
                "url": f"https://api.example.com{test_case['path']}",
                "assertions": test_case.get("assertions", [])
            })
            
            assert result["status"] in ["passed", "failed"]
            assert "response_time" in result
            assert "status_code" in result

    def test_coverage_analysis_workflow(self, db_session, sample_openapi_spec):
        """Test coverage analysis workflow."""
        from src.database.models import APISpec, TestCase, TestExecution
        
        # Create test data
        spec = APISpec(
            name="Coverage Test API",
            spec_content=sample_openapi_spec,
            base_url="https://api.example.com"
        )
        db_session.add(spec)
        db_session.commit()
        
        test_case = TestCase(
            api_spec_id=spec.id,
            name="Test GET /users",
            method="GET",
            path="/users",
            expected_status=200
        )
        db_session.add(test_case)
        db_session.commit()
        
        execution = TestExecution(
            test_case_id=test_case.id,
            status="passed",
            response_time=150.0,
            status_code=200,
            response_body={"users": []},
            assertions_passed=1,
            assertions_failed=0
        )
        db_session.add(execution)
        db_session.commit()
        
        # Analyze coverage
        analyzer = CoverageAnalyzer(db_session)
        coverage = analyzer.calculate_endpoint_coverage(spec.id, sample_openapi_spec)
        
        assert coverage["total_endpoints"] == 2
        assert coverage["covered_endpoints"] == 1
        assert coverage["coverage_percentage"] == 50.0
        
        # Get recommendations
        recommendations = analyzer.get_coverage_recommendations(spec.id, sample_openapi_spec)
        assert len(recommendations) > 0
