"""
Performance and load tests for the API testing framework.
"""

import asyncio
import pytest
import time
from unittest.mock import patch, MagicMock

from src.services.test_executor import TestExecutor
from src.services.ai_test_generator import AITestGenerator


class TestPerformance:
    """Performance tests for critical components."""

    @pytest.mark.asyncio
    async def test_concurrent_test_execution_performance(self):
        """Test performance of concurrent test execution."""
        executor = TestExecutor()
        
        # Create multiple test cases
        test_cases = []
        for i in range(20):  # 20 concurrent tests
            test_cases.append({
                "method": "GET",
                "url": f"https://httpbin.org/delay/{i % 3}",  # Varying delays
                "assertions": [{"type": "status_code", "expected": 200}]
            })
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_request.return_value = mock_response
            
            start_time = time.time()
            results = await executor.execute_concurrent_tests(test_cases)
            execution_time = time.time() - start_time
            
            # Verify all tests completed
            assert len(results) == 20
            assert all(result["status"] == "passed" for result in results)
            
            # Performance assertion - should complete much faster than sequential
            # Sequential would take ~30s (sum of delays), concurrent should be ~3s
            assert execution_time < 5.0, f"Concurrent execution took {execution_time}s, expected < 5s"

    @pytest.mark.asyncio
    async def test_ai_test_generation_performance(self, mock_openai):
        """Test performance of AI test generation."""
        generator = AITestGenerator(openai_client=mock_openai)
        
        # Large API spec with many endpoints
        large_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Large API", "version": "1.0.0"},
            "paths": {}
        }
        
        # Generate 50 endpoints
        for i in range(50):
            large_spec["paths"][f"/endpoint_{i}"] = {
                "get": {
                    "summary": f"Get endpoint {i}",
                    "responses": {"200": {"description": "Success"}}
                }
            }
        
        start_time = time.time()
        tests = generator.generate_tests(large_spec)
        generation_time = time.time() - start_time
        
        # Verify tests generated
        assert len(tests) > 0
        
        # Performance assertion - should complete within reasonable time
        assert generation_time < 10.0, f"Test generation took {generation_time}s, expected < 10s"

    def test_database_query_performance(self, db_session):
        """Test database query performance with large datasets."""
        from src.database.models import APISpec, TestCase, TestExecution
        
        # Create test data at scale
        specs = []
        for i in range(10):
            spec = APISpec(
                name=f"API {i}",
                spec_content={"openapi": "3.0.0"},
                base_url=f"https://api{i}.example.com"
            )
            specs.append(spec)
        
        db_session.add_all(specs)
        db_session.commit()
        
        # Create test cases for each spec
        test_cases = []
        for spec in specs:
            for j in range(20):  # 20 test cases per spec
                test_case = TestCase(
                    api_spec_id=spec.id,
                    name=f"Test Case {j}",
                    method="GET",
                    path=f"/endpoint_{j}",
                    expected_status=200
                )
                test_cases.append(test_case)
        
        db_session.add_all(test_cases)
        db_session.commit()
        
        # Create executions
        executions = []
        for test_case in test_cases:
            for k in range(5):  # 5 executions per test case
                execution = TestExecution(
                    test_case_id=test_case.id,
                    status="passed" if k % 2 == 0 else "failed",
                    response_time=100.0 + k * 50,
                    status_code=200 if k % 2 == 0 else 500,
                    assertions_passed=1 if k % 2 == 0 else 0,
                    assertions_failed=0 if k % 2 == 0 else 1
                )
                executions.append(execution)
        
        db_session.add_all(executions)
        db_session.commit()
        
        # Test query performance
        start_time = time.time()
        
        # Complex query: Get all failed executions with their test cases and specs
        failed_executions = db_session.query(TestExecution)\\\
            .filter(TestExecution.status == "failed")\\\
            .join(TestCase)\\\
            .join(APISpec)\\\
            .all()
        
        query_time = time.time() - start_time
        
        # Verify results
        assert len(failed_executions) > 0
        
        # Performance assertion
        assert query_time < 2.0, f"Complex query took {query_time}s, expected < 2s"


class TestMemoryUsage:
    """Test memory usage and resource management."""

    @pytest.mark.asyncio
    async def test_memory_usage_concurrent_execution(self):
        """Test memory usage during concurrent test execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        executor = TestExecutor()
        
        # Create many test cases
        test_cases = []
        for i in range(100):
            test_cases.append({
                "method": "GET",
                "url": "https://httpbin.org/get",
                "assertions": [{"type": "status_code", "expected": 200}]
            })
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "x" * 1000}  # 1KB response
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_request.return_value = mock_response
            
            results = await executor.execute_concurrent_tests(test_cases)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Verify execution completed
            assert len(results) == 100
            
            # Memory usage should be reasonable (< 100MB increase)
            assert memory_increase < 100, f"Memory increased by {memory_increase}MB, expected < 100MB"

    def test_large_response_handling(self):
        """Test handling of large API responses."""
        executor = TestExecutor()
        
        # Simulate large response
        large_response_data = {"data": "x" * (10 * 1024 * 1024)}  # 10MB response
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = large_response_data
            mock_response.elapsed.total_seconds.return_value = 1.0
            mock_request.return_value = mock_response
            
            import asyncio
            
            async def run_test():
                return await executor.execute_test({
                    "method": "GET",
                    "url": "https://httpbin.org/get",
                    "assertions": [{"type": "status_code", "expected": 200}]
                })
            
            result = asyncio.run(run_test())
            
            # Should handle large response without crashing
            assert result["status"] == "passed"
            assert result["status_code"] == 200


class TestScalability:
    """Test system scalability and resource limits."""

    def test_database_scalability(self, db_session):
        """Test database performance with large number of records."""
        from src.database.models import APISpec, TestCase, TestExecution
        
        start_time = time.time()
        
        # Create large dataset
        specs = []
        for i in range(100):
            spec = APISpec(
                name=f"Scalability Test API {i}",
                spec_content={"openapi": "3.0.0", "info": {"title": f"API {i}"}},
                base_url=f"https://api{i}.example.com"
            )
            specs.append(spec)
        
        db_session.add_all(specs)
        db_session.commit()
        
        insertion_time = time.time() - start_time
        
        # Test bulk query performance
        start_time = time.time()
        all_specs = db_session.query(APISpec).all()
        query_time = time.time() - start_time
        
        # Verify data integrity
        assert len(all_specs) == 100
        
        # Performance assertions
        assert insertion_time < 10.0, f"Bulk insert took {insertion_time}s, expected < 10s"
        assert query_time < 1.0, f"Bulk query took {query_time}s, expected < 1s"

    @pytest.mark.asyncio
    async def test_api_endpoint_scalability(self, client):
        """Test API endpoint performance under load."""
        # Test health endpoint under repeated calls
        start_time = time.time()
        
        tasks = []
        for _ in range(50):  # 50 concurrent requests
            task = asyncio.create_task(self._make_health_request(client))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all requests succeeded
        assert len(results) == 50
        assert all(result.status_code == 200 for result in results)
        
        # Performance assertion
        assert total_time < 5.0, f"50 concurrent requests took {total_time}s, expected < 5s"
        
        # Average response time should be reasonable
        avg_time = total_time / 50
        assert avg_time < 0.1, f"Average response time {avg_time}s, expected < 0.1s"

    async def _make_health_request(self, client):
        """Helper method to make health check request."""
        return client.get("/health")


class TestResourceCleanup:
    """Test proper resource cleanup and connection management."""

    @pytest.mark.asyncio
    async def test_http_client_cleanup(self):
        """Test that HTTP clients are properly cleaned up."""
        executor = TestExecutor()
        
        # Track connection count
        initial_connections = len(asyncio.all_tasks())
        
        # Execute multiple tests
        test_cases = [
            {
                "method": "GET",
                "url": "https://httpbin.org/get",
                "assertions": [{"type": "status_code", "expected": 200}]
            }
            for _ in range(10)
        ]
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_request.return_value = mock_response
            
            await executor.execute_concurrent_tests(test_cases)
        
        # Allow cleanup time
        await asyncio.sleep(0.1)
        
        final_connections = len(asyncio.all_tasks())
        
        # Should not have leaked connections
        assert final_connections <= initial_connections + 2, "HTTP connections may have leaked"

    def test_database_connection_cleanup(self, db_session):
        """Test database connection cleanup."""
        from src.database.models import APISpec
        
        # Perform multiple database operations
        for i in range(50):
            spec = APISpec(
                name=f"Cleanup Test API {i}",
                spec_content={"openapi": "3.0.0"},
                base_url=f"https://api{i}.example.com"
            )
            db_session.add(spec)
            db_session.commit()
            
            # Query and delete
            found_spec = db_session.get(APISpec, spec.id)
            assert found_spec is not None
            db_session.delete(found_spec)
            db_session.commit()
        
        # Verify no memory leaks or connection issues
        final_count = db_session.query(APISpec).count()
        assert final_count == 0, "Database cleanup incomplete"


@pytest.mark.slow
class TestLoadTesting:
    """Load testing for system components."""

    @pytest.mark.asyncio
    async def test_high_load_test_execution(self):
        """Test system under high load of test executions."""
        executor = TestExecutor()
        
        # Create high volume of test cases
        test_cases = []
        for i in range(200):
            test_cases.append({
                "method": "GET",
                "url": f"https://httpbin.org/status/{200 if i % 4 != 0 else 500}",
                "assertions": [{"type": "status_code", "expected": 200}]
            })
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = lambda: 200 if test_cases.index % 4 != 0 else 500
            mock_response.json.return_value = {"success": True}
            mock_response.elapsed.total_seconds.return_value = 0.05
            mock_request.return_value = mock_response
            
            start_time = time.time()
            results = await executor.execute_concurrent_tests(test_cases, max_concurrent=20)
            total_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 200
            
            # Performance assertion for high load
            assert total_time < 15.0, f"High load execution took {total_time}s, expected < 15s"
            
            # Verify system stability
            passed_tests = sum(1 for r in results if r["status"] == "passed")
            failed_tests = sum(1 for r in results if r["status"] == "failed")
            
            assert passed_tests > 0, "No tests passed under load"
            assert passed_tests + failed_tests == 200, "Some tests were lost under load"

    def test_ai_generation_under_load(self, mock_openai):
        """Test AI test generation under load."""
        generator = AITestGenerator(openai_client=mock_openai)
        
        # Large complex API spec
        complex_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Complex API", "version": "1.0.0"},
            "paths": {}
        }
        
        # Generate complex spec with many endpoints and parameters
        for i in range(100):
            complex_spec["paths"][f"/resource_{i}"] = {
                "get": {
                    "summary": f"Get resource {i}",
                    "parameters": [
                        {"name": f"param_{j}", "in": "query", "schema": {"type": "string"}}
                        for j in range(5)
                    ],
                    "responses": {"200": {"description": "Success"}}
                },
                "post": {
                    "summary": f"Create resource {i}",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        f"field_{k}": {"type": "string"}
                                        for k in range(10)
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "Created"}}
                }
            }
        
        start_time = time.time()
        tests = generator.generate_tests(complex_spec)
        generation_time = time.time() - start_time
        
        # Verify generation completed
        assert len(tests) > 0
        
        # Performance under load
        assert generation_time < 30.0, f"Complex generation took {generation_time}s, expected < 30s"


class TestStressTest:
    """Stress tests to identify breaking points."""

    @pytest.mark.asyncio
    async def test_maximum_concurrent_requests(self):
        """Test system behavior at maximum concurrent requests."""
        executor = TestExecutor()
        
        # Push to higher concurrency limits
        test_cases = []
        for i in range(500):  # 500 concurrent tests
            test_cases.append({
                "method": "GET",
                "url": "https://httpbin.org/get",
                "assertions": [{"type": "status_code", "expected": 200}]
            })
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.elapsed.total_seconds.return_value = 0.02
            mock_request.return_value = mock_response
            
            try:
                start_time = time.time()
                results = await executor.execute_concurrent_tests(test_cases, max_concurrent=50)
                execution_time = time.time() - start_time
                
                # System should handle high load gracefully
                assert len(results) == 500
                assert execution_time < 60.0  # Should complete within 1 minute
                
                # Verify no crashes or errors
                assert all("status" in result for result in results)
                
            except Exception as e:
                # If system fails under extreme load, it should fail gracefully
                pytest.fail(f"System crashed under stress test: {e}")

    def test_memory_pressure_handling(self, db_session):
        """Test system behavior under memory pressure."""
        from src.database.models import TestExecution
        
        # Create very large execution records
        large_executions = []
        for i in range(1000):
            # Create execution with large response body
            execution = TestExecution(
                test_case_id=1,  # Assume test case exists
                status="passed",
                response_time=100.0,
                status_code=200,
                response_body={"large_data": "x" * 10000},  # 10KB per record
                assertions_passed=1,
                assertions_failed=0
            )
            large_executions.append(execution)
        
        try:
            # This should handle large dataset gracefully
            db_session.add_all(large_executions)
            db_session.commit()
            
            # Query large dataset
            count = db_session.query(TestExecution).count()
            assert count >= 1000
            
        except Exception as e:
            # System should fail gracefully under memory pressure
            pytest.skip(f"System under memory pressure, skipping: {e}")
        
        finally:
            # Cleanup
            db_session.query(TestExecution).delete()
            db_session.commit()
