# API Reference

This document provides comprehensive reference for the AI-Powered API Testing Framework REST API.

## Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

The API uses JWT tokens for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## API Endpoints

### Health Check

#### GET /health
Check the health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "ai_service": "healthy"
  }
}
```

### API Specifications

#### GET /api/specs
List all API specifications.

**Query Parameters:**
- `limit` (optional): Maximum number of results (default: 100)
- `offset` (optional): Number of results to skip (default: 0)
- `search` (optional): Search term for filtering specs

**Response:**
```json
[
  {
    "id": 1,
    "name": "User Management API",
    "description": "API for user management operations",
    "base_url": "https://api.example.com",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z",
    "test_case_count": 15,
    "last_execution": "2024-01-01T13:00:00Z"
  }
]
```

#### POST /api/specs
Create a new API specification.

**Request Body:**
```json
{
  "name": "My API",
  "description": "Description of the API",
  "spec_content": {
    "openapi": "3.0.0",
    "info": {
      "title": "My API",
      "version": "1.0.0"
    },
    "paths": {
      "/users": {
        "get": {
          "summary": "Get users",
          "responses": {
            "200": {
              "description": "Success"
            }
          }
        }
      }
    }
  },
  "base_url": "https://api.example.com"
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "name": "My API",
  "description": "Description of the API",
  "base_url": "https://api.example.com",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

#### GET /api/specs/{spec_id}
Get a specific API specification by ID.

**Response:**
```json
{
  "id": 1,
  "name": "User Management API",
  "description": "API for user management operations",
  "spec_content": {...},
  "base_url": "https://api.example.com",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

#### PUT /api/specs/{spec_id}
Update an existing API specification.

**Request Body:** Same as POST /api/specs

**Response:** `200 OK` with updated specification

#### DELETE /api/specs/{spec_id}
Delete an API specification and all related data.

**Response:** `204 No Content`

### Test Cases

#### GET /api/specs/{spec_id}/test-cases
Get all test cases for a specific API specification.

**Response:**
```json
[
  {
    "id": 1,
    "api_spec_id": 1,
    "name": "Test GET /users endpoint",
    "method": "GET",
    "path": "/users",
    "headers": {"Content-Type": "application/json"},
    "query_params": {"limit": "10"},
    "body": null,
    "expected_status": 200,
    "assertions": [
      {
        "type": "status_code",
        "expected": 200
      }
    ],
    "created_at": "2024-01-01T12:00:00Z"
  }
]
```

#### POST /api/specs/{spec_id}/generate-tests
Generate test cases using AI for the specified API.

**Request Body:**
```json
{
  "test_types": ["happy_path", "edge_cases", "error_scenarios"],
  "coverage_focus": ["endpoints", "methods", "parameters"],
  "use_rag": true,
  "max_tests": 50
}
```

**Response:** `201 Created`
```json
[
  {
    "id": 1,
    "name": "Test GET /users endpoint",
    "method": "GET",
    "path": "/users",
    "expected_status": 200,
    "generated_by": "ai",
    "confidence_score": 0.92
  }
]
```

#### POST /api/test-cases
Create a custom test case.

**Request Body:**
```json
{
  "api_spec_id": 1,
  "name": "Custom test case",
  "method": "POST",
  "path": "/users",
  "headers": {"Content-Type": "application/json"},
  "body": {
    "name": "John Doe",
    "email": "john@example.com"
  },
  "expected_status": 201,
  "assertions": [
    {
      "type": "status_code",
      "expected": 201
    },
    {
      "type": "json_path",
      "path": "$.id",
      "expected_type": "integer"
    }
  ]
}
```

**Response:** `201 Created` with created test case

#### GET /api/test-cases/{test_case_id}
Get a specific test case by ID.

#### PUT /api/test-cases/{test_case_id}
Update an existing test case.

#### DELETE /api/test-cases/{test_case_id}
Delete a test case.

### Test Execution

#### POST /api/test-cases/{test_case_id}/execute
Execute a single test case.

**Response:**
```json
{
  "id": 1,
  "test_case_id": 1,
  "status": "passed",
  "response_time": 245.5,
  "status_code": 200,
  "response_body": {"users": [...]},
  "response_headers": {"Content-Type": "application/json"},
  "assertions_passed": 2,
  "assertions_failed": 0,
  "detailed_results": {
    "assertions": [
      {
        "type": "status_code",
        "expected": 200,
        "actual": 200,
        "passed": true
      }
    ]
  },
  "executed_at": "2024-01-01T12:00:00Z"
}
```

#### POST /api/specs/{spec_id}/execute-all
Execute all test cases for an API specification.

**Request Body:**
```json
{
  "parallel": true,
  "max_concurrent": 10,
  "timeout": 30,
  "stop_on_failure": false
}
```

**Response:**
```json
{
  "execution_id": "exec_123",
  "total_tests": 25,
  "status": "running",
  "started_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

#### GET /api/executions
Get test execution history.

**Query Parameters:**
- `spec_id` (optional): Filter by API specification
- `status` (optional): Filter by execution status
- `limit`, `offset`: Pagination

#### GET /api/executions/{execution_id}
Get details of a specific test execution.

### Coverage Analysis

#### GET /api/specs/{spec_id}/coverage
Get coverage analysis for an API specification.

**Response:**
```json
{
  "api_spec_id": 1,
  "overall_coverage": 75.5,
  "endpoint_coverage": {
    "total_endpoints": 20,
    "covered_endpoints": 15,
    "percentage": 75.0
  },
  "method_coverage": {
    "GET": 90.0,
    "POST": 70.0,
    "PUT": 60.0,
    "DELETE": 40.0
  },
  "parameter_coverage": {
    "path_parameters": 85.0,
    "query_parameters": 70.0,
    "header_parameters": 60.0,
    "body_parameters": 80.0
  },
  "uncovered_endpoints": [
    {
      "path": "/users/{id}",
      "method": "DELETE",
      "priority": "high"
    }
  ],
  "recommendations": [
    {
      "type": "missing_endpoint",
      "endpoint": "/users/{id}",
      "method": "DELETE",
      "priority": "high",
      "reason": "Critical user management operation"
    }
  ],
  "generated_at": "2024-01-01T12:00:00Z"
}
```

#### GET /api/specs/{spec_id}/coverage/trends
Get coverage trends over time.

**Query Parameters:**
- `period`: `day`, `week`, `month` (default: `week`)
- `limit`: Number of data points

**Response:**
```json
{
  "period": "week",
  "data_points": [
    {
      "date": "2024-01-01",
      "overall_coverage": 70.0,
      "endpoint_coverage": 65.0,
      "test_count": 45
    }
  ]
}
```

### Self-Healing

#### GET /api/executions/{execution_id}/healing-suggestions
Get healing suggestions for a failed test execution.

**Response:**
```json
{
  "execution_id": 1,
  "suggestions": [
    {
      "id": "heal_001",
      "type": "timeout_adjustment",
      "description": "Increase request timeout to handle slow responses",
      "confidence_score": 0.85,
      "priority": "high",
      "action_data": {
        "current_timeout": 5000,
        "suggested_timeout": 10000,
        "reasoning": "Multiple timeout failures detected"
      },
      "estimated_impact": "Should fix 70% of similar failures"
    }
  ],
  "failure_pattern": {
    "pattern_type": "timeout",
    "frequency": 5,
    "success_rate_improvement": 0.7
  }
}
```

#### POST /api/executions/{execution_id}/apply-healing
Apply a healing suggestion.

**Request Body:**
```json
{
  "suggestion_id": "heal_001",
  "auto_rerun": true
}
```

**Response:**
```json
{
  "status": "applied",
  "changes": [
    "Increased timeout from 5s to 10s",
    "Added retry mechanism with exponential backoff"
  ],
  "rerun_result": {
    "status": "passed",
    "improvement": "Test now passes consistently"
  }
}
```

### Reinforcement Learning

#### GET /api/rl/metrics
Get RL optimization metrics.

**Response:**
```json
{
  "total_episodes": 1500,
  "avg_reward": 0.78,
  "exploration_rate": 0.15,
  "learning_rate": 0.01,
  "policy_performance": {
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.79
  },
  "recent_improvements": [
    {
      "metric": "test_efficiency",
      "improvement": 0.15,
      "period": "last_week"
    }
  ]
}
```

#### POST /api/rl/train
Trigger RL training episode.

**Request Body:**
```json
{
  "training_data": [
    {
      "test_id": 1,
      "execution_time": 150,
      "success": true,
      "coverage_improvement": 0.05
    }
  ],
  "algorithm": "q_learning"
}
```

**Response:**
```json
{
  "status": "training_completed",
  "episode_reward": 0.85,
  "actions_taken": 12,
  "policy_updated": true,
  "next_training": "2024-01-01T14:00:00Z"
}
```

#### POST /api/rl/optimize-selection
Get optimized test selection recommendations.

**Request Body:**
```json
{
  "api_spec_id": 1,
  "budget_ms": 30000,
  "coverage_target": 0.8,
  "priority_weights": {
    "coverage": 0.4,
    "execution_time": 0.3,
    "failure_rate": 0.3
  }
}
```

**Response:**
```json
{
  "selected_tests": [
    {
      "test_id": 5,
      "priority_score": 0.92,
      "estimated_time": 200,
      "coverage_contribution": 0.15
    }
  ],
  "total_estimated_time": 28500,
  "expected_coverage": 0.82,
  "confidence": 0.89
}
```

## WebSocket API

### Connection
Connect to `ws://localhost:8000/ws` for real-time updates.

### Message Types

#### Test Execution Updates
```json
{
  "type": "test_execution_update",
  "data": {
    "execution_id": 1,
    "test_case_id": 5,
    "status": "running",
    "progress": 0.65,
    "current_test": "GET /users/{id}"
  }
}
```

#### Coverage Updates
```json
{
  "type": "coverage_update",
  "data": {
    "api_spec_id": 1,
    "coverage_percentage": 78.5,
    "new_endpoints_covered": ["/users/{id}"]
  }
}
```

#### Healing Notifications
```json
{
  "type": "healing_applied",
  "data": {
    "execution_id": 1,
    "suggestion_id": "heal_001",
    "success": true,
    "improvement": "Test failure rate reduced by 70%"
  }
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "name",
      "reason": "Field is required"
    },
    "request_id": "req_12345"
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Request data validation failed
- `NOT_FOUND`: Requested resource not found
- `UNAUTHORIZED`: Authentication required or invalid
- `FORBIDDEN`: Insufficient permissions
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error
- `AI_SERVICE_ERROR`: AI service unavailable
- `EXECUTION_FAILED`: Test execution failed

## Rate Limiting

The API implements rate limiting:
- **Default**: 100 requests per minute per IP
- **Authenticated**: 1000 requests per minute per user
- **Headers**: Rate limit info included in response headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

## Pagination

List endpoints support pagination:

**Request:**
```
GET /api/specs?limit=20&offset=40
```

**Response Headers:**
```
X-Total-Count: 150
X-Page-Count: 8
Link: </api/specs?limit=20&offset=60>; rel="next"
```

## Filtering and Sorting

Most list endpoints support filtering and sorting:

**Query Parameters:**
- `sort`: Field to sort by (e.g., `created_at`, `name`)
- `order`: Sort order (`asc`, `desc`)
- `filter[field]`: Filter by field value

**Example:**
```
GET /api/specs?sort=created_at&order=desc&filter[status]=active
```

## Async Operations

Long-running operations return operation IDs for tracking:

**Response:**
```json
{
  "operation_id": "op_12345",
  "status": "pending",
  "created_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:05:00Z"
}
```

Check status with:
```
GET /api/operations/{operation_id}
```

## Bulk Operations

### Bulk Test Execution
```
POST /api/executions/bulk
```

**Request Body:**
```json
{
  "test_case_ids": [1, 2, 3, 4, 5],
  "execution_config": {
    "parallel": true,
    "max_concurrent": 5,
    "timeout": 30
  }
}
```

### Bulk Test Case Creation
```
POST /api/test-cases/bulk
```

**Request Body:**
```json
{
  "api_spec_id": 1,
  "test_cases": [
    {
      "name": "Test 1",
      "method": "GET",
      "path": "/users"
    }
  ]
}
```

## Advanced Features

### AI-Powered Features

#### Custom Prompts
```
POST /api/ai/custom-generation
```

**Request Body:**
```json
{
  "api_spec_id": 1,
  "custom_prompt": "Generate security-focused test cases",
  "focus_areas": ["authentication", "authorization", "input_validation"]
}
```

#### RAG Queries
```
POST /api/ai/rag-query
```

**Request Body:**
```json
{
  "query": "API testing patterns for user management",
  "max_results": 5,
  "similarity_threshold": 0.7
}
```

### Machine Learning

#### Model Training
```
POST /api/ml/train
```

**Request Body:**
```json
{
  "algorithm": "q_learning",
  "training_data_period": "last_30_days",
  "hyperparameters": {
    "learning_rate": 0.01,
    "discount_factor": 0.95
  }
}
```

#### Performance Predictions
```
POST /api/ml/predict
```

**Request Body:**
```json
{
  "test_configuration": {
    "api_spec_id": 1,
    "selected_tests": [1, 2, 3]
  }
}
```

**Response:**
```json
{
  "predicted_coverage": 0.85,
  "predicted_execution_time": 450,
  "confidence": 0.78,
  "recommendations": [
    "Add test case for DELETE /users/{id} to improve coverage"
  ]
}
```

## SDK Usage Examples

### Python SDK
```python
from api_testing_client import APITestingClient

client = APITestingClient("http://localhost:8000", api_key="your-key")

# Upload API spec
spec = client.specs.create(
    name="My API",
    spec_content=openapi_spec
)

# Generate tests
tests = client.specs.generate_tests(spec.id)

# Execute tests
results = client.executions.run_all(spec.id)

# Get coverage
coverage = client.coverage.get(spec.id)
```

### JavaScript SDK
```javascript
import { APITestingClient } from '@api-testing/client';

const client = new APITestingClient('http://localhost:8000', {
  apiKey: 'your-key'
});

// Upload and test API
const spec = await client.specs.create({
  name: 'My API',
  specContent: openApiSpec
});

const tests = await client.specs.generateTests(spec.id);
const results = await client.executions.runAll(spec.id);
```

## Metrics and Monitoring

### Prometheus Metrics
Available at `/metrics`:

- `api_requests_total`: Total API requests
- `api_request_duration_seconds`: Request duration
- `test_executions_total`: Total test executions
- `test_execution_duration_seconds`: Test execution time
- `ai_generation_requests_total`: AI generation requests
- `coverage_percentage`: Current coverage percentage

### Custom Metrics
```
GET /api/metrics/custom
```

Query custom metrics with PromQL-like syntax:
```
GET /api/metrics/custom?query=test_success_rate{api_spec_id="1"}[7d]
```

This API reference provides comprehensive coverage of all available endpoints and their usage patterns.
