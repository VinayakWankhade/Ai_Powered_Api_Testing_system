# Usage Examples

## Quick Start Guide

### 1. Setup and Installation

```bash
# Clone the repository
git clone https://github.com/VinayakWankhade/Ai_Powered_Api_Testing_system.git
cd Ai_Powered_Api_Testing_system

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations

# Initialize database
python -c "from src.database.connection import create_tables; create_tables()"

# Start the API service
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker Deployment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale services
docker-compose up -d --scale worker=3
```

### 3. Basic API Usage

#### Upload API Specification

```python
import requests

# Upload OpenAPI specification
response = requests.post("http://localhost:8000/api/v1/upload-spec", json={
    "name": "Pet Store API",
    "version": "1.0.0",
    "spec_type": "openapi",
    "spec_content": """{
        "openapi": "3.0.0",
        "info": {"title": "Pet Store", "version": "1.0.0"},
        "paths": {
            "/pets": {
                "get": {
                    "summary": "List pets",
                    "responses": {"200": {"description": "Success"}}
                }
            }
        }
    }""",
    "base_url": "https://petstore.swagger.io/v2"
})

api_spec_id = response.json()["id"]
print(f"API Spec uploaded with ID: {api_spec_id}")
```

#### Generate Test Cases

```python
# Generate AI-powered test cases
response = requests.post("http://localhost:8000/api/v1/generate-tests", json={
    "api_spec_id": api_spec_id,
    "endpoint_path": "/pets",
    "method": "GET",
    "test_types": ["functional", "edge_case"],
    "count": 5
})

test_cases = response.json()["test_cases"]
print(f"Generated {len(test_cases)} test cases")
```

#### Execute Tests

```python
# Run test execution session
response = requests.post("http://localhost:8000/api/v1/run-tests", json={
    "api_spec_id": api_spec_id,
    "session_name": "Automated Test Run"
})

session_id = response.json()["session_id"]
results = response.json()["results"]
print(f"Session {session_id}: {results['passed']} passed, {results['failed']} failed")
```

#### Self-Heal Failed Tests

```python
# Heal failed tests
if results['failed'] > 0:
    heal_response = requests.post("http://localhost:8000/api/v1/heal-tests", json={
        "session_id": session_id,
        "max_healing_attempts": 3,
        "auto_revalidate": True
    })
    
    healing_results = heal_response.json()
    print(f"Healed {healing_results['successfully_healed']} out of {healing_results['total_failed_tests']} failed tests")
```

#### RL Optimization

```python
# Optimize test selection using reinforcement learning
optimize_response = requests.post("http://localhost:8000/api/v1/optimize-tests", json={
    "api_spec_id": api_spec_id,
    "algorithm": "ppo",
    "training_episodes": 1000
})

optimization_results = optimize_response.json()
print(f"RL optimization completed with score: {optimization_results['optimization_score']}")
```

#### Get Coverage Report

```python
# Get coverage metrics
coverage_response = requests.get(f"http://localhost:8000/api/v1/coverage-report/{api_spec_id}")
coverage = coverage_response.json()["coverage"]

print(f"Endpoint Coverage: {coverage['endpoint_coverage_pct']:.1f}%")
print(f"Bugs Found: {coverage['bugs_found']}")
print(f"Quality Score: {coverage['quality_score']:.1f}")
```

### 4. Advanced Usage

#### Custom Test Generation with Context

```python
response = requests.post("http://localhost:8000/api/v1/generate-tests", json={
    "api_spec_id": api_spec_id,
    "endpoint_path": "/users/{id}",
    "method": "PUT",
    "test_types": ["functional", "security"],
    "count": 3,
    "custom_context": "This endpoint updates user profiles and requires authentication. Test for SQL injection and authorization bypass."
})
```

#### Batch Test Suite Generation

```python
response = requests.post("http://localhost:8000/api/v1/generate-test-suite", json={
    "api_spec_id": api_spec_id,
    "include_all_endpoints": True,
    "test_types": ["functional", "edge_case", "security"]
})

test_suite = response.json()["test_suite"]
print(f"Generated complete test suite with {test_suite['total_test_cases']} test cases")
```

#### Get Optimization Recommendations

```python
current_selection = [1, 2, 3, 4, 5]  # Test case IDs
recommendations_response = requests.get(
    f"http://localhost:8000/api/v1/recommendations/{api_spec_id}",
    params={"current_selection": current_selection}
)

recommendations = recommendations_response.json()
for rec in recommendations["recommendations"]:
    print(f"{rec['priority'].upper()}: {rec['message']}")
```

### 5. Integration Examples

#### CI/CD Pipeline Integration (GitHub Actions)

```yaml
name: AI-Powered API Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  api-testing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install requests
      
      - name: Run AI-Powered API Tests
        env:
          API_ENDPOINT: ${{ secrets.API_TESTING_ENDPOINT }}
          API_SPEC_ID: ${{ secrets.API_SPEC_ID }}
        run: |
          python -c "
          import requests
          import os
          
          # Run test suite
          response = requests.post(
              f'{os.environ[\"API_ENDPOINT\"]}/api/v1/run-tests',
              json={'api_spec_id': int(os.environ['API_SPEC_ID'])}
          )
          
          results = response.json()['results']
          print(f'Tests: {results[\"total_tests\"]}, Passed: {results[\"passed\"]}, Failed: {results[\"failed\"]}')
          
          # Fail the build if tests failed
          if results['failed'] > 0:
              exit(1)
          "
```

#### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        API_ENDPOINT = credentials('api-testing-endpoint')
        API_SPEC_ID = credentials('api-spec-id')
    }
    
    stages {
        stage('API Testing') {
            steps {
                script {
                    def response = sh(
                        script: """
                            curl -X POST "${API_ENDPOINT}/api/v1/run-tests" \
                                 -H "Content-Type: application/json" \
                                 -d '{"api_spec_id": ${API_SPEC_ID}}'
                        """,
                        returnStdout: true
                    ).trim()
                    
                    def results = readJSON text: response
                    
                    echo "Test Results: ${results.results.passed} passed, ${results.results.failed} failed"
                    
                    if (results.results.failed > 0) {
                        error("API tests failed!")
                    }
                }
            }
        }
    }
}
```

### 6. Dashboard Usage

Access the dashboard at `http://localhost:8050` to view:

- **System Overview**: Real-time statistics
- **Coverage Metrics**: Visual coverage analysis
- **Test Execution Trends**: Historical test performance
- **RL Performance**: Learning curve visualization

### 7. Monitoring and Alerting

#### Prometheus Metrics

The system exposes metrics at `/metrics` endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'api-testing'
    static_configs:
      - targets: ['localhost:8000']
```

#### Custom Monitoring Script

```python
import requests
import time

def monitor_system():
    while True:
        try:
            response = requests.get("http://localhost:8000/status")
            status = response.json()
            
            if status["system"]["status"] != "operational":
                print(f"ALERT: System status is {status['system']['status']}")
                
            # Check recent test results
            stats = status["statistics"]
            print(f"Stats: {stats['api_specifications']} APIs, {stats['test_cases']} tests")
            
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_system()
```

### 8. Best Practices

1. **API Specification Management**
   - Keep specifications up to date
   - Use version control for spec files
   - Document API changes

2. **Test Case Organization**
   - Use meaningful test names
   - Group tests by functionality
   - Maintain test coverage > 80%

3. **RL Optimization**
   - Train models regularly
   - Monitor learning curves
   - Adjust hyperparameters based on API characteristics

4. **Self-Healing**
   - Enable auto-revalidation
   - Review healed tests manually
   - Update documentation based on common failures

5. **Production Deployment**
   - Use environment-specific configurations
   - Implement proper logging and monitoring
   - Set up automated backups
   - Use secure API keys management
