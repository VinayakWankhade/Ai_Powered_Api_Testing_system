# AI-Powered API Testing System

An advanced, agentic framework for automated API testing that integrates Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and hybrid Reinforcement Learning (RL) for intelligent test generation, execution, and optimization.

## 🚀 Features

### Core Capabilities
- **🤖 AI-Powered Test Generation**: Uses LLMs with RAG to generate context-aware test cases
- **📋 API Specification Ingestion**: Supports OpenAPI/Swagger specs and raw API logs
- **🏃‍♂️ Sandboxed Test Execution**: Concurrent test execution with comprehensive logging
- **🧠 Hybrid RL Optimization**: Combines Q-learning, PPO, and evolutionary algorithms
- **🔧 Self-Healing Mechanism**: Automatic failure analysis and test repair
- **📊 Coverage Tracking**: Real-time coverage metrics and visualization
- **🌐 RESTful API Service**: Complete API for CI/CD integration
- **📈 Interactive Dashboard**: Visual insights into test evolution and performance

### Advanced Features
- **Retrieval-Augmented Generation**: Context-aware test generation using vector databases
- **Multi-Algorithm RL**: Q-learning, PPO, and evolutionary optimization
- **Test Selection Optimization**: Smart test case selection for maximum coverage
- **Real-time Analytics**: Performance metrics and learning curves visualization
- **CI/CD Integration**: Ready-to-use deployment configurations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   AI Engine     │    │   RL Optimizer  │
│   Service       │────│   (LLM + RAG)   │────│   (Hybrid)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └─────────────────────────────────────────────────────┐
                                 │                             │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   Database      │    │   Execution     │    │   Self-Healing  │
         │   (SQLAlchemy)  │────│   Engine        │────│   Mechanism     │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
                                         │
                                ┌─────────────────┐
                                │   Dashboard     │
                                │   & Reporting   │
                                └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- PostgreSQL (optional, SQLite by default)
- Redis (for Celery task queue)
- OpenAI API key

### Quick Start

```bash
# Clone the repository
git clone https://github.com/VinayakWankhade/Ai_Powered_Api_Testing_system.git
cd Ai_Powered_Api_Testing_system

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize the database
python -c "from src.database.connection import create_tables; create_tables()"

# Start the API service
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Configuration

Create a `.env` file with the following configuration:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Database Configuration
DATABASE_URL=sqlite:///./data/api_testing.db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/api_testing.log

# RAG Configuration
CHROMADB_PATH=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RL Configuration
RL_MODEL_PATH=./data/rl_models
TENSORBOARD_LOG_DIR=./logs/tensorboard

# Testing Configuration
TEST_TIMEOUT=300
MAX_CONCURRENT_TESTS=5
SANDBOX_MODE=true
```

## 🔧 Usage

### 1. Upload API Specification

```bash
# Upload OpenAPI specification
curl -X POST "http://localhost:8000/api/v1/upload-spec" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API",
    "version": "1.0.0",
    "spec_type": "openapi",
    "spec_content": "your_openapi_spec_here"
  }'
```

### 2. Generate Test Cases

```bash
# Generate AI-powered test cases
curl -X POST "http://localhost:8000/api/v1/generate-tests" \
  -H "Content-Type: application/json" \
  -d '{
    "api_spec_id": 1,
    "endpoint_path": "/users",
    "method": "GET",
    "test_types": ["functional", "edge_case"],
    "count": 5
  }'
```

### 3. Execute Tests

```bash
# Run test execution session
curl -X POST "http://localhost:8000/api/v1/run-tests" \
  -H "Content-Type: application/json" \
  -d '{
    "api_spec_id": 1,
    "session_name": "Regression Test Suite"
  }'
```

### 4. Optimize with RL

```bash
# Train RL model for test optimization
curl -X POST "http://localhost:8000/api/v1/optimize-tests" \
  -H "Content-Type: application/json" \
  -d '{
    "api_spec_id": 1,
    "algorithm": "ppo",
    "training_episodes": 1000
  }'
```

### 5. View Coverage Report

```bash
# Get coverage metrics
curl -X GET "http://localhost:8000/api/v1/coverage-report/1"
```

## 🎯 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/upload-spec` | POST | Upload API specification |
| `/api/v1/generate-tests` | POST | Generate AI test cases |
| `/api/v1/run-tests` | POST | Execute test session |
| `/api/v1/heal-tests` | POST | Self-heal failed tests |
| `/api/v1/coverage-report/{api_spec_id}` | GET | Get coverage metrics |
| `/api/v1/optimize-tests` | POST | Run RL optimization |

### Advanced Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/specs` | GET | List API specifications |
| `/api/v1/test-cases/{api_spec_id}` | GET | Get test cases |
| `/api/v1/execution-history` | GET | View execution history |
| `/api/v1/rl-performance/{api_spec_id}` | GET | RL model performance |
| `/api/v1/recommendations/{api_spec_id}` | GET | Get optimization recommendations |

## 🧪 Testing Framework Components

### 1. AI Test Generator
- **LLM Integration**: Uses OpenAI GPT models for intelligent test case generation
- **RAG System**: Retrieves relevant documentation and examples for context
- **Test Types**: Supports functional, edge case, security, and performance tests
- **Contextual Generation**: Considers API specification, historical data, and patterns

### 2. Test Execution Engine
- **Sandboxed Execution**: Safe, isolated test execution environment
- **Concurrent Processing**: Parallel test execution with controlled concurrency
- **Comprehensive Logging**: Detailed execution logs and metrics
- **Coverage Tracking**: Real-time endpoint and parameter coverage analysis

### 3. Hybrid RL Optimizer
- **Multi-Algorithm Approach**: Q-learning, PPO, and evolutionary algorithms
- **Test Selection**: Optimizes test case selection for maximum coverage
- **Performance Learning**: Continuously learns from execution results
- **Adaptive Strategies**: Adjusts testing strategies based on API characteristics

### 4. Self-Healing Mechanism
- **Failure Analysis**: Automatic analysis of test failures
- **Documentation Retrieval**: RAG-powered error context retrieval
- **Test Repair**: LLM-based test case repair and regeneration
- **Validation**: Re-execution and validation of healed tests

## 📊 Dashboard Features

- **Real-time Metrics**: Live test execution and coverage metrics
- **RL Learning Curves**: Visualization of model training progress
- **Coverage Evolution**: Historical coverage growth tracking
- **Bug Discovery Trends**: Analysis of defect detection patterns
- **Performance Analytics**: Test execution performance metrics

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Scale services
docker-compose up -d --scale api=3 --scale worker=2
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: API Testing with AI Framework
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run AI-Powered API Tests
        run: |
          curl -X POST "${{ secrets.API_TESTING_ENDPOINT }}/api/v1/run-tests" \
            -H "Authorization: Bearer ${{ secrets.API_TOKEN }}" \
            -d '{"api_spec_id": 1, "trigger": "ci_cd"}'
```

## 📈 Performance & Scalability

- **Concurrent Execution**: Configurable parallel test execution
- **Database Optimization**: Indexed queries and connection pooling
- **Caching Layer**: Redis-based caching for improved performance
- **Horizontal Scaling**: Microservice architecture for easy scaling
- **Resource Management**: Efficient memory and CPU utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the powerful LLM capabilities
- Stable Baselines3 for RL algorithm implementations
- FastAPI for the excellent web framework
- ChromaDB for vector database capabilities
- The open-source community for various dependencies

## 📞 Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/VinayakWankhade/Ai_Powered_Api_Testing_system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VinayakWankhade/Ai_Powered_Api_Testing_system/discussions)

---

**Built with ❤️ by Vinayak Wankhade**

*Transforming API testing with the power of AI and machine learning*
