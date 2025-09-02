# AI-Powered API Testing Framework

A comprehensive, intelligent API testing framework that leverages artificial intelligence for automated test generation, execution, and optimization.

## 🚀 Features

### Core Capabilities
- **AI-Powered Test Generation**: Automatically generate comprehensive test cases from OpenAPI specifications using GPT models
- **RAG-Enhanced Testing**: Retrieval-Augmented Generation for context-aware test creation based on similar API patterns
- **Intelligent Test Execution**: Concurrent, scalable test execution with advanced assertion frameworks
- **Coverage Analysis**: Deep API coverage analysis with gap identification and improvement recommendations
- **Self-Healing Tests**: Automatic failure pattern analysis and intelligent test repair suggestions
- **Reinforcement Learning Optimization**: RL-based test selection and execution optimization

### Technology Stack
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, Redis, Celery
- **AI/ML**: OpenAI GPT, ChromaDB (RAG), Custom RL algorithms
- **Frontend**: React 18, TypeScript, React Query, Recharts
- **Infrastructure**: Docker, docker-compose, Nginx
- **Testing**: Pytest, comprehensive test suite

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL (or use Docker)
- Redis (or use Docker)

### Environment Setup

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd api_testing
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Database Setup**
   ```bash
   # Using Docker
   docker-compose up -d postgres redis
   
   # Initialize database
   python -c "from src.database.connection import create_tables; create_tables()"
   ```

4. **Start Services**
   ```bash
   # Backend API
   uvicorn src.api.main:app --reload --port 8000
   
   # Celery Worker (new terminal)
   celery -A src.tasks worker --loglevel=info
   
   # Frontend (new terminal)
   cd frontend
   npm install
   npm start
   ```

### Docker Deployment

```bash
# Full stack deployment
docker-compose up -d

# Individual services
docker-compose up -d api postgres redis
docker-compose up -d frontend nginx
```

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend│    │   AI Services   │
│   - Dashboard    │◄──►│   - REST API     │◄──►│   - OpenAI GPT  │
│   - Test UI      │    │   - WebSocket    │    │   - RAG System  │
│   - Analytics    │    │   - Auth         │    │   - RL Optimizer│
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                        │                        │
          │                        │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Infrastructure │    │   Data Layer    │    │   ML Pipeline   │
│   - Docker       │    │   - PostgreSQL  │    │   - Training    │
│   - Nginx        │    │   - Redis       │    │   - Model Store │
│   - Monitoring   │    │   - ChromaDB    │    │   - Optimization│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. API Specification Management
- **Ingestion**: Upload/fetch OpenAPI specs
- **Validation**: Schema validation and parsing
- **Storage**: Versioned spec management

#### 2. AI Test Generation
- **GPT Integration**: Structured prompts for test case generation
- **RAG Enhancement**: Context from similar APIs
- **Edge Case Detection**: Boundary value and error condition testing

#### 3. Test Execution Engine
- **Concurrent Execution**: Configurable parallelism
- **Assertion Framework**: Flexible assertion types
- **Result Tracking**: Comprehensive execution metrics

#### 4. Coverage Analysis
- **Endpoint Coverage**: Track tested vs. available endpoints
- **Method Coverage**: HTTP method coverage analysis
- **Parameter Coverage**: Request parameter testing coverage
- **Gap Analysis**: Identify untested areas

#### 5. Self-Healing System
- **Pattern Recognition**: Identify common failure patterns
- **AI-Powered Suggestions**: Generate intelligent fix recommendations
- **Auto-Application**: Optional automatic healing application

#### 6. RL Optimization
- **Test Selection**: Optimize test case selection for maximum coverage
- **Resource Allocation**: Intelligent resource distribution
- **Continuous Learning**: Improve optimization over time

## 📊 Usage Guide

### 1. Upload API Specification

**Via Web UI:**
1. Navigate to "API Specs" page
2. Click "Upload New Spec"
3. Provide spec file or URL
4. Configure base URL and authentication

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/specs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API",
    "description": "API description",
    "spec_content": {...openapi_spec...}
  }'
```

### 2. Generate Test Cases

**Auto-Generation:**
```bash
curl -X POST "http://localhost:8000/api/specs/{spec_id}/generate-tests" \
  -H "Content-Type: application/json" \
  -d '{"test_types": ["happy_path", "edge_cases", "error_scenarios"]}'
```

**Custom Test Creation:**
Use the web UI to manually create and customize test cases.

### 3. Execute Tests

**Single Test:**
```bash
curl -X POST "http://localhost:8000/api/test-cases/{test_id}/execute"
```

**Batch Execution:**
```bash
curl -X POST "http://localhost:8000/api/specs/{spec_id}/execute-all"
```

### 4. Analyze Results

- **Coverage Reports**: View endpoint and method coverage
- **Performance Metrics**: Response times, throughput analysis
- **Failure Analysis**: Identify patterns and trends
- **AI Insights**: Get recommendations for improvement

### 5. Self-Healing

When tests fail, the system automatically:
1. Analyzes failure patterns
2. Generates healing suggestions
3. Optionally applies fixes
4. Re-runs tests to verify healing

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=development
SECRET_KEY=your-secret-key
DEBUG=true

# Database
DATABASE_URL=postgresql://user:password@localhost/api_testing
REDIS_URL=redis://localhost:6379/0

# AI Services
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4
ENABLE_RAG=true
CHROMADB_PERSIST_DIR=./data/chromadb

# Execution Settings
MAX_CONCURRENT_TESTS=20
DEFAULT_TIMEOUT=30
RETRY_ATTEMPTS=3

# RL Configuration
RL_LEARNING_RATE=0.01
RL_EXPLORATION_RATE=0.1
RL_DISCOUNT_FACTOR=0.95

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_TELEMETRY=true
```

### Advanced Configuration

**Custom AI Prompts:**
Edit `src/config/prompts.py` to customize AI generation prompts.

**RL Parameters:**
Modify `src/config/rl_config.py` for reinforcement learning tuning.

**Coverage Thresholds:**
Set coverage targets in `src/config/coverage.py`.

## 🧪 Testing

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific test categories
pytest tests/test_api.py -v              # API tests
pytest tests/test_services.py -v         # Service tests
pytest tests/test_models.py -v           # Database tests
pytest tests/test_performance.py -v      # Performance tests

# With coverage
pytest tests/ --cov=src --cov-report=html

# Load tests (slow)
pytest tests/ -m slow
```

### Test Configuration

**Test Database:**
Tests use SQLite in-memory database for isolation.

**Mocking:**
- OpenAI API calls are mocked
- HTTP requests are mocked
- External services are mocked

**Fixtures:**
Comprehensive fixtures for all major components.

## 📈 Monitoring & Observability

### Metrics Collection
- **Test Execution Metrics**: Success rates, response times, throughput
- **Coverage Metrics**: Endpoint coverage, parameter coverage, method coverage
- **AI Metrics**: Generation success rates, suggestion accuracy
- **System Metrics**: Resource usage, performance indicators

### Dashboard Features
- **Real-time Updates**: Live test execution status
- **Interactive Charts**: Coverage trends, performance graphs
- **Alerts**: Failure notifications, coverage threshold alerts
- **Reports**: Exportable coverage and performance reports

### Logging
Structured JSON logging with:
- Request/response tracking
- Error details and stack traces
- Performance metrics
- AI decision logging

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control
- API key management
- Secure secret storage

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Rate limiting

### Infrastructure Security
- Non-root Docker containers
- Network segmentation
- Secure defaults
- Regular security updates

## 🚢 Deployment

### Production Deployment

**Docker (Recommended):**
```bash
# Production build
docker-compose -f docker-compose.prod.yml up -d

# With SSL/TLS
docker-compose -f docker-compose.prod.yml -f docker-compose.ssl.yml up -d
```

**Manual Deployment:**
```bash
# Backend
gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Frontend
npm run build
# Serve build/ directory with nginx or similar

# Workers
celery -A src.tasks worker --concurrency=4
celery -A src.tasks beat
```

### Scaling Considerations

**Horizontal Scaling:**
- Load balance multiple API instances
- Scale Celery workers based on queue depth
- Use Redis Cluster for high availability

**Database Optimization:**
- Connection pooling
- Read replicas for analytics
- Proper indexing strategy

**Caching Strategy:**
- Redis for session storage
- API response caching
- Static asset caching

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Set up development environment
4. Make changes with tests
5. Submit pull request

### Code Standards
- **Python**: PEP 8, type hints, docstrings
- **TypeScript**: ESLint, Prettier, strict mode
- **Testing**: Minimum 90% coverage
- **Documentation**: Update docs with changes

### Pull Request Process
1. Ensure all tests pass
2. Update documentation
3. Add changelog entry
4. Request review

## 📚 API Documentation

### REST API
- **Interactive Docs**: `http://localhost:8000/docs`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`
- **ReDoc**: `http://localhost:8000/redoc`

### WebSocket API
- **Connection**: `ws://localhost:8000/ws`
- **Real-time Updates**: Test execution status, coverage updates
- **Events**: Test started, completed, failed, healed

## 🔍 Troubleshooting

### Common Issues

**Database Connection Errors:**
- Verify PostgreSQL is running
- Check DATABASE_URL configuration
- Ensure database exists and is accessible

**AI Service Errors:**
- Verify OPENAI_API_KEY is set
- Check API quota and limits
- Ensure internet connectivity

**Test Execution Failures:**
- Check target API availability
- Verify network connectivity
- Review timeout settings

**Performance Issues:**
- Monitor resource usage
- Adjust concurrency limits
- Check database indexes

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### Health Checks
- **API Health**: `GET /health`
- **Database**: `GET /health/db`
- **Redis**: `GET /health/redis`
- **AI Services**: `GET /health/ai`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT integration
- FastAPI for the excellent web framework
- ChromaDB for vector database capabilities
- React community for frontend ecosystem

---

For more detailed documentation, see the `docs/` directory:
- [API Reference](./api-reference.md)
- [Deployment Guide](./deployment.md)
- [Contributing Guide](./contributing.md)
- [Architecture Deep Dive](./architecture.md)
