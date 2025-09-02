"""
Pytest configuration and fixtures for the API testing framework.
"""

import asyncio
import os
import pytest
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.database.connection import get_db
from src.database.models import Base
from src.services.ai_test_generator import AITestGenerator
from src.services.rag_system import RAGSystem


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Create a test client with test database."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(db_session):
    """Create an async test client."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing AI features."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "test_cases": [
            {
                "name": "Test GET /users endpoint",
                "method": "GET",
                "path": "/users",
                "headers": {"Content-Type": "application/json"},
                "query_params": {},
                "body": null,
                "expected_status": 200,
                "assertions": [
                    {
                        "type": "status_code",
                        "expected": 200
                    }
                ]
            }
        ]
    }
    """
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing."""
    mock_rag = MagicMock(spec=RAGSystem)
    mock_rag.query_similar_specs.return_value = [
        {"content": "Sample API spec", "score": 0.9}
    ]
    return mock_rag


@pytest.fixture
def sample_openapi_spec():
    """Sample OpenAPI specification for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "description": "A test API for our testing framework"
        },
        "servers": [
            {"url": "https://api.example.com/v1"}
        ],
        "paths": {
            "/users": {
                "get": {
                    "summary": "Get all users",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "name": {"type": "string"},
                                                "email": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/users/{user_id}": {
                "get": {
                    "summary": "Get user by ID",
                    "parameters": [
                        {
                            "name": "user_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "User found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "name": {"type": "string"},
                                            "email": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "User not found"
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_test_execution_result():
    """Sample test execution result for testing."""
    return {
        "test_case_id": "test_123",
        "status": "passed",
        "response_time": 250.5,
        "status_code": 200,
        "response_body": {"users": [{"id": 1, "name": "John Doe"}]},
        "assertions": [
            {
                "type": "status_code",
                "expected": 200,
                "actual": 200,
                "passed": True
            }
        ],
        "timestamp": "2024-01-01T12:00:00Z"
    }


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.update({
        "ENVIRONMENT": "test",
        "DATABASE_URL": "sqlite:///./test.db",
        "REDIS_URL": "redis://localhost:6379/1",
        "OPENAI_API_KEY": "test-key",
        "SECRET_KEY": "test-secret-key",
        "LOG_LEVEL": "DEBUG"
    })
    yield
    # Cleanup after tests
    for key in ["ENVIRONMENT", "DATABASE_URL", "REDIS_URL", "OPENAI_API_KEY", "SECRET_KEY", "LOG_LEVEL"]:
        os.environ.pop(key, None)
