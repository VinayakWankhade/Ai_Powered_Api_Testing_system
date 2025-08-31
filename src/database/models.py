"""
Database models for the AI-powered API testing framework.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, 
    Integer, JSON, String, Text, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class TestStatus(PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"

class SpecType(PyEnum):
    OPENAPI = "openapi"
    SWAGGER = "swagger"
    RAW_LOGS = "raw_logs"
    CUSTOM = "custom"

class TestType(PyEnum):
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EDGE_CASE = "edge_case"
    GENERATED = "generated"

class RLAlgorithm(PyEnum):
    Q_LEARNING = "q_learning"
    PPO = "ppo"
    EVOLUTIONARY = "evolutionary"

class APISpecification(Base):
    """Store API specifications and their metadata."""
    __tablename__ = "api_specifications"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    spec_type = Column(Enum(SpecType), nullable=False)
    base_url = Column(String(500), nullable=True)
    
    # Store the raw specification content
    raw_content = Column(JSON, nullable=False)
    parsed_endpoints = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    test_cases = relationship("TestCase", back_populates="api_spec", cascade="all, delete-orphan")
    execution_sessions = relationship("ExecutionSession", back_populates="api_spec")
    
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_spec_name_version'),
        Index('idx_spec_type_active', 'spec_type', 'is_active'),
    )

class TestCase(Base):
    """Generated and executed test cases."""
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)
    api_spec_id = Column(Integer, ForeignKey("api_specifications.id"), nullable=False)
    
    # Test identification
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    test_type = Column(Enum(TestType), nullable=False)
    endpoint = Column(String(500), nullable=False)
    method = Column(String(10), nullable=False)
    
    # Test definition
    test_data = Column(JSON, nullable=False)  # Request data, headers, params
    expected_response = Column(JSON, nullable=True)
    assertions = Column(JSON, nullable=True)
    
    # Generation metadata
    generated_by_llm = Column(Boolean, default=True)
    generation_prompt = Column(Text, nullable=True)
    generation_context = Column(JSON, nullable=True)
    
    # RL optimization
    rl_score = Column(Float, default=0.0)
    selection_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    api_spec = relationship("APISpecification", back_populates="test_cases")
    executions = relationship("TestExecution", back_populates="test_case")
    
    __table_args__ = (
        Index('idx_endpoint_method', 'endpoint', 'method'),
        Index('idx_test_type_active', 'test_type', 'is_active'),
        Index('idx_rl_score', 'rl_score'),
    )

class ExecutionSession(Base):
    """Test execution sessions."""
    __tablename__ = "execution_sessions"

    id = Column(Integer, primary_key=True, index=True)
    api_spec_id = Column(Integer, ForeignKey("api_specifications.id"), nullable=False)
    
    # Session metadata
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    trigger = Column(String(50), nullable=True)  # manual, scheduled, ci_cd, rl_optimization
    
    # Execution statistics
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    error_tests = Column(Integer, default=0)
    skipped_tests = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Coverage
    endpoint_coverage = Column(JSON, nullable=True)
    method_coverage = Column(JSON, nullable=True)
    response_code_coverage = Column(JSON, nullable=True)
    
    # RL insights
    rl_algorithm_used = Column(Enum(RLAlgorithm), nullable=True)
    rl_optimization_score = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    api_spec = relationship("APISpecification", back_populates="execution_sessions")
    test_executions = relationship("TestExecution", back_populates="session")

class TestExecution(Base):
    """Individual test execution results."""
    __tablename__ = "test_executions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("execution_sessions.id"), nullable=False)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=False)
    
    # Execution results
    status = Column(Enum(TestStatus), nullable=False)
    response_time_ms = Column(Float, nullable=True)
    response_code = Column(Integer, nullable=True)
    response_body = Column(JSON, nullable=True)
    response_headers = Column(JSON, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    
    # Assertion results
    assertion_results = Column(JSON, nullable=True)
    coverage_contribution = Column(JSON, nullable=True)
    
    # Self-healing
    required_healing = Column(Boolean, default=False)
    healing_attempts = Column(Integer, default=0)
    healed_successfully = Column(Boolean, default=False)
    healing_log = Column(JSON, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    session = relationship("ExecutionSession", back_populates="test_executions")
    test_case = relationship("TestCase", back_populates="executions")
    
    __table_args__ = (
        Index('idx_status_session', 'status', 'session_id'),
        Index('idx_response_time', 'response_time_ms'),
        Index('idx_healing_required', 'required_healing'),
    )

class DocumentationStore(Base):
    """Store API documentation and related content for RAG."""
    __tablename__ = "documentation_store"

    id = Column(Integer, primary_key=True, index=True)
    api_spec_id = Column(Integer, ForeignKey("api_specifications.id"), nullable=True)
    
    # Document metadata
    title = Column(String(255), nullable=False)
    doc_type = Column(String(50), nullable=False)  # endpoint_doc, example, error_guide, etc.
    source = Column(String(255), nullable=True)
    
    # Content
    content = Column(Text, nullable=False)
    structured_content = Column(JSON, nullable=True)
    
    # Vector embeddings (stored as JSON for simplicity)
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    
    # Usage tracking
    retrieval_count = Column(Integer, default=0)
    last_retrieved_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_doc_type', 'doc_type'),
        Index('idx_spec_doc_type', 'api_spec_id', 'doc_type'),
    )

class RLModel(Base):
    """Store RL model states and performance metrics."""
    __tablename__ = "rl_models"

    id = Column(Integer, primary_key=True, index=True)
    api_spec_id = Column(Integer, ForeignKey("api_specifications.id"), nullable=False)
    
    # Model metadata
    algorithm = Column(Enum(RLAlgorithm), nullable=False)
    model_version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Model state
    model_state = Column(JSON, nullable=True)  # Serialized model parameters
    hyperparameters = Column(JSON, nullable=True)
    
    # Performance metrics
    episodes_trained = Column(Integer, default=0)
    average_reward = Column(Float, default=0.0)
    best_reward = Column(Float, default=0.0)
    convergence_score = Column(Float, default=0.0)
    
    # Training history
    training_history = Column(JSON, nullable=True)
    
    # Model status
    is_active = Column(Boolean, default=True)
    is_trained = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('api_spec_id', 'algorithm', 'model_version', name='uq_rl_model'),
        Index('idx_algorithm_active', 'algorithm', 'is_active'),
    )

class CoverageMetrics(Base):
    """Track coverage metrics over time."""
    __tablename__ = "coverage_metrics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("execution_sessions.id"), nullable=False)
    
    # Coverage percentages
    endpoint_coverage_pct = Column(Float, default=0.0)
    method_coverage_pct = Column(Float, default=0.0)
    response_code_coverage_pct = Column(Float, default=0.0)
    parameter_coverage_pct = Column(Float, default=0.0)
    
    # Detailed coverage data
    covered_endpoints = Column(JSON, nullable=True)
    missed_endpoints = Column(JSON, nullable=True)
    edge_cases_covered = Column(JSON, nullable=True)
    
    # Bug detection
    bugs_found = Column(Integer, default=0)
    new_bugs_found = Column(Integer, default=0)
    bug_severity_distribution = Column(JSON, nullable=True)
    
    # Improvement metrics
    improvement_over_previous = Column(Float, default=0.0)
    quality_score = Column(Float, default=0.0)
    
    # Metadata
    measured_at = Column(DateTime, server_default=func.now())

class AIGenerationLog(Base):
    """Log AI generation requests and responses for debugging and optimization."""
    __tablename__ = "ai_generation_logs"

    id = Column(Integer, primary_key=True, index=True)
    test_case_id = Column(Integer, ForeignKey("test_cases.id"), nullable=True)
    
    # Generation request
    prompt_template = Column(Text, nullable=False)
    prompt_variables = Column(JSON, nullable=True)
    final_prompt = Column(Text, nullable=False)
    
    # AI response
    ai_model = Column(String(100), nullable=False)
    raw_response = Column(Text, nullable=False)
    parsed_response = Column(JSON, nullable=True)
    
    # Generation metadata
    generation_time_ms = Column(Float, nullable=True)
    token_usage = Column(JSON, nullable=True)
    temperature = Column(Float, nullable=True)
    
    # Quality metrics
    validation_passed = Column(Boolean, nullable=True)
    validation_errors = Column(JSON, nullable=True)
    human_rating = Column(Float, nullable=True)  # 1-5 scale
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
