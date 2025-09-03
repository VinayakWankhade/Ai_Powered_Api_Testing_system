# Architecture and End-to-End Workflow

This document provides system architecture and end‑to‑end workflow diagrams for the AI‑Powered API Testing Framework.

## 1) System Architecture (component view)

```mermaid
graph TD
  UI[Frontend (React)] --> API[FastAPI Service]

  subgraph API Layer
    API --> R1[Specs Router]
    API --> R2[Test Generation Router]
    API --> R3[Test Execution Router]
    API --> R4[Healing Router]
    API --> R5[Coverage/Analytics Router]
    API --> R6[RL Optimization Router]
  end

  subgraph Services
    R2 --> Gen[AITestGenerator]
    R3 --> Exec[HybridTestExecutor]
    R5 --> Cov[CoverageTracker]
    R4 --> Heal[SelfHealingSystem]
    R6 --> RL[HybridRLOptimizer]
  end

  Gen --> LLM[LLMGateway]
  Heal --> LLM
  LLM --> Mon[CostMonitor]

  subgraph Providers
    LLM --> P1[LocalOllamaProvider]
    LLM --> P2[OpenAIProvider (stub)]
    LLM --> P3[AnthropicProvider (stub)]
  end

  Services --> DB[(SQLAlchemy DB: SQLite/Postgres)]
  RAG[(ChromaDB - optional)] -.-> Gen
```

Key flows:
- API routers orchestrate domain services (generation, execution, coverage, healing, RL).
- AITestGenerator and SelfHealingSystem route prompts via LLMGateway, which applies cost control, caching, rate limiting, and provider selection (prefers free LOCAL/Ollama when configured).
- Results are persisted via SQLAlchemy models; analytics and dashboards read aggregated data.

## 2) LLM Gateway internals (simplified)

```mermaid
flowchart LR
  A[Request] --> B(ModelSelector)
  B --> C{RateLimiter}
  C -->|ok| D(ResponseCache)
  D -->|hit| Z[Return cached]
  D -->|miss| E(Provider Registry)
  E --> F1[LocalOllamaProvider]
  E --> F2[OpenAI/Anthropic]
  F1 --> G[Response]
  F2 --> G
  G --> H(CostTracker)
  H --> I(Cache Store)
  I --> Z
```

Highlights:
- ModelSelector chooses a model subject to user constraints and budgets.
- RateLimiter enforces provider throughput; ResponseCache avoids repeat spend.
- Provider Registry abstracts concrete providers; LocalOllamaProvider enables zero‑cost local inference.
- CostTracker + CostMonitor provide budget enforcement and predictive alerts.

## 3) End‑to‑End Workflow (sequence)

```mermaid
sequenceDiagram
  autonumber
  participant Client
  participant API as FastAPI
  participant Gen as AITestGenerator
  participant LLM as LLMGateway
  participant Prov as Local/Cloud Provider
  participant Exec as HybridTestExecutor
  participant DB as Database
  participant Cov as CoverageTracker
  participant Heal as SelfHealingSystem
  participant RL as HybridRLOptimizer

  Client->>API: Upload Spec (POST /api/v1/upload-spec)
  API->>DB: Store APISpecification

  Client->>API: Generate Tests (POST /api/v1/generate-tests)
  API->>Gen: generate_test_cases(...)
  Gen->>LLM: select model + prompt
  LLM->>Prov: call provider (prefer LOCAL)
  Prov-->>LLM: response (tests)
  LLM->>DB: record RequestMetrics
  LLM-->>Gen: test cases
  Gen-->>API: test cases

  Client->>API: Run Tests (POST /api/v1/run-tests)
  API->>Exec: execute_test_suite()
  Exec->>DB: write ExecutionSession + TestExecution
  API-->>Client: run summary

  Client->>API: Coverage Report (GET /api/v1/coverage-report/{id})
  API->>Cov: get_coverage_report(id)
  Cov->>DB: read executions/spec
  Cov-->>API: coverage metrics
  API-->>Client: report

  Client->>API: Heal Tests (POST /api/v1/heal-tests)
  API->>Heal: heal_tests(failed)
  Heal->>LLM: propose fixes
  LLM->>Prov: provider call
  Prov-->>LLM: fix suggestions
  Heal->>DB: update test cases
  API-->>Client: healing summary

  Client->>API: Optimize (POST /api/v1/optimize-tests)
  API->>RL: optimize_test_selection()
  RL->>DB: read history/coverage
  RL-->>API: optimized selection
  API-->>Client: recommendations
```

## 4) Data model (core entities)
- APISpecification: OpenAPI source, parsed_endpoints, base_url, metadata.
- TestCase: endpoint/method, test data, assertions, expected responses, generated_by_llm.
- ExecutionSession and TestExecution: run-level and per-test results, timings, statuses.
- CoverageReport/Metrics: aggregates for dashboards and gap analysis.
- RLModel: algorithm/version, episodes, performance.
- HealingSuggestion: suggestion type, confidence, priority, application results.

## 5) Operational concerns
- Security: API key auth, secure CORS, rate limiting, security headers, input sanitization.
- Cost: Local provider (Ollama) preferred; budgets/alerts via CostMonitor.
- Performance: Async concurrency, LRU cache, batching (future optimization hooks).
- Observability: Health/status endpoints, structured logging, probe scripts.

---

For hands‑on examples, see:
- docs/usage_examples.md
- scripts/test_local_provider.py (local LLM smoke test)
- scripts/probe_endpoints.py (API probe)

