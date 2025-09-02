"""
Cost-Bounded LLM Gateway for managing AI model usage with intelligent cost control,
rate limiting, and optimization for automated API testing workflows.
"""

import asyncio
import json
import time
import httpx
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import statistics

from ..database.connection import get_db_session
from ..database.models import TestExecution, APISpecification
from ..utils.logger import get_logger

# Providers
from .providers.base import BaseProvider
from .providers.local_ollama import LocalOllamaProvider

logger = get_logger(__name__)

class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"

class ModelTier(Enum):
    """Model capability tiers for cost optimization."""
    BASIC = "basic"          # Simple tasks, lowest cost
    STANDARD = "standard"    # Balanced capability/cost
    ADVANCED = "advanced"    # Complex tasks, higher cost
    PREMIUM = "premium"      # Most capable, highest cost

class RequestPriority(Enum):
    """Priority levels for request handling."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class CostLimitType(Enum):
    """Types of cost limits."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    tier: ModelTier
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_tokens: int
    context_window: int
    supports_streaming: bool = False
    supports_function_calling: bool = False
    latency_ms_avg: int = 1000

@dataclass
class CostLimit:
    """Cost limit configuration."""
    limit_type: CostLimitType
    amount_usd: float
    current_usage: float = 0.0
    reset_time: Optional[datetime] = None

@dataclass
class RequestMetrics:
    """Metrics for a single LLM request."""
    request_id: str
    provider: ModelProvider
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False

@dataclass
class LLMRequest:
    """LLM request with metadata."""
    request_id: str
    prompt: str
    model_preference: Optional[str] = None
    tier_preference: Optional[ModelTier] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    priority: RequestPriority = RequestPriority.NORMAL
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    cache_enabled: bool = True

@dataclass
class LLMResponse:
    """LLM response with metadata."""
    request_id: str
    content: str
    model_used: str
    provider_used: ModelProvider
    tokens_used: int
    cost_usd: float
    latency_ms: int
    cached: bool = False
    error: Optional[str] = None

class RateLimiter:
    """Sliding window rate limiter for LLM requests."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_seconds:
                self.requests.popleft()
            
            # Check if we can make another request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    async def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        async with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            # Time until oldest request ages out
            oldest_request = self.requests[0]
            return max(0.0, (oldest_request + self.window_seconds) - time.time())

class ResponseCache:
    """LRU cache for LLM responses to reduce costs."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order = deque()
        self._lock = asyncio.Lock()
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate cache key for request."""
        content = f"{prompt}|{model}|{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, prompt: str, model: str, temperature: float) -> Optional[LLMResponse]:
        """Get cached response if available."""
        async with self._lock:
            cache_key = self._generate_cache_key(prompt, model, temperature)
            
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                
                # Check TTL
                if time.time() - cached_data["timestamp"] < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.access_order.remove(cache_key)
                    self.access_order.append(cache_key)
                    
                    # Return cached response
                    response_data = cached_data["response"]
                    response_data.cached = True
                    return response_data
                else:
                    # Expired, remove from cache
                    del self.cache[cache_key]
                    self.access_order.remove(cache_key)
            
            return None
    
    async def put(self, prompt: str, model: str, temperature: float, response: LLMResponse):
        """Cache response."""
        async with self._lock:
            cache_key = self._generate_cache_key(prompt, model, temperature)
            
            # Make room if needed
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            # Store response
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            self.access_order.append(cache_key)
    
    async def clear_expired(self):
        """Remove expired entries from cache."""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for cache_key, data in self.cache.items():
                if current_time - data["timestamp"] >= self.ttl_seconds:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")

class CostTracker:
    """Tracks and manages LLM usage costs."""
    
    def __init__(self):
        self.usage_history: List[RequestMetrics] = []
        self.cost_limits: Dict[CostLimitType, CostLimit] = {}
        self.provider_costs: Dict[ModelProvider, float] = defaultdict(float)
        self._lock = asyncio.Lock()
    
    async def add_usage(self, metrics: RequestMetrics):
        """Record usage metrics."""
        async with self._lock:
            self.usage_history.append(metrics)
            self.provider_costs[metrics.provider] += metrics.cost_usd
            
            # Update cost limit usage
            await self._update_cost_limits(metrics.cost_usd)
    
    async def _update_cost_limits(self, cost_usd: float):
        """Update usage for all applicable cost limits."""
        now = datetime.now()
        
        for limit_type, cost_limit in self.cost_limits.items():
            # Reset if needed
            if cost_limit.reset_time and now >= cost_limit.reset_time:
                cost_limit.current_usage = 0.0
                cost_limit.reset_time = self._calculate_next_reset(limit_type, now)
            
            cost_limit.current_usage += cost_usd
    
    def _calculate_next_reset(self, limit_type: CostLimitType, current_time: datetime) -> datetime:
        """Calculate next reset time for cost limit."""
        if limit_type == CostLimitType.HOURLY:
            return current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif limit_type == CostLimitType.DAILY:
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif limit_type == CostLimitType.WEEKLY:
            days_ahead = 6 - current_time.weekday()
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead + 1)
        elif limit_type == CostLimitType.MONTHLY:
            if current_time.month == 12:
                return current_time.replace(year=current_time.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                return current_time.replace(month=current_time.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # TOTAL - never resets
            return datetime.max
    
    async def set_cost_limit(self, limit_type: CostLimitType, amount_usd: float):
        """Set cost limit for specified period."""
        async with self._lock:
            now = datetime.now()
            reset_time = self._calculate_next_reset(limit_type, now) if limit_type != CostLimitType.TOTAL else None
            
            self.cost_limits[limit_type] = CostLimit(
                limit_type=limit_type,
                amount_usd=amount_usd,
                current_usage=0.0,
                reset_time=reset_time
            )
            
            logger.info(f"Set {limit_type.value} cost limit: ${amount_usd:.4f}")
    
    async def check_cost_limits(self, projected_cost: float) -> Dict[str, Any]:
        """Check if request would exceed cost limits."""
        async with self._lock:
            violations = []
            can_proceed = True
            
            for limit_type, cost_limit in self.cost_limits.items():
                projected_usage = cost_limit.current_usage + projected_cost
                
                if projected_usage > cost_limit.amount_usd:
                    violations.append({
                        "limit_type": limit_type.value,
                        "limit_amount": cost_limit.amount_usd,
                        "current_usage": cost_limit.current_usage,
                        "projected_usage": projected_usage,
                        "overage": projected_usage - cost_limit.amount_usd
                    })
                    can_proceed = False
            
            return {
                "can_proceed": can_proceed,
                "violations": violations,
                "projected_cost": projected_cost
            }
    
    async def get_usage_stats(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for specified time range."""
        async with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            recent_usage = [m for m in self.usage_history if m.timestamp >= cutoff_time]
            
            if not recent_usage:
                return {"message": "No usage data in specified time range"}
            
            # Calculate statistics
            total_requests = len(recent_usage)
            total_cost = sum(m.cost_usd for m in recent_usage)
            total_tokens = sum(m.total_tokens for m in recent_usage)
            successful_requests = sum(1 for m in recent_usage if m.success)
            
            # Provider breakdown
            provider_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
            for metrics in recent_usage:
                stats = provider_stats[metrics.provider]
                stats["requests"] += 1
                stats["cost"] += metrics.cost_usd
                stats["tokens"] += metrics.total_tokens
            
            # Model breakdown
            model_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
            for metrics in recent_usage:
                stats = model_stats[metrics.model_name]
                stats["requests"] += 1
                stats["cost"] += metrics.cost_usd
                stats["tokens"] += metrics.total_tokens
            
            # Performance statistics
            latencies = [m.latency_ms for m in recent_usage if m.success]
            costs_per_request = [m.cost_usd for m in recent_usage if m.success]
            
            return {
                "time_range_hours": time_range_hours,
                "summary": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": successful_requests / total_requests,
                    "total_cost_usd": total_cost,
                    "total_tokens": total_tokens,
                    "average_cost_per_request": total_cost / total_requests,
                    "average_tokens_per_request": total_tokens / total_requests
                },
                "performance": {
                    "average_latency_ms": statistics.mean(latencies) if latencies else 0,
                    "median_latency_ms": statistics.median(latencies) if latencies else 0,
                    "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
                    "average_cost_per_request": statistics.mean(costs_per_request) if costs_per_request else 0
                },
                "provider_breakdown": dict(provider_stats),
                "model_breakdown": dict(model_stats),
                "cost_limits": {
                    limit_type.value: {
                        "limit": limit.amount_usd,
                        "used": limit.current_usage,
                        "remaining": limit.amount_usd - limit.current_usage,
                        "utilization_percent": (limit.current_usage / limit.amount_usd) * 100
                    }
                    for limit_type, limit in self.cost_limits.items()
                }
            }

class ModelSelector:
    """Selects optimal model based on task complexity and cost constraints."""
    
    def __init__(self, available_models: Dict[str, ModelConfig]):
        self.available_models = available_models
        self.model_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.task_complexity_cache: Dict[str, float] = {}
    
    async def select_model(
        self,
        prompt: str,
        constraints: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ModelConfig:
        """Select optimal model based on prompt and constraints."""
        
        # Analyze task complexity
        complexity_score = await self._analyze_task_complexity(prompt, context or {})
        
        # Filter models based on constraints
        viable_models = await self._filter_viable_models(constraints, complexity_score)
        
        if not viable_models:
            # Fallback to cheapest available model
            cheapest_model = min(
                self.available_models.values(),
                key=lambda m: m.cost_per_1k_input_tokens + m.cost_per_1k_output_tokens
            )
            logger.warning(f"No viable models found, falling back to cheapest: {cheapest_model.model_name}")
            return cheapest_model
        
        # Score models based on cost-effectiveness
        best_model = await self._score_models(viable_models, complexity_score, constraints)
        
        logger.info(f"Selected model: {best_model.model_name} (complexity: {complexity_score:.2f})")
        return best_model
    
    async def _analyze_task_complexity(self, prompt: str, context: Dict[str, Any]) -> float:
        """Analyze task complexity to determine required model tier."""
        
        # Cache complexity analysis
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        if prompt_hash in self.task_complexity_cache:
            return self.task_complexity_cache[prompt_hash]
        
        complexity_score = 0.0
        
        # Prompt length analysis
        prompt_length = len(prompt.split())
        if prompt_length > 500:
            complexity_score += 0.4
        elif prompt_length > 200:
            complexity_score += 0.2
        elif prompt_length > 100:
            complexity_score += 0.1
        
        # Content analysis
        prompt_lower = prompt.lower()
        
        # Code generation/analysis tasks
        if any(keyword in prompt_lower for keyword in ["generate", "create", "write", "implement", "code"]):
            complexity_score += 0.3
        
        # Complex reasoning tasks
        if any(keyword in prompt_lower for keyword in ["analyze", "compare", "evaluate", "reason", "explain"]):
            complexity_score += 0.2
        
        # Multi-step tasks
        if any(keyword in prompt_lower for keyword in ["steps", "workflow", "process", "sequence"]):
            complexity_score += 0.2
        
        # Technical content
        if any(keyword in prompt_lower for keyword in ["api", "endpoint", "http", "json", "test", "validation"]):
            complexity_score += 0.1
        
        # Context complexity
        context_items = len(context)
        if context_items > 10:
            complexity_score += 0.2
        elif context_items > 5:
            complexity_score += 0.1
        
        # Cap at 1.0
        complexity_score = min(complexity_score, 1.0)
        
        # Cache result
        self.task_complexity_cache[prompt_hash] = complexity_score
        
        return complexity_score
    
    async def _filter_viable_models(
        self,
        constraints: Dict[str, Any],
        complexity_score: float
    ) -> List[ModelConfig]:
        """Filter models based on constraints and complexity."""
        
        max_cost_per_request = constraints.get("max_cost_per_request", 1.0)
        required_capabilities = constraints.get("required_capabilities", [])
        max_latency_ms = constraints.get("max_latency_ms", 30000)
        
        viable_models = []
        
        for model in self.available_models.values():
            # Check cost constraint (estimated)
            estimated_tokens = 100 + len(constraints.get("prompt", "").split()) * 1.2
            estimated_cost = (
                (estimated_tokens / 1000) * model.cost_per_1k_input_tokens +
                (estimated_tokens * 0.5 / 1000) * model.cost_per_1k_output_tokens
            )
            
            if estimated_cost > max_cost_per_request:
                continue
            
            # Check latency constraint
            if model.latency_ms_avg > max_latency_ms:
                continue
            
            # Check capability requirements
            if "function_calling" in required_capabilities and not model.supports_function_calling:
                continue
            
            if "streaming" in required_capabilities and not model.supports_streaming:
                continue
            
            # Check if model tier matches complexity
            required_tier = self._get_required_tier_for_complexity(complexity_score)
            if model.tier.value < required_tier.value:
                continue
            
            viable_models.append(model)
        
        return viable_models
    
    def _get_required_tier_for_complexity(self, complexity_score: float) -> ModelTier:
        """Get required model tier based on complexity score."""
        
        if complexity_score >= 0.8:
            return ModelTier.PREMIUM
        elif complexity_score >= 0.6:
            return ModelTier.ADVANCED
        elif complexity_score >= 0.3:
            return ModelTier.STANDARD
        else:
            return ModelTier.BASIC
    
    async def _score_models(
        self,
        models: List[ModelConfig],
        complexity_score: float,
        constraints: Dict[str, Any]
    ) -> ModelConfig:
        """Score models and select best one."""
        
        model_scores = []
        
        for model in models:
            # Base score from cost efficiency
            cost_score = 1.0 / (model.cost_per_1k_input_tokens + model.cost_per_1k_output_tokens + 0.001)
            
            # Performance bonus based on historical data
            if model.model_name in self.model_performance_history:
                avg_latency = statistics.mean(self.model_performance_history[model.model_name])
                performance_score = 1.0 / (avg_latency / 1000 + 0.1)  # Convert to seconds
            else:
                performance_score = 1.0 / (model.latency_ms_avg / 1000 + 0.1)
            
            # Tier appropriateness score
            required_tier = self._get_required_tier_for_complexity(complexity_score)
            tier_values = {"basic": 1, "standard": 2, "advanced": 3, "premium": 4}
            
            required_value = tier_values[required_tier.value]
            model_value = tier_values[model.tier.value]
            
            # Prefer models that match required tier (penalty for over/under capability)
            if model_value == required_value:
                tier_score = 1.0
            elif model_value > required_value:
                tier_score = 0.7  # Penalty for overkill
            else:
                tier_score = 0.3  # Penalty for underkill
            
            # Combined score
            final_score = (cost_score * 0.4) + (performance_score * 0.3) + (tier_score * 0.3)
            
            model_scores.append((model, final_score))
        
        # Return highest scoring model
        best_model, best_score = max(model_scores, key=lambda x: x[1])
        
        logger.debug(f"Model selection: {best_model.model_name} scored {best_score:.3f}")
        return best_model
    
    async def record_model_performance(self, model_name: str, latency_ms: int):
        """Record model performance for future selection."""
        
        history = self.model_performance_history[model_name]
        history.append(latency_ms)
        
        # Keep only recent history (last 100 requests)
        if len(history) > 100:
            history.pop(0)

class LLMGateway:
    """
    Main gateway for cost-bounded LLM interactions.
    
    Features:
    - Intelligent model selection based on task complexity
    - Cost tracking and budget enforcement
    - Rate limiting and request queuing
    - Response caching for cost optimization
    - Performance monitoring and optimization
    - Provider abstraction and failover
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = get_db_session()
        
        # Provider-specific settings (e.g., local provider like Ollama)
        self.provider_settings: Dict[str, Any] = config.get("providers", {})
        
        # Core components
        self.cost_tracker = CostTracker()
        self.cache = ResponseCache(
            max_size=config.get("cache_size", 1000),
            ttl_hours=config.get("cache_ttl_hours", 24)
        )
        
        # Rate limiters per provider
        self.rate_limiters: Dict[ModelProvider, RateLimiter] = {}
        
        # Model configurations
        self.available_models: Dict[str, ModelConfig] = {}
        self.model_selector = ModelSelector(self.available_models)

        # Provider registry
        self.providers: Dict[ModelProvider, BaseProvider] = {}
        
        # Request queue and processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: List[asyncio.Task] = []
        self.is_processing = False
        
        # Initialize components
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize gateway components."""
        
        try:
            # Load model configurations
            await self._load_model_configs()
            
            # Setup rate limiters
            await self._setup_rate_limiters()
            
            # Setup cost limits
            await self._setup_cost_limits()

            # Setup providers
            await self._setup_providers()
            
            # Start request processing
            await self._start_processing()
            
            logger.info("LLM Gateway initialized successfully")
            
        except Exception as e:
            logger.error(f"Gateway initialization failed: {str(e)}")
            raise
    
    async def _load_model_configs(self):
        """Load available model configurations."""
        
        # Default model configurations (would normally load from config/database)
        default_models = [
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o-mini",
                tier=ModelTier.STANDARD,
                cost_per_1k_input_tokens=0.00015,
                cost_per_1k_output_tokens=0.0006,
                max_tokens=16384,
                context_window=128000,
                supports_streaming=True,
                supports_function_calling=True,
                latency_ms_avg=1200
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                tier=ModelTier.PREMIUM,
                cost_per_1k_input_tokens=0.0025,
                cost_per_1k_output_tokens=0.01,
                max_tokens=4096,
                context_window=128000,
                supports_streaming=True,
                supports_function_calling=True,
                latency_ms_avg=2000
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                tier=ModelTier.BASIC,
                cost_per_1k_input_tokens=0.00025,
                cost_per_1k_output_tokens=0.00125,
                max_tokens=4096,
                context_window=200000,
                supports_streaming=True,
                supports_function_calling=False,
                latency_ms_avg=800
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                tier=ModelTier.ADVANCED,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                max_tokens=8192,
                context_window=200000,
                supports_streaming=True,
                supports_function_calling=True,
                latency_ms_avg=1500
            )
        ]
        
        # Load custom models from config
        custom_models = self.config.get("custom_models", [])
        
        # Conditionally add local (free) models if provider configured
        local_cfg = self.provider_settings.get(ModelProvider.LOCAL.value)
        if local_cfg:
            default_local_models = local_cfg.get("default_models", ["llama3.1:8b"])  # Ollama model names
            for m in default_local_models:
                default_models.append(
                    ModelConfig(
                        provider=ModelProvider.LOCAL,
                        model_name=m,
                        tier=ModelTier.STANDARD,
                        cost_per_1k_input_tokens=0.0,
                        cost_per_1k_output_tokens=0.0,
                        max_tokens=local_cfg.get("max_tokens", 4096),
                        context_window=local_cfg.get("context_window", 8192),
                        supports_streaming=False,
                        supports_function_calling=False,
                        latency_ms_avg=local_cfg.get("latency_ms_avg", 800)
                    )
                )
        
        all_models = default_models + [
            ModelConfig(**model_data) for model_data in custom_models
        ]
        
        # Index by model name
        for model in all_models:
            self.available_models[model.model_name] = model
        
        logger.info(f"Loaded {len(self.available_models)} model configurations")
    
    async def _setup_rate_limiters(self):
        """Setup rate limiters for each provider."""
        
        rate_limits = self.config.get("rate_limits", {
            ModelProvider.OPENAI.value: {"requests_per_minute": 60, "window_seconds": 60},
            ModelProvider.ANTHROPIC.value: {"requests_per_minute": 50, "window_seconds": 60}
        })
        
        for provider_name, limits in rate_limits.items():
            try:
                provider = ModelProvider(provider_name)
                self.rate_limiters[provider] = RateLimiter(
                    max_requests=limits["requests_per_minute"],
                    window_seconds=limits["window_seconds"]
                )
            except ValueError:
                logger.warning(f"Unknown provider in rate limits: {provider_name}")
    
    async def _setup_cost_limits(self):
        """Setup cost limits from configuration."""
        
        cost_limits = self.config.get("cost_limits", {})
        
        for limit_type_str, amount in cost_limits.items():
            try:
                limit_type = CostLimitType(limit_type_str)
                await self.cost_tracker.set_cost_limit(limit_type, amount)
            except ValueError:
                logger.warning(f"Unknown cost limit type: {limit_type_str}")
    
    async def _setup_providers(self):
        """Instantiate provider implementations based on configuration."""
        try:
            # Local provider (Ollama)
            local_cfg = self.provider_settings.get(ModelProvider.LOCAL.value)
            if local_cfg:
                self.providers[ModelProvider.LOCAL] = LocalOllamaProvider(settings=local_cfg)
                logger.info("Registered LocalOllamaProvider")
        except Exception as e:
            logger.error(f"Provider setup failed: {str(e)}")

    async def _start_processing(self):
        """Start background request processing."""
        
        if self.is_processing:
            return
        
        self.is_processing = True
        
        # Start worker tasks
        num_workers = self.config.get("worker_threads", 3)
        for i in range(num_workers):
            task = asyncio.create_task(self._process_requests())
            self.processing_tasks.append(task)
        
        logger.info(f"Started {num_workers} request processing workers")
    
    async def _process_requests(self):
        """Background task to process queued requests."""
        
        while self.is_processing:
            try:
                # Get request from queue
                request_data = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                request: LLMRequest = request_data["request"]
                response_future: asyncio.Future = request_data["future"]
                
                # Process request
                try:
                    response = await self._execute_request(request)
                    response_future.set_result(response)
                except Exception as e:
                    response_future.set_exception(e)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Request processing error: {str(e)}")
    
    async def _execute_request(self, request: LLMRequest) -> LLMResponse:
        """Execute individual LLM request."""
        
        start_time = time.time()
        
        try:
            # Check cache first
            if request.cache_enabled:
                cached_response = await self.cache.get(
                    request.prompt,
                    request.model_preference or "auto",
                    request.temperature
                )
                
                if cached_response:
                    logger.debug(f"Cache hit for request {request.request_id}")
                    cached_response.request_id = request.request_id
                    return cached_response
            
            # Select model
            constraints = {
                "max_cost_per_request": self.config.get("max_cost_per_request", 0.1),
                "max_latency_ms": request.timeout_seconds * 1000,
                "required_capabilities": [],
                "prompt": request.prompt,
                "prefer_local": self.config.get("prefer_local", True)
            }
            
            if request.model_preference:
                if request.model_preference in self.available_models:
                    selected_model = self.available_models[request.model_preference]
                else:
                    logger.warning(f"Preferred model {request.model_preference} not available")
                    selected_model = await self.model_selector.select_model(request.prompt, constraints, request.context)
            else:
                selected_model = await self.model_selector.select_model(request.prompt, constraints, request.context)
            
            # Estimate cost
            estimated_input_tokens = len(request.prompt.split()) * 1.3  # Rough estimation
            estimated_output_tokens = min(request.max_tokens or 1000, selected_model.max_tokens)
            
            estimated_cost = (
                (estimated_input_tokens / 1000) * selected_model.cost_per_1k_input_tokens +
                (estimated_output_tokens / 1000) * selected_model.cost_per_1k_output_tokens
            )
            
            # Check cost limits
            cost_check = await self.cost_tracker.check_cost_limits(estimated_cost)
            if not cost_check["can_proceed"]:
                raise Exception(f"Cost limit exceeded: {cost_check['violations']}")
            
            # Check rate limits
            if selected_model.provider in self.rate_limiters:
                rate_limiter = self.rate_limiters[selected_model.provider]
                
                if not await rate_limiter.check_rate_limit():
                    wait_time = await rate_limiter.get_wait_time()
                    if wait_time > 0:
                        logger.info(f"Rate limited, waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        
                        # Try again after waiting
                        if not await rate_limiter.check_rate_limit():
                            raise Exception("Rate limit still exceeded after waiting")
            
            # Make actual LLM request
            response = await self._call_llm_provider(request, selected_model)
            
            # Calculate actual cost
            actual_cost = (
                (response.tokens_used / 1000) * selected_model.cost_per_1k_input_tokens +
                (len(response.content.split()) * 1.3 / 1000) * selected_model.cost_per_1k_output_tokens
            )
            
            response.cost_usd = actual_cost
            response.latency_ms = int((time.time() - start_time) * 1000)
            
            # Record metrics
            metrics = RequestMetrics(
                request_id=request.request_id,
                provider=selected_model.provider,
                model_name=selected_model.model_name,
                prompt_tokens=response.tokens_used,
                completion_tokens=len(response.content.split()),
                total_tokens=response.tokens_used + len(response.content.split()),
                cost_usd=actual_cost,
                latency_ms=response.latency_ms,
                timestamp=datetime.now(),
                success=True,
                cache_hit=False
            )
            
            await self.cost_tracker.add_usage(metrics)
            await self.model_selector.record_model_performance(selected_model.model_name, response.latency_ms)
            
            # Cache response
            if request.cache_enabled:
                await self.cache.put(request.prompt, selected_model.model_name, request.temperature, response)
            
            logger.info(f"Request {request.request_id} completed: {selected_model.model_name}, ${actual_cost:.6f}, {response.latency_ms}ms")
            
            return response
            
        except Exception as e:
            # Record failed request metrics
            metrics = RequestMetrics(
                request_id=request.request_id,
                provider=ModelProvider.OPENAI,  # Default for error tracking
                model_name="unknown",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            await self.cost_tracker.add_usage(metrics)
            
            logger.error(f"Request {request.request_id} failed: {str(e)}")
            raise
    
    async def _call_llm_provider(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        """Make actual call to LLM provider."""
        
        # This would integrate with actual provider APIs
        # For now, return mock response
        
        # Delegate to provider registry if available
        provider_impl = self.providers.get(model.provider)
        if provider_impl:
            return await provider_impl.generate(request, model)

        if model.provider == ModelProvider.OPENAI:
            return await self._call_openai(request, model)
        elif model.provider == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(request, model)
        else:
            # Mock response for other providers
            await asyncio.sleep(model.latency_ms_avg / 1000)  # Simulate latency
            
            return LLMResponse(
                request_id=request.request_id,
                content=f"Mock response from {model.model_name} for prompt: {request.prompt[:50]}...",
                model_used=model.model_name,
                provider_used=model.provider,
                tokens_used=len(request.prompt.split()),
                cost_usd=0.0,  # Will be calculated later
                latency_ms=0   # Will be calculated later
            )
    
    async def _call_openai(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        """Call OpenAI API."""
        
        try:
            # Mock OpenAI API call (would use actual openai library)
            await asyncio.sleep(model.latency_ms_avg / 1000)  # Simulate latency
            
            # Simulate token usage
            input_tokens = len(request.prompt.split())
            output_tokens = min(request.max_tokens or 1000, model.max_tokens // 2)
            
            response_content = f"Generated response from {model.model_name} for API testing task. This is a detailed analysis and implementation based on the provided prompt."
            
            return LLMResponse(
                request_id=request.request_id,
                content=response_content,
                model_used=model.model_name,
                provider_used=model.provider,
                tokens_used=input_tokens,
                cost_usd=0.0,  # Calculated later
                latency_ms=0   # Calculated later
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    async def _call_anthropic(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        """Call Anthropic API."""
        
        try:
            # Mock Anthropic API call (would use actual anthropic library)
            await asyncio.sleep(model.latency_ms_avg / 1000)  # Simulate latency
            
            # Simulate token usage
            input_tokens = len(request.prompt.split())
            output_tokens = min(request.max_tokens or 1000, model.max_tokens // 2)
            
            response_content = f"Claude response from {model.model_name} providing comprehensive API testing analysis and recommendations."
            
            return LLMResponse(
                request_id=request.request_id,
                content=response_content,
                model_used=model.model_name,
                provider_used=model.provider,
                tokens_used=input_tokens,
                cost_usd=0.0,  # Calculated later
                latency_ms=0   # Calculated later
            )
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
        """Call a local provider (e.g., Ollama) to get a free response."""
        
        local_cfg = self.provider_settings.get(ModelProvider.LOCAL.value, {})
        provider_type = local_cfg.get("type", "ollama")
        base_url = local_cfg.get("base_url", "http://localhost:11434")
        timeout = request.timeout_seconds
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if provider_type == "ollama":
                    # Use Ollama /api/generate endpoint (non-streaming)
                    endpoint = f"{base_url.rstrip('/')}/api/generate"
                    payload = {
                        "model": model.model_name,
                        "prompt": request.prompt,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature
                        }
                    }
                    if request.max_tokens:
                        # Ollama's num_predict controls max new tokens
                        payload["options"]["num_predict"] = min(request.max_tokens, model.max_tokens)
                    
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data.get("response") or data.get("message", "")
                else:
                    # Unknown local provider type; fallback to mock
                    await asyncio.sleep(model.latency_ms_avg / 1000)
                    content = f"Local provider mock response for {model.model_name}"
                
                # We typically lack token counts from local backends; approximate input tokens
                input_tokens = len(request.prompt.split())
                
                return LLMResponse(
                    request_id=request.request_id,
                    content=content,
                    model_used=model.model_name,
                    provider_used=model.provider,
                    tokens_used=input_tokens,
                    cost_usd=0.0,
                    latency_ms=0
                )
        except httpx.HTTPError as e:
            msg = (
                f"Local provider request failed: {str(e)}. "
                f"Ensure the local server is running at {base_url} and the model '{model.model_name}' is available."
            )
            logger.error(msg)
            raise Exception(msg)
        except Exception as e:
            logger.error(f"Local provider error: {str(e)}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        model_preference: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        priority: RequestPriority = RequestPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate text using optimal model selection."""
        
        request = LLMRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            model_preference=model_preference,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            context=context or {}
        )
        
        # Create future for async response
        response_future = asyncio.Future()
        
        # Queue request
        await self.request_queue.put({
            "request": request,
            "future": response_future
        })
        
        # Wait for response
        try:
            response = await asyncio.wait_for(response_future, timeout=request.timeout_seconds)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {request.request_id} timed out")
            raise Exception("Request timeout")
    
    async def generate_test_code(
        self,
        api_spec: Dict[str, Any],
        test_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate test code for API specification."""
        
        # Build specialized prompt for test generation
        prompt = self._build_test_generation_prompt(api_spec, test_type, constraints or {})
        
        # Use advanced model for code generation
        response = await self.generate_text(
            prompt=prompt,
            tier_preference=ModelTier.ADVANCED,
            max_tokens=2048,
            temperature=0.3,  # Lower temperature for code generation
            context={
                "task_type": "code_generation",
                "api_spec_id": api_spec.get("id"),
                "test_type": test_type
            }
        )
        
        return response
    
    def _build_test_generation_prompt(
        self,
        api_spec: Dict[str, Any],
        test_type: str,
        constraints: Dict[str, Any]
    ) -> str:
        """Build specialized prompt for test generation."""
        
        base_url = api_spec.get("base_url", "")
        endpoints = api_spec.get("endpoints", [])
        auth_method = api_spec.get("auth_method", "none")
        
        prompt = f"""
Generate comprehensive {test_type} tests for the following API:

API Base URL: {base_url}
Authentication: {auth_method}

Available Endpoints:
"""
        
        for endpoint in endpoints[:5]:  # Limit to first 5 endpoints
            method = endpoint.get("method", "GET")
            path = endpoint.get("path", "/")
            description = endpoint.get("description", "")
            
            prompt += f"- {method} {path}: {description}\n"
        
        prompt += f"""

Test Requirements:
- Generate Python code using httpx for HTTP requests
- Include proper error handling and assertions
- Test both success and error scenarios
- Validate response structure and data types
- Include performance timing
- Follow API testing best practices

Constraints:
- Maximum execution time: {constraints.get('max_execution_time', 300)} seconds
- Maximum network requests: {constraints.get('max_requests', 100)}
- Security level: {constraints.get('security_level', 'medium')}

Please generate complete, executable test code that can run in a sandboxed environment.
"""
        
        return prompt
    
    async def generate_test_analysis(
        self,
        test_results: List[Dict[str, Any]],
        analysis_type: str = "comprehensive"
    ) -> LLMResponse:
        """Generate analysis of test results."""
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(test_results, analysis_type)
        
        # Use standard model for analysis
        response = await self.generate_text(
            prompt=prompt,
            tier_preference=ModelTier.STANDARD,
            max_tokens=1500,
            temperature=0.5,
            context={
                "task_type": "analysis",
                "analysis_type": analysis_type,
                "test_count": len(test_results)
            }
        )
        
        return response
    
    def _build_analysis_prompt(self, test_results: List[Dict[str, Any]], analysis_type: str) -> str:
        """Build prompt for test result analysis."""
        
        # Summarize test results
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.get("status") == "passed")
        failed_tests = total_tests - passed_tests
        
        prompt = f"""
Analyze the following API test results and provide {analysis_type} insights:

Test Summary:
- Total tests: {total_tests}
- Passed: {passed_tests}
- Failed: {failed_tests}
- Success rate: {(passed_tests/total_tests)*100:.1f}%

Test Results Sample:
"""
        
        # Include sample of results (limit to avoid token limits)
        for i, result in enumerate(test_results[:10]):
            endpoint = result.get("endpoint", "unknown")
            status = result.get("status", "unknown")
            response_time = result.get("response_time_ms", 0)
            
            prompt += f"{i+1}. {endpoint}: {status} ({response_time}ms)\n"
        
        if len(test_results) > 10:
            prompt += f"... and {len(test_results) - 10} more results\n"
        
        prompt += """
Please provide:
1. Overall assessment of API health and quality
2. Identification of problematic endpoints or patterns
3. Performance analysis and optimization recommendations
4. Reliability and consistency insights
5. Suggested areas for additional testing

Focus on actionable insights that can improve API quality and testing strategy.
"""
        
        return prompt
    
    async def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost and usage summary."""
        
        try:
            # Get usage stats for different time ranges
            hourly_stats = await self.cost_tracker.get_usage_stats(1)
            daily_stats = await self.cost_tracker.get_usage_stats(24)
            weekly_stats = await self.cost_tracker.get_usage_stats(24 * 7)
            
            # Get cache statistics
            cache_stats = {
                "cache_size": len(self.cache.cache),
                "max_cache_size": self.cache.max_size,
                "cache_utilization": len(self.cache.cache) / self.cache.max_size
            }
            
            # Calculate cost projections
            if daily_stats.get("summary", {}).get("total_cost_usd", 0) > 0:
                daily_cost = daily_stats["summary"]["total_cost_usd"]
                projected_monthly_cost = daily_cost * 30
            else:
                projected_monthly_cost = 0.0
            
            return {
                "current_costs": {
                    "last_hour": hourly_stats.get("summary", {}).get("total_cost_usd", 0),
                    "last_24h": daily_stats.get("summary", {}).get("total_cost_usd", 0),
                    "last_7d": weekly_stats.get("summary", {}).get("total_cost_usd", 0)
                },
                "projections": {
                    "monthly_projected_usd": projected_monthly_cost
                },
                "usage_stats": {
                    "requests_24h": daily_stats.get("summary", {}).get("total_requests", 0),
                    "avg_cost_per_request": daily_stats.get("summary", {}).get("average_cost_per_request", 0),
                    "success_rate": daily_stats.get("summary", {}).get("success_rate", 0)
                },
                "cache_performance": cache_stats,
                "cost_limits": daily_stats.get("cost_limits", {}),
                "provider_breakdown": daily_stats.get("provider_breakdown", {}),
                "model_breakdown": daily_stats.get("model_breakdown", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get cost summary: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_costs(self) -> Dict[str, Any]:
        """Analyze usage and provide cost optimization recommendations."""
        
        try:
            stats = await self.cost_tracker.get_usage_stats(24 * 7)  # 7 days
            
            if not stats.get("summary"):
                return {"message": "Insufficient usage data for optimization"}
            
            recommendations = []
            potential_savings = 0.0
            
            # Analyze model usage patterns
            model_breakdown = stats.get("model_breakdown", {})
            
            for model_name, model_stats in model_breakdown.items():
                if model_name in self.available_models:
                    model_config = self.available_models[model_name]
                    
                    # Check if using expensive model for simple tasks
                    if model_config.tier in [ModelTier.PREMIUM, ModelTier.ADVANCED]:
                        avg_complexity = self._estimate_average_complexity_for_model(model_name)
                        
                        if avg_complexity < 0.3 and model_config.tier == ModelTier.PREMIUM:
                            # Suggest downgrading to advanced model
                            current_cost = model_stats["cost"]
                            potential_saving = current_cost * 0.4  # Assume 40% savings
                            potential_savings += potential_saving
                            
                            recommendations.append({
                                "type": "model_downgrade",
                                "current_model": model_name,
                                "suggested_tier": ModelTier.ADVANCED.value,
                                "potential_saving_usd": potential_saving,
                                "reason": "Low complexity tasks detected"
                            })
                        
                        elif avg_complexity < 0.5 and model_config.tier == ModelTier.ADVANCED:
                            # Suggest downgrading to standard model
                            current_cost = model_stats["cost"]
                            potential_saving = current_cost * 0.6  # Assume 60% savings
                            potential_savings += potential_saving
                            
                            recommendations.append({
                                "type": "model_downgrade",
                                "current_model": model_name,
                                "suggested_tier": ModelTier.STANDARD.value,
                                "potential_saving_usd": potential_saving,
                                "reason": "Simple tasks detected"
                            })
            
            # Analyze cache effectiveness
            cache_hit_rate = self._calculate_cache_hit_rate()
            if cache_hit_rate < 0.2:
                recommendations.append({
                    "type": "cache_optimization",
                    "suggestion": "Increase cache TTL or size",
                    "current_hit_rate": cache_hit_rate,
                    "reason": "Low cache hit rate indicates potential for better caching"
                })
            
            # Analyze request batching opportunities
            avg_requests_per_hour = stats["summary"]["total_requests"] / (24 * 7)
            if avg_requests_per_hour > 10:
                recommendations.append({
                    "type": "batching_opportunity",
                    "suggestion": "Implement request batching for similar prompts",
                    "potential_saving_percent": 15,
                    "reason": "High request volume suitable for batching optimization"
                })
            
            return {
                "total_potential_savings_usd": potential_savings,
                "recommendations": recommendations,
                "current_usage_stats": stats["summary"],
                "optimization_score": self._calculate_optimization_score(stats, potential_savings)
            }
            
        except Exception as e:
            logger.error(f"Cost optimization analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_average_complexity_for_model(self, model_name: str) -> float:
        """Estimate average task complexity for a model based on usage history."""
        
        # This would analyze historical prompts and their complexity
        # For now, return reasonable estimates
        
        recent_requests = [m for m in self.cost_tracker.usage_history 
                         if m.model_name == model_name and m.timestamp >= datetime.now() - timedelta(days=7)]
        
        if not recent_requests:
            return 0.5  # Default medium complexity
        
        # Rough complexity estimation based on token usage and success rate
        avg_tokens = statistics.mean([m.total_tokens for m in recent_requests])
        success_rate = sum(1 for m in recent_requests if m.success) / len(recent_requests)
        
        # Simple heuristic: more tokens and lower success rate = higher complexity
        complexity = min(1.0, (avg_tokens / 1000) * 0.5 + (1.0 - success_rate) * 0.3 + 0.2)
        
        return complexity
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent requests."""
        
        recent_requests = [m for m in self.cost_tracker.usage_history 
                         if m.timestamp >= datetime.now() - timedelta(hours=24)]
        
        if not recent_requests:
            return 0.0
        
        cache_hits = sum(1 for m in recent_requests if m.cache_hit)
        return cache_hits / len(recent_requests)
    
    def _calculate_optimization_score(self, stats: Dict[str, Any], potential_savings: float) -> float:
        """Calculate optimization score (0-100)."""
        
        current_cost = stats.get("summary", {}).get("total_cost_usd", 0)
        
        if current_cost == 0:
            return 100.0  # Perfect score if no costs
        
        # Factors affecting optimization score
        cache_hit_rate = self._calculate_cache_hit_rate()
        success_rate = stats.get("summary", {}).get("success_rate", 1.0)
        savings_potential = min(1.0, potential_savings / current_cost)
        
        # Weight different factors
        cache_score = cache_hit_rate * 30
        success_score = success_rate * 40  
        efficiency_score = (1.0 - savings_potential) * 30
        
        total_score = cache_score + success_score + efficiency_score
        
        return min(100.0, max(0.0, total_score))
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and processing status."""
        
        return {
            "queue_size": self.request_queue.qsize(),
            "active_workers": len([t for t in self.processing_tasks if not t.done()]),
            "total_workers": len(self.processing_tasks),
            "is_processing": self.is_processing,
            "cache_size": len(self.cache.cache),
            "available_models": len(self.available_models)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of gateway components."""
        
        health_status = {
            "overall": "healthy",
            "components": {},
            "issues": []
        }
        
        try:
            # Check cost tracker
            try:
                await self.cost_tracker.get_usage_stats(1)
                health_status["components"]["cost_tracker"] = "healthy"
            except Exception as e:
                health_status["components"]["cost_tracker"] = "unhealthy"
                health_status["issues"].append(f"Cost tracker error: {str(e)}")
            
            # Check cache
            try:
                await self.cache.clear_expired()
                health_status["components"]["cache"] = "healthy"
            except Exception as e:
                health_status["components"]["cache"] = "unhealthy"
                health_status["issues"].append(f"Cache error: {str(e)}")
            
            # Check rate limiters
            healthy_rate_limiters = 0
            for provider, limiter in self.rate_limiters.items():
                try:
                    await limiter.check_rate_limit()
                    healthy_rate_limiters += 1
                except Exception as e:
                    health_status["issues"].append(f"Rate limiter error for {provider.value}: {str(e)}")
            
            health_status["components"]["rate_limiters"] = f"{healthy_rate_limiters}/{len(self.rate_limiters)} healthy"
            
            # Check request processing
            active_workers = len([t for t in self.processing_tasks if not t.done()])
            health_status["components"]["request_processing"] = {
                "status": "healthy" if active_workers > 0 else "degraded",
                "active_workers": active_workers,
                "queue_size": self.request_queue.qsize()
            }
            
            # Overall health assessment
            if health_status["issues"]:
                health_status["overall"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "overall": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Gracefully shutdown the gateway."""
        
        logger.info("Shutting down LLM Gateway...")
        
        # Stop processing
        self.is_processing = False
        
        # Wait for current requests to complete (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.processing_tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some processing tasks did not complete within timeout")
        
        # Cancel remaining tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
        
        # Clear cache
        self.cache.cache.clear()
        
        # Close database session
        if hasattr(self, 'db'):
            self.db.close()
        
        logger.info("LLM Gateway shutdown complete")
    
    def __del__(self):
        """Cleanup on garbage collection."""
        
        if hasattr(self, 'is_processing') and self.is_processing:
            # Cannot await in __del__, so just set flag
            self.is_processing = False
        
        if hasattr(self, 'db'):
            self.db.close()


class PromptOptimizer:
    """Optimizes prompts for better performance and cost efficiency."""
    
    def __init__(self):
        self.optimization_patterns = {
            # Common optimizations for API testing prompts
            "redundant_phrases": [
                ("please generate", "generate"),
                ("i need you to", ""),
                ("can you help me", ""),
                ("please create", "create"),
                ("make sure to", "ensure")
            ],
            "verbose_instructions": [
                ("comprehensive and detailed", "comprehensive"),
                ("thorough and complete", "complete"),
                ("extensive and exhaustive", "extensive")
            ]
        }
    
    async def optimize_prompt(self, prompt: str, target_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Optimize prompt for cost and performance."""
        
        original_prompt = prompt
        optimized_prompt = prompt.lower()
        
        optimization_log = []
        
        # Remove redundant phrases
        for original, replacement in self.optimization_patterns["redundant_phrases"]:
            if original in optimized_prompt:
                optimized_prompt = optimized_prompt.replace(original, replacement)
                optimization_log.append(f"Removed redundant phrase: '{original}'")
        
        # Simplify verbose instructions
        for original, replacement in self.optimization_patterns["verbose_instructions"]:
            if original in optimized_prompt:
                optimized_prompt = optimized_prompt.replace(original, replacement)
                optimization_log.append(f"Simplified verbose instruction: '{original}' -> '{replacement}'")
        
        # Remove extra whitespace
        optimized_prompt = " ".join(optimized_prompt.split())
        
        # Token count estimation
        original_tokens = len(original_prompt.split()) * 1.3
        optimized_tokens = len(optimized_prompt.split()) * 1.3
        
        # Target token optimization
        if target_tokens and optimized_tokens > target_tokens:
            # Truncate while preserving important parts
            optimized_prompt = await self._truncate_intelligently(optimized_prompt, target_tokens)
            optimization_log.append(f"Truncated to target {target_tokens} tokens")
        
        token_savings = original_tokens - optimized_tokens
        cost_savings_percent = (token_savings / original_tokens) * 100 if original_tokens > 0 else 0
        
        return {
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "optimization_log": optimization_log,
            "token_reduction": {
                "original_tokens": int(original_tokens),
                "optimized_tokens": int(optimized_tokens),
                "tokens_saved": int(token_savings),
                "cost_savings_percent": cost_savings_percent
            },
            "should_use_optimized": token_savings > 5  # Only use if meaningful savings
        }
    
    async def _truncate_intelligently(self, prompt: str, target_tokens: int) -> str:
        """Intelligently truncate prompt while preserving important content."""
        
        target_chars = target_tokens * 4  # Rough chars per token
        
        if len(prompt) <= target_chars:
            return prompt
        
        # Try to preserve beginning and end, remove middle if possible
        sentences = prompt.split('. ')
        
        if len(sentences) > 3:
            # Keep first and last sentences, truncate middle
            first_sentence = sentences[0] + '.'
            last_sentence = sentences[-1]
            
            truncated = f"{first_sentence} ... {last_sentence}"
            
            if len(truncated) <= target_chars:
                return truncated
        
        # Simple truncation as fallback
        return prompt[:target_chars] + "..."


class RequestBatcher:
    """Batches similar requests for cost optimization."""
    
    def __init__(self, batch_size: int = 5, batch_timeout_seconds: int = 2):
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.pending_batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def add_request(self, request: LLMRequest) -> Optional[List[LLMRequest]]:
        """Add request to batch, returns batch if ready."""
        
        async with self._lock:
            # Create batch key based on model preference and context
            batch_key = self._get_batch_key(request)
            
            # Add to pending batch
            self.pending_batches[batch_key].append({
                "request": request,
                "timestamp": time.time()
            })
            
            current_batch = self.pending_batches[batch_key]
            
            # Check if batch is ready
            if len(current_batch) >= self.batch_size:
                # Return batch and clear
                batch_requests = [item["request"] for item in current_batch]
                del self.pending_batches[batch_key]
                return batch_requests
            
            # Check if oldest request in batch is too old
            oldest_timestamp = min(item["timestamp"] for item in current_batch)
            if time.time() - oldest_timestamp >= self.batch_timeout_seconds:
                # Return partial batch and clear
                batch_requests = [item["request"] for item in current_batch]
                del self.pending_batches[batch_key]
                return batch_requests
            
            return None
    
    def _get_batch_key(self, request: LLMRequest) -> str:
        """Generate batch key for grouping similar requests."""
        
        # Group by model preference and task type
        key_parts = [
            request.model_preference or "auto",
            request.context.get("task_type", "general"),
            str(request.tier_preference.value if request.tier_preference else "auto"),
            str(int(request.temperature * 10))  # Group by temperature
        ]
        
        return "|".join(key_parts)
    
    async def get_pending_batches(self) -> Dict[str, Any]:
        """Get information about pending batches."""
        
        async with self._lock:
            batch_info = {}
            
            for batch_key, requests in self.pending_batches.items():
                oldest_timestamp = min(item["timestamp"] for item in requests)
                
                batch_info[batch_key] = {
                    "size": len(requests),
                    "age_seconds": time.time() - oldest_timestamp,
                    "ready_in_seconds": max(0, self.batch_timeout_seconds - (time.time() - oldest_timestamp))
                }
            
            return {
                "total_pending_batches": len(batch_info),
                "total_pending_requests": sum(len(requests) for requests in self.pending_batches.values()),
                "batch_details": batch_info
            }
