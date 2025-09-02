"""
Secure AI integration module with authentication, rate limiting, and output validation.
"""

import os
import time
import hashlib
import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    import openai
    from openai import OpenAI
    import tiktoken
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    OpenAI = None

from ..utils.logger import get_logger, security_logger

logger = get_logger(__name__)

@dataclass
class AIUsageMetrics:
    """AI usage metrics for monitoring and rate limiting."""
    requests_per_hour: int = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    last_request: Optional[datetime] = None
    errors: int = 0

class SecureAIClient:
    """Secure AI client with rate limiting, authentication, and monitoring."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.ai_rate_limit = int(os.getenv("AI_RATE_LIMIT_PER_HOUR", "1000"))
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured. AI features will be limited.")
            self.client = None
        elif not AI_AVAILABLE:
            logger.warning("OpenAI library not available. AI features will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # Rate limiting tracking
        self.usage_metrics = defaultdict(AIUsageMetrics)
        self.request_times = defaultdict(lambda: deque(maxlen=1000))
        
        # Token counting
        if AI_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None
        
        # Cost estimation (approximate prices per 1K tokens)
        self.token_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
    
    def _check_rate_limits(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        # Clean old requests
        user_requests = self.request_times[user_id]
        while user_requests and user_requests[0] < hour_ago:
            user_requests.popleft()
        
        # Check hourly limit
        if len(user_requests) >= self.ai_rate_limit:
            security_logger.log_security_event(
                event_type="ai_rate_limit_exceeded",
                severity="warning",
                message="AI rate limit exceeded for user",
                user_id=user_id,
                details={"requests_in_hour": len(user_requests), "limit": self.ai_rate_limit}
            )
            return False
        
        return True
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer:
            # Fallback estimation
            return int(len(text.split()) * 1.3)
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            return int(len(text.split()) * 1.3)
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for API call."""
        costs = self.token_costs.get(model, self.token_costs["gpt-4"])
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def _validate_ai_output(self, output: str, expected_format: str = "json") -> bool:
        """Validate AI output format and content."""
        if not output or len(output.strip()) == 0:
            return False
        
        if expected_format == "json":
            try:
                parsed = json.loads(output)
                
                # Basic security checks
                if isinstance(parsed, dict):
                    # Check for potential injection attempts
                    dangerous_keys = ['__import__', 'eval', 'exec', 'open', 'file']
                    for key in dangerous_keys:
                        if key in str(parsed).lower():
                            logger.warning(f"Potentially dangerous content in AI output: {key}")
                            return False
                
                return True
            except json.JSONDecodeError:
                return False
        
        # Additional validation for other formats
        if len(output) > 50000:  # Suspiciously large output
            logger.warning("AI output exceeds expected size limit")
            return False
        
        return True
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize AI prompt to prevent injection attacks."""
        if not prompt:
            return ""
        
        # Remove potential command injection patterns
        dangerous_patterns = [
            r'`[^`]*`',  # Backticks
            r'\$\([^)]*\)',  # Command substitution
            r'\${[^}]*}',  # Variable substitution
            r'exec\s*\(',  # exec calls
            r'eval\s*\(',  # eval calls
            r'import\s+os',  # os imports
            r'__import__',  # import function
        ]
        
        for pattern in dangerous_patterns:
            prompt = re.sub(pattern, '[FILTERED]', prompt, flags=re.IGNORECASE)
        
        # Truncate if too long
        if len(prompt) > 10000:
            prompt = prompt[:10000] + "... [truncated for security]"
        
        return prompt
    
    async def generate_completion(
        self,
        prompt: str,
        user_id: str = "system",
        model: str = "gpt-4-turbo",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        expected_format: str = "json",
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate AI completion with security controls."""
        if not self.client:
            raise ValueError("AI client not configured. Please set OPENAI_API_KEY.")
        
        # Rate limiting check
        if not self._check_rate_limits(user_id):
            raise ValueError("AI rate limit exceeded. Please try again later.")
        
        # Input validation
        input_tokens = self._count_tokens(prompt)
        if input_tokens > 8000:  # Conservative limit
            raise ValueError("Input prompt too long. Please reduce prompt size.")
        
        # Sanitize inputs
        prompt = self._sanitize_prompt(prompt)
        if system_message:
            system_message = self._sanitize_prompt(system_message)
        
        try:
            # Record request time
            now = datetime.now(timezone.utc)
            self.request_times[user_id].append(now)
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            start_time = time.time()
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=30
            )
            
            duration = time.time() - start_time
            content = response.choices[0].message.content
            
            # Validate output
            if not self._validate_ai_output(content, expected_format):
                security_logger.log_security_event(
                    event_type="ai_output_validation_failed",
                    severity="warning",
                    message="AI output failed validation",
                    user_id=user_id,
                    details={"model": model, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:12]}
                )
                raise ValueError("AI output failed security validation")
            
            # Calculate metrics
            output_tokens = self._count_tokens(content)
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            
            # Update usage metrics
            metrics = self.usage_metrics[user_id]
            metrics.requests_per_hour += 1
            metrics.tokens_used += input_tokens + output_tokens
            metrics.cost_estimate += cost
            metrics.last_request = now
            
            # Log successful completion
            logger.info(
                f"AI completion successful - User: {hashlib.sha256(user_id.encode()).hexdigest()[:12]} Model: {model}"
            )
            
            return {
                "content": content,
                "metadata": {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_estimate": cost,
                    "duration": duration,
                    "timestamp": now.isoformat()
                }
            }
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                logger.warning(f"OpenAI rate limit exceeded: {e}")
                raise ValueError("AI service rate limit exceeded. Please try again later.")
            elif "authentication" in str(e).lower():
                security_logger.log_security_event(
                    event_type="ai_authentication_failed",
                    severity="error",
                    message="OpenAI authentication failed",
                    details={"error": str(e)}
                )
                raise ValueError("AI service authentication failed.")
            else:
                logger.error(f"AI completion failed: {e}")
                self.usage_metrics[user_id].errors += 1
                raise ValueError(f"AI completion failed: {str(e)}")
    
    def get_usage_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get usage metrics for a user."""
        metrics = self.usage_metrics[user_id]
        
        # Clean old request times
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        user_requests = self.request_times[user_id]
        while user_requests and user_requests[0] < hour_ago:
            user_requests.popleft()
        
        return {
            "requests_this_hour": len(user_requests),
            "total_tokens_used": metrics.tokens_used,
            "estimated_cost": metrics.cost_estimate,
            "error_count": metrics.errors,
            "last_request": metrics.last_request.isoformat() if metrics.last_request else None,
            "rate_limit": self.ai_rate_limit,
            "remaining_requests": max(0, self.ai_rate_limit - len(user_requests))
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check AI service health."""
        if not self.client:
            return {
                "status": "unavailable", 
                "reason": "API key not configured or OpenAI library not available"
            }
        
        return {
            "status": "available",
            "client_configured": True,
            "rate_limit": self.ai_rate_limit
        }

class AIOutputValidator:
    """Validator for AI-generated content."""
    
    @staticmethod
    def validate_test_case(test_case: Dict[str, Any]) -> bool:
        """Validate AI-generated test case."""
        required_fields = ['name', 'method', 'endpoint']
        
        for field in required_fields:
            if field not in test_case:
                return False
        
        # Validate method
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if test_case.get('method', '').upper() not in valid_methods:
            return False
        
        # Validate endpoint format
        endpoint = test_case.get('endpoint', '')
        if not endpoint.startswith('/') or len(endpoint) > 500:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = ['eval', 'exec', '__import__', 'subprocess', 'system']
        test_content = str(test_case).lower()
        for pattern in suspicious_patterns:
            if pattern in test_content:
                logger.warning(f"Suspicious pattern '{pattern}' found in AI-generated test case")
                return False
        
        return True
    
    @staticmethod
    def validate_api_analysis(analysis: Dict[str, Any]) -> bool:
        """Validate AI-generated API analysis."""
        if not isinstance(analysis, dict):
            return False
        
        # Check for reasonable structure
        expected_sections = ['summary', 'endpoints', 'recommendations']
        if not any(section in analysis for section in expected_sections):
            return False
        
        # Validate content length
        total_content = str(analysis)
        if len(total_content) > 50000:  # 50KB limit
            return False
        
        return True

# Global secure AI client
secure_ai_client = SecureAIClient()
