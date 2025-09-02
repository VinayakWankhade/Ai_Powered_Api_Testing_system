"""
Secure AI integration module with authentication, rate limiting, and output validation.
"""

import os
import time
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

import openai
from openai import OpenAI
import tiktoken

from ..utils.logger import get_logger, security_logger
from ..config.security_config import get_config

logger = get_logger(__name__)
config = get_config()

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
        self.api_key = config.openai_api_key
        if not self.api_key:
            logger.warning("OpenAI API key not configured. AI features will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # Rate limiting tracking
        self.usage_metrics = defaultdict(AIUsageMetrics)
        self.request_times = defaultdict(lambda: deque(maxlen=1000))
        
        # Token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
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
        hourly_limit = config.ai_rate_limit
        if len(user_requests) >= hourly_limit:
            security_logger.log_security_event(
                event_type="ai_rate_limit_exceeded",
                severity="warning",
                message=f"AI rate limit exceeded for user",
                user_id=user_id,
                details={"requests_in_hour": len(user_requests), "limit": hourly_limit}
            )
            return False
        
        return True
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            # Fallback estimation
            return len(text.split()) * 1.3
    
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
                import json
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
        """
        Generate AI completion with security controls.
        
        Args:
            prompt: The user prompt
            user_id: User identifier for rate limiting
            model: AI model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            expected_format: Expected output format for validation
            system_message: Optional system message
        
        Returns:
            Dict containing response and metadata
        """
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
                timeout=30  # 30 second timeout
            )
            
            duration = time.time() - start_time
            
            # Extract response content
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
            output_tokens = self._count_tokens(content)\n            cost = self._estimate_cost(model, input_tokens, output_tokens)\n            \n            # Update usage metrics\n            metrics = self.usage_metrics[user_id]\n            metrics.requests_per_hour += 1\n            metrics.tokens_used += input_tokens + output_tokens\n            metrics.cost_estimate += cost\n            metrics.last_request = now\n            \n            # Log successful completion\n            logger.info(\n                f\"AI completion successful\",\n                user_id=hashlib.sha256(user_id.encode()).hexdigest()[:12],\n                model=model,\n                input_tokens=input_tokens,\n                output_tokens=output_tokens,\n                cost=cost,\n                duration=duration\n            )\n            \n            return {\n                "content": content,\n                "metadata": {\n                    "model": model,\n                    "input_tokens": input_tokens,\n                    "output_tokens": output_tokens,\n                    "cost_estimate": cost,\n                    "duration": duration,\n                    "timestamp": now.isoformat()\n                }\n            }\n            \n        except openai.RateLimitError as e:\n            logger.warning(f\"OpenAI rate limit exceeded: {e}\")\n            raise ValueError("AI service rate limit exceeded. Please try again later.")\n        \n        except openai.AuthenticationError as e:\n            security_logger.log_security_event(\n                event_type="ai_authentication_failed",\n                severity="error",\n                message="OpenAI authentication failed",\n                details={"error": str(e)}\n            )\n            raise ValueError("AI service authentication failed.")\n        \n        except Exception as e:\n            logger.error(f\"AI completion failed: {e}\")\n            self.usage_metrics[user_id].errors += 1\n            raise ValueError(f"AI completion failed: {str(e)}")\n    \n    def _sanitize_prompt(self, prompt: str) -> str:\n        """Sanitize AI prompt to prevent injection attacks."""\n        if not prompt:\n            return ""\n        \n        # Remove potential command injection patterns\n        dangerous_patterns = [\n            r'`[^`]*`',  # Backticks\n            r'\\$\\([^)]*\\)',  # Command substitution\n            r'\\${[^}]*}',  # Variable substitution\n            r'exec\\s*\\(',  # exec calls\n            r'eval\\s*\\(',  # eval calls\n            r'import\\s+os',  # os imports\n            r'__import__',  # import function\n        ]\n        \n        import re\n        for pattern in dangerous_patterns:\n            prompt = re.sub(pattern, '[FILTERED]', prompt, flags=re.IGNORECASE)\n        \n        # Truncate if too long\n        if len(prompt) > 10000:\n            prompt = prompt[:10000] + \"... [truncated for security]\"\n        \n        return prompt\n    \n    def get_usage_metrics(self, user_id: str) -> Dict[str, Any]:\n        """Get usage metrics for a user."""\n        metrics = self.usage_metrics[user_id]\n        \n        # Clean old request times\n        now = datetime.now(timezone.utc)\n        hour_ago = now - timedelta(hours=1)\n        user_requests = self.request_times[user_id]\n        while user_requests and user_requests[0] < hour_ago:\n            user_requests.popleft()\n        \n        return {\n            "requests_this_hour": len(user_requests),\n            "total_tokens_used": metrics.tokens_used,\n            "estimated_cost": metrics.cost_estimate,\n            "error_count": metrics.errors,\n            "last_request": metrics.last_request.isoformat() if metrics.last_request else None,\n            "rate_limit": config.ai_rate_limit,\n            "remaining_requests": max(0, config.ai_rate_limit - len(user_requests))\n        }\n    \n    def health_check(self) -> Dict[str, Any]:\n        """Check AI service health."""\n        if not self.client:\n            return {\n                "status": "unavailable",\n                "reason": "API key not configured"\n            }\n        \n        try:\n            # Simple test request\n            test_response = self.client.chat.completions.create(\n                model="gpt-3.5-turbo",\n                messages=[{"role": "user", "content": "Hello"}],\n                max_tokens=5,\n                timeout=10\n            )\n            \n            return {\n                "status": "healthy",\n                "model_available": True,\n                "response_time": "< 10s"\n            }\n            \n        except Exception as e:\n            logger.warning(f"AI health check failed: {e}")\n            return {\n                "status": "degraded",\n                "reason": str(e)\n            }\n\nclass AIOutputValidator:\n    """Validator for AI-generated content."""\n    \n    @staticmethod\n    def validate_test_case(test_case: Dict[str, Any]) -> bool:\n        \"\"\"Validate AI-generated test case.\"\"\"\n        required_fields = ['name', 'method', 'endpoint']\n        \n        for field in required_fields:\n            if field not in test_case:\n                return False\n        \n        # Validate method\n        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']\n        if test_case.get('method', '').upper() not in valid_methods:\n            return False\n        \n        # Validate endpoint format\n        endpoint = test_case.get('endpoint', '')\n        if not endpoint.startswith('/') or len(endpoint) > 500:\n            return False\n        \n        # Check for suspicious patterns\n        suspicious_patterns = ['eval', 'exec', '__import__', 'subprocess', 'system']\n        test_content = str(test_case).lower()\n        for pattern in suspicious_patterns:\n            if pattern in test_content:\n                logger.warning(f\"Suspicious pattern '{pattern}' found in AI-generated test case\")\n                return False\n        \n        return True\n    \n    @staticmethod\n    def validate_api_analysis(analysis: Dict[str, Any]) -> bool:\n        \"\"\"Validate AI-generated API analysis.\"\"\"\n        if not isinstance(analysis, dict):\n            return False\n        \n        # Check for reasonable structure\n        expected_sections = ['summary', 'endpoints', 'recommendations']\n        if not any(section in analysis for section in expected_sections):\n            return False\n        \n        # Validate content length\n        total_content = str(analysis)\n        if len(total_content) > 50000:  # 50KB limit\n            return False\n        \n        return True\n    \n    @staticmethod\n    def sanitize_ai_output(content: str) -> str:\n        \"\"\"Sanitize AI output to remove potentially dangerous content.\"\"\"\n        if not content:\n            return \"\"\n        \n        # Remove potential code injection patterns\n        import re\n        \n        # Remove script tags\n        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)\n        \n        # Remove dangerous function calls\n        dangerous_functions = ['eval', 'exec', 'system', 'subprocess', '__import__']\n        for func in dangerous_functions:\n            content = re.sub(f'{func}\\s*\\(', f'{func}_FILTERED(', content, flags=re.IGNORECASE)\n        \n        return content\n\nclass AITaskManager:\n    \"\"\"Manage AI tasks with security and monitoring.\"\"\"\n    \n    def __init__(self):\n        self.ai_client = SecureAIClient()\n        self.validator = AIOutputValidator()\n        self.active_tasks = {}\n    \n    async def generate_test_cases(\n        self,\n        api_spec: Dict[str, Any],\n        user_id: str,\n        test_types: List[str] = None,\n        max_tests: int = 50\n    ) -> Dict[str, Any]:\n        \"\"\"Generate test cases with security controls.\"\"\"\n        test_types = test_types or [\"functional\"]\n        \n        # Validate inputs\n        if len(str(api_spec)) > 100000:  # 100KB limit\n            raise ValueError(\"API specification too large for processing\")\n        \n        if max_tests > 100:\n            raise ValueError(\"Maximum 100 test cases per request\")\n        \n        # Prepare system message\n        system_message = self._get_test_generation_system_message()\n        \n        # Prepare prompt\n        prompt = self._create_test_generation_prompt(api_spec, test_types, max_tests)\n        \n        # Generate with AI\n        result = await self.ai_client.generate_completion(\n            prompt=prompt,\n            user_id=user_id,\n            system_message=system_message,\n            expected_format=\"json\",\n            max_tokens=3000\n        )\n        \n        # Parse and validate results\n        try:\n            import json\n            test_cases = json.loads(result[\"content\"])\n            \n            # Validate each test case\n            validated_tests = []\n            for test_case in test_cases.get(\"test_cases\", []):\n                if self.validator.validate_test_case(test_case):\n                    validated_tests.append(test_case)\n                else:\n                    logger.warning(\"AI-generated test case failed validation\")\n            \n            return {\n                \"test_cases\": validated_tests,\n                \"generated_count\": len(test_cases.get(\"test_cases\", [])),\n                \"validated_count\": len(validated_tests),\n                \"metadata\": result[\"metadata\"]\n            }\n            \n        except json.JSONDecodeError as e:\n            logger.error(f\"Failed to parse AI-generated test cases: {e}\")\n            raise ValueError(\"AI generated invalid test case format\")\n    \n    async def analyze_api_spec(\n        self,\n        api_spec: Dict[str, Any],\n        user_id: str\n    ) -> Dict[str, Any]:\n        \"\"\"Analyze API specification with AI.\"\"\"\n        # Validate input size\n        if len(str(api_spec)) > 200000:  # 200KB limit\n            raise ValueError(\"API specification too large for analysis\")\n        \n        system_message = \"\"\"\n        You are an API security and quality analyst. Analyze the provided API specification and provide:\n        1. Security recommendations\n        2. Quality assessment\n        3. Coverage analysis\n        4. Potential issues\n        \n        Return your analysis in valid JSON format.\n        \"\"\"\n        \n        prompt = f\"\"\"\n        Please analyze this API specification:\n        \n        {json.dumps(api_spec, indent=2)}\n        \n        Provide analysis in JSON format with sections: summary, security_issues, quality_score, recommendations.\n        \"\"\"\n        \n        result = await self.ai_client.generate_completion(\n            prompt=prompt,\n            user_id=user_id,\n            system_message=system_message,\n            expected_format=\"json\",\n            max_tokens=2000\n        )\n        \n        # Validate and parse analysis\n        try:\n            import json\n            analysis = json.loads(result[\"content\"])\n            \n            if self.validator.validate_api_analysis(analysis):\n                return {\n                    \"analysis\": analysis,\n                    \"metadata\": result[\"metadata\"]\n                }\n            else:\n                raise ValueError(\"AI analysis failed validation\")\n                \n        except json.JSONDecodeError as e:\n            logger.error(f\"Failed to parse AI analysis: {e}\")\n            raise ValueError(\"AI generated invalid analysis format\")\n    \n    def _get_test_generation_system_message(self) -> str:\n        \"\"\"Get system message for test generation.\"\"\"\n        return \"\"\"\n        You are an expert API test engineer. Generate comprehensive test cases for the provided API specification.\n        \n        Guidelines:\n        - Generate realistic test scenarios\n        - Include positive and negative test cases\n        - Consider edge cases and boundary conditions\n        - Ensure all generated content is safe and valid\n        - Return results in valid JSON format\n        \n        Never include executable code, imports, or system commands in test cases.\n        \"\"\"\n    \n    def _create_test_generation_prompt(self, api_spec: Dict[str, Any], \n                                     test_types: List[str], max_tests: int) -> str:\n        \"\"\"Create prompt for test generation.\"\"\"\n        return f\"\"\"\n        Generate {max_tests} test cases for this API specification:\n        \n        Test Types: {', '.join(test_types)}\n        \n        API Specification:\n        {json.dumps(api_spec, indent=2)}\n        \n        Return JSON format:\n        {{\n            \"test_cases\": [\n                {{\n                    \"name\": \"test name\",\n                    \"description\": \"test description\",\n                    \"method\": \"HTTP_METHOD\",\n                    \"endpoint\": \"/path\",\n                    \"headers\": {{}},\n                    \"body\": {{}},\n                    \"expected_status\": 200,\n                    \"test_type\": \"functional\"\n                }}\n            ]\n        }}\n        \"\"\"\n    \n    def get_system_health(self) -> Dict[str, Any]:\n        \"\"\"Get AI system health status.\"\"\"\n        return {\n            \"ai_client\": self.ai_client.health_check(),\n            \"active_tasks\": len(self.active_tasks),\n            \"total_users\": len(self.ai_client.usage_metrics)\n        }\n\n# Global AI task manager\nai_task_manager = AITaskManager()
