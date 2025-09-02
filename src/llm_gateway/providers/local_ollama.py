from __future__ import annotations

import asyncio
from typing import Any, Dict

import httpx

from .base import BaseProvider, ProviderError
from ..gateway import LLMRequest, LLMResponse, ModelConfig, ModelProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LocalOllamaProvider(BaseProvider):
    """Local provider using Ollama's HTTP API."""

    async def generate(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        base_url = self.settings.get("base_url", "http://localhost:11434")
        timeout = request.timeout_seconds

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                endpoint = f"{base_url.rstrip('/')}/api/generate"
                payload: Dict[str, Any] = {
                    "model": model.model_name,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                    },
                }
                if request.max_tokens:
                    payload["options"]["num_predict"] = min(request.max_tokens, model.max_tokens)

                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("response") or data.get("message", "")

                input_tokens = len(request.prompt.split())
                return LLMResponse(
                    request_id=request.request_id,
                    content=content,
                    model_used=model.model_name,
                    provider_used=ModelProvider.LOCAL,
                    tokens_used=input_tokens,
                    cost_usd=0.0,
                    latency_ms=0,
                )
        except httpx.HTTPError as e:
            msg = (
                f"Local Ollama request failed: {str(e)}. "
                f"Ensure server is running at {base_url} and model '{model.model_name}' is available."
            )
            logger.error(msg)
            raise ProviderError(msg)
        except Exception as e:
            logger.error(f"Local provider error: {str(e)}")
            raise ProviderError(str(e))

