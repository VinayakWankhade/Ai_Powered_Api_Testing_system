from __future__ import annotations

import asyncio
from ..utils.logger import get_logger
from .base import BaseProvider
from ..gateway import LLMRequest, LLMResponse, ModelConfig, ModelProvider

logger = get_logger(__name__)


class OpenAIProvider(BaseProvider):
    """Stub OpenAI provider. Replace with real API integration later."""

    async def generate(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        await asyncio.sleep(model.latency_ms_avg / 1000)
        content = (
            f"Generated response from {model.model_name} for API testing task. "
            f"This is a detailed analysis and implementation based on the provided prompt."
        )
        return LLMResponse(
            request_id=request.request_id,
            content=content,
            model_used=model.model_name,
            provider_used=ModelProvider.OPENAI,
            tokens_used=len(request.prompt.split()),
            cost_usd=0.0,
            latency_ms=0,
        )

