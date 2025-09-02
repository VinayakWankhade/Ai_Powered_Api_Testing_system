from __future__ import annotations

import asyncio
from ..utils.logger import get_logger
from .base import BaseProvider
from ..gateway import LLMRequest, LLMResponse, ModelConfig, ModelProvider

logger = get_logger(__name__)


class AnthropicProvider(BaseProvider):
    """Stub Anthropic provider. Replace with real API integration later."""

    async def generate(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        await asyncio.sleep(model.latency_ms_avg / 1000)
        content = (
            f"Claude response from {model.model_name} providing comprehensive API testing analysis and recommendations."
        )
        return LLMResponse(
            request_id=request.request_id,
            content=content,
            model_used=model.model_name,
            provider_used=ModelProvider.ANTHROPIC,
            tokens_used=len(request.prompt.split()),
            cost_usd=0.0,
            latency_ms=0,
        )

