from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..gateway import LLMRequest, LLMResponse, ModelConfig


class ProviderError(Exception):
    """Raised when a provider call fails."""


class BaseProvider(ABC):
    """Abstract base interface for LLM providers."""

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}

    @abstractmethod
    async def generate(self, request: LLMRequest, model: ModelConfig) -> LLMResponse:
        """Generate a response for the given request using the specified model."""
        raise NotImplementedError

