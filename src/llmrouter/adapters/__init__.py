"""Provider adapter contracts and implementations."""

from llmrouter.adapters.base import ErrorClass, NormalizedRequest, NormalizedResponse, ProviderError
from llmrouter.adapters.openai_compat import OpenAICompatAdapter
from llmrouter.adapters.anthropic import AnthropicAdapter
from llmrouter.adapters.gemini import GeminiAdapter
from llmrouter.adapters.cloudflare import CloudflareAdapter

__all__ = [
    "AnthropicAdapter", "CloudflareAdapter", "ErrorClass", "GeminiAdapter",
    "NormalizedRequest", "NormalizedResponse", "OpenAICompatAdapter", "ProviderError",
]
