from .base import AgentProvider, ProviderResponse, ToolCall
from .anthropic import AnthropicProvider
from .openai_provider import OpenAIProvider
from .gemini import GeminiProvider
from .deepseek import DeepSeekProvider

PROVIDER_MAP = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
}


def create_provider(provider_name: str, model: str, api_key: str, api_url: str | None = None, user_agent: str | None = None) -> AgentProvider:
    cls = PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown provider {provider_name!r}. "
            f"Choose from: {', '.join(PROVIDER_MAP)}"
        )
    return cls(api_key=api_key, model=model, api_url=api_url, user_agent=user_agent)
