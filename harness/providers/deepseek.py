from __future__ import annotations

from .openai_provider import OpenAIProvider

API_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAIProvider):
    def __init__(self, api_key: str, model: str, api_url: str | None = None, user_agent: str | None = None):
        super().__init__(api_key=api_key, model=model, api_url=api_url or API_URL, user_agent=user_agent)

    def _token_param(self) -> str:
        return "max_tokens"
