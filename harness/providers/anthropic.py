from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .base import AgentProvider, ProviderResponse, ToolCall

log = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(AgentProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
        }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            log.error("Anthropic API error %d: %s", e.code, error_body)
            raise

        text_parts = []
        tool_calls = []

        for block in result.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        input=block["input"],
                    )
                )

        usage = result.get("usage", {})

        return ProviderResponse(
            stop_reason=result.get("stop_reason", "end_turn"),
            text_content="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
