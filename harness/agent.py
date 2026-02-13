from __future__ import annotations

import json
import logging
import time
from typing import Any

from .providers.base import AgentProvider, ProviderResponse
from .tools import ToolExecutor, get_tool_schemas

log = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        provider: AgentProvider,
        tool_executor: ToolExecutor,
        system_prompt: str,
        task_id: str,
        max_tool_calls: int = 25,
        max_tokens: int = 4096,
        verbose: bool = False,
    ):
        self.provider = provider
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.task_id = task_id
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.verbose = verbose

        self.messages: list[dict] = []
        self.tool_call_count = 0
        self.tool_calls_log: list[dict] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.invalid_tool_calls = 0
        self.invalid_json_attempts = 0

    def _vprint(self, *args, **kwargs):
        """Print to stdout only when verbose mode is on."""
        if self.verbose:
            print(*args, **kwargs, flush=True)

    def run(self) -> dict[str, Any]:
        start_time = time.time()
        tools = get_tool_schemas(include_final_answer=True)

        # Initial user message
        self.messages.append({
            "role": "user",
            "content": (
                "Analyze the binary file in the workspace and submit your findings "
                "using the final_answer tool. The binary is located at the path "
                "shown in the system prompt. Use the available RE tools to examine it."
            ),
        })

        final_answer = None
        max_steps_hit = False

        self._vprint(f"\n{'='*70}")
        self._vprint(f"  AGENT LOOP START — {self.task_id}")
        self._vprint(f"{'='*70}")

        while self.tool_call_count < self.max_tool_calls:
            try:
                response = self.provider.create_message(
                    system=self.system_prompt,
                    messages=self.messages,
                    tools=tools,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                log.error("[%s] Provider error: %s", self.task_id, e)
                self._vprint(f"\n  !! Provider error: {e}")
                break

            self.input_tokens += response.input_tokens
            self.output_tokens += response.output_tokens

            self._vprint(
                f"\n  [tokens: +{response.input_tokens} in / "
                f"+{response.output_tokens} out | "
                f"stop: {response.stop_reason}]"
            )

            if response.stop_reason == "tool_use" and response.tool_calls:
                # Show agent reasoning
                if response.text_content:
                    self._vprint(f"\n  AGENT REASONING:")
                    for line in response.text_content.splitlines():
                        self._vprint(f"    {line}")

                # Build assistant content blocks
                assistant_content = []
                if response.text_content:
                    assistant_content.append({
                        "type": "text",
                        "text": response.text_content,
                    })

                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })

                self.messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool call
                tool_results = []
                for tc in response.tool_calls:
                    self.tool_call_count += 1
                    self.tool_calls_log.append({
                        "call_number": self.tool_call_count,
                        "tool": tc.name,
                        "input": tc.input,
                    })

                    log.info(
                        "[%s] Tool call #%d: %s(%s)",
                        self.task_id, self.tool_call_count, tc.name,
                        json.dumps(tc.input, default=str)[:200],
                    )

                    self._vprint(
                        f"\n  TOOL #{self.tool_call_count}: {tc.name}"
                        f"({json.dumps(tc.input, default=str)})"
                    )

                    result = self.tool_executor.execute(tc.name, tc.input)

                    if result.get("is_final_answer"):
                        final_answer = result["answer"]
                        self.tool_calls_log[-1]["is_final_answer"] = True
                        log.info("[%s] Final answer received.", self.task_id)
                        self._vprint(f"\n  FINAL ANSWER SUBMITTED:")
                        self._vprint(
                            f"    {json.dumps(final_answer, indent=2, default=str)}"
                        )
                        break

                    if result.get("error"):
                        self.invalid_tool_calls += 1
                        output_text = f"Error: {result['error']}"
                    else:
                        output_text = result.get("output", "(no output)")

                    self.tool_calls_log[-1]["output_preview"] = output_text[:500]

                    # Show tool output (truncated for readability)
                    self._vprint(f"  OUTPUT ({len(output_text)} chars):")
                    preview = output_text[:2000]
                    for line in preview.splitlines()[:40]:
                        self._vprint(f"    {line}")
                    if len(output_text) > 2000 or len(output_text.splitlines()) > 40:
                        self._vprint(f"    ... [truncated in verbose view]")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": output_text,
                    })

                if final_answer is not None:
                    break

                if tool_results:
                    self.messages.append({"role": "user", "content": tool_results})

                    # Budget warning when running low on tool calls
                    remaining = self.max_tool_calls - self.tool_call_count
                    if remaining == 5:
                        self.messages.append({
                            "role": "user",
                            "content": (
                                "IMPORTANT: You have only 5 tool calls remaining. "
                                "Start wrapping up your analysis and submit your "
                                "findings using the final_answer tool soon. "
                                "Submit your best answer with what you've found so far "
                                "rather than running out of tool calls."
                            ),
                        })
                        self._vprint(f"\n  ** Budget warning injected (5 calls left) **")
                    elif remaining == 2:
                        self.messages.append({
                            "role": "user",
                            "content": (
                                "CRITICAL: You have only 2 tool calls left. "
                                "You MUST call the final_answer tool NOW with your "
                                "best analysis. Do not use any more investigation tools."
                            ),
                        })
                        self._vprint(f"\n  ** Critical budget warning (2 calls left) **")

            elif response.stop_reason == "end_turn":
                if response.text_content:
                    self._vprint(f"\n  AGENT TEXT (end_turn, no tool call):")
                    for line in response.text_content.splitlines():
                        self._vprint(f"    {line}")

                # Agent stopped without calling a tool — try to extract JSON
                extracted = self._try_extract_json(response.text_content)
                if extracted is not None:
                    final_answer = extracted
                    log.info("[%s] Extracted final answer from text.", self.task_id)
                    self._vprint(f"\n  (extracted JSON answer from text)")
                    break

                # Prompt agent to use final_answer tool
                self.invalid_json_attempts += 1
                self._vprint(f"  (nudging agent to use final_answer tool)")
                self.messages.append({
                    "role": "assistant",
                    "content": response.text_content,
                })
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Please submit your analysis using the final_answer tool. "
                        "Do not respond with plain text — you must call the "
                        "final_answer tool with your findings."
                    ),
                })

            elif response.stop_reason == "max_tokens":
                log.warning("[%s] Hit max_tokens.", self.task_id)
                self._vprint(f"\n  !! Hit max_tokens — continuing")
                # Add what we got and continue
                if response.text_content:
                    self.messages.append({
                        "role": "assistant",
                        "content": response.text_content,
                    })
                    self.messages.append({
                        "role": "user",
                        "content": "Please continue your analysis and submit via final_answer tool.",
                    })
            else:
                log.warning("[%s] Unexpected stop_reason: %s", self.task_id, response.stop_reason)
                self._vprint(f"\n  !! Unexpected stop: {response.stop_reason}")
                break

        else:
            max_steps_hit = True
            log.warning("[%s] Hit max tool calls limit (%d).", self.task_id, self.max_tool_calls)
            self._vprint(f"\n  !! Hit max tool calls limit ({self.max_tool_calls})")

        self._vprint(f"\n{'='*70}")
        self._vprint(f"  AGENT LOOP END — {self.task_id}")
        self._vprint(
            f"  {self.tool_call_count} tool calls | "
            f"{self.input_tokens} in + {self.output_tokens} out tokens | "
            f"{time.time() - start_time:.1f}s"
        )
        self._vprint(f"{'='*70}\n")

        wall_time = time.time() - start_time

        # Compute tool usage stats
        tool_calls_by_type: dict[str, int] = {}
        seen_calls: set[str] = set()
        redundant_tool_calls = 0
        for entry in self.tool_calls_log:
            name = entry["tool"]
            tool_calls_by_type[name] = tool_calls_by_type.get(name, 0) + 1
            call_key = f"{name}:{json.dumps(entry['input'], sort_keys=True, default=str)}"
            if call_key in seen_calls:
                redundant_tool_calls += 1
            seen_calls.add(call_key)

        return {
            "task_id": self.task_id,
            "final_answer": final_answer,
            "transcript": self.messages,
            "tool_call_count": self.tool_call_count,
            "tool_calls_by_type": tool_calls_by_type,
            "tool_calls_log": self.tool_calls_log,
            "redundant_tool_calls": redundant_tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "invalid_json_attempts": self.invalid_json_attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "wall_time_seconds": round(wall_time, 2),
            "max_steps_hit": max_steps_hit,
            "has_valid_answer": final_answer is not None,
        }

    def _try_extract_json(self, text: str) -> dict | None:
        if not text:
            return None
        # Look for JSON blocks in text
        import re
        patterns = [
            r"```json\s*(.*?)```",
            r"```\s*(.*?)```",
            r"\{[^{}]*\"file_type\"[^{}]*\}",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                try:
                    data = json.loads(match.group(1) if match.lastindex else match.group(0))
                    if isinstance(data, dict) and "file_type" in data:
                        return data
                except (json.JSONDecodeError, IndexError):
                    continue
        return None
