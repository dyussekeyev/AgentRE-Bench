from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgentLoop
from .config import BenchmarkConfig
from .metrics import (
    AggregateMetrics,
    TaskMetrics,
    collect_task_metrics,
    compute_aggregate,
)
from .providers import create_provider
from .tools import ToolExecutor

log = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    task_id: str
    binary_path: Path
    ground_truth_path: Path
    difficulty: int


def load_tasks(manifest_path: Path, project_root: Path) -> list[TaskConfig]:
    with open(manifest_path) as f:
        data = json.load(f)

    tasks = []
    for entry in data["tasks"]:
        tasks.append(
            TaskConfig(
                task_id=entry["task_id"],
                binary_path=project_root / "binaries" / entry["binary_name"],
                ground_truth_path=project_root / entry["ground_truth"],
                difficulty=entry["difficulty"],
            )
        )
    return tasks


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert reverse engineer analyzing a binary executable.

Your task is to analyze the binary located at: {binary_path}

Use the available tools to examine the binary and determine:
1. **File type** (e.g. ELF64)
2. **Whether strings are encoded/encrypted** (true/false)
3. **The decoded C2 (command and control) address** — the IP:port or URL the binary connects to
4. **Techniques used** — specific techniques like socket_connect, xor_encoding, anti_debug_ptrace, etc.
5. **C2 protocol** — the communication protocol (TCP, HTTP, DNS, ICMP, etc.)

{bonus_instructions}

When you have completed your analysis, call the **final_answer** tool with your findings.
Be precise with technique names — only claim techniques you have evidence for.
Do not guess or hallucinate techniques you cannot confirm from the binary analysis.
"""

BONUS_INSTRUCTIONS = """\
This is an advanced sample. In addition to the standard fields, also determine:
- **Encryption details**: algorithm (e.g. RC4, AES), key, and how the key is stored
- **Decoded strings**: any encrypted/encoded strings you can recover
- **Anti-analysis techniques**: specific anti-debugging and anti-analysis methods

Provide these in the encryption_details, decoded_strings, and anti_analysis fields of your final_answer.
"""


def build_system_prompt(task: TaskConfig, config: BenchmarkConfig) -> str:
    if config.use_docker:
        binary_display = f"/workspace/{task.binary_path.name}"
    else:
        binary_display = str(task.binary_path)

    bonus = BONUS_INSTRUCTIONS if task.difficulty >= 13 else ""

    return SYSTEM_PROMPT_TEMPLATE.format(
        binary_path=binary_display,
        bonus_instructions=bonus,
    )


def run_single_task(
    task: TaskConfig,
    config: BenchmarkConfig,
) -> tuple[TaskMetrics, dict[str, Any]]:
    log.info("=" * 60)
    log.info("Task: %s (difficulty %d)", task.task_id, task.difficulty)
    log.info("=" * 60)

    # Validate binary exists
    if not task.binary_path.exists():
        log.error("Binary not found: %s", task.binary_path)
        raise FileNotFoundError(f"Binary not found: {task.binary_path}")

    # Load ground truth
    gt = json.loads(task.ground_truth_path.read_text())

    # Create tool executor
    tool_executor = ToolExecutor(config, task.binary_path)

    # Create provider
    api_key = config.resolve_api_key()
    provider = create_provider(config.provider, config.model, api_key)

    # Build system prompt
    system_prompt = build_system_prompt(task, config)

    # Run agent loop
    agent_loop = AgentLoop(
        provider=provider,
        tool_executor=tool_executor,
        system_prompt=system_prompt,
        task_id=task.task_id,
        max_tool_calls=config.max_tool_calls,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
    )
    agent_result = agent_loop.run()

    # Save agent output
    config.agent_outputs_dir.mkdir(parents=True, exist_ok=True)
    agent_output_path = config.agent_outputs_dir / f"{task.task_id}.json"

    final_answer = agent_result.get("final_answer") or {}
    with open(agent_output_path, "w") as f:
        json.dump(final_answer, f, indent=2)

    # Score using the existing scorer
    sys.path.insert(0, str(config.project_root))
    from scorer import score_sample

    score_result = score_sample(gt, final_answer, str(task.ground_truth_path))
    score_result["sample"] = task.task_id

    log.info(
        "Score for %s: %.4f (tier: %s)",
        task.task_id,
        score_result["final_score"],
        score_result["tier"],
    )

    # Collect metrics
    metrics = collect_task_metrics(task.task_id, agent_result, score_result)

    # Save transcript
    config.transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = config.transcripts_dir / f"{task.task_id}_transcript.json"
    transcript_data = {
        "task_id": task.task_id,
        "model": config.model,
        "provider": config.provider,
        "difficulty": task.difficulty,
        "score": score_result,
        "agent_result": {
            k: v
            for k, v in agent_result.items()
            if k != "transcript"  # transcript can be huge; save separately if needed
        },
        "metrics": metrics.to_dict(),
    }
    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=2, default=str)

    # Save full transcript separately
    full_transcript_path = config.transcripts_dir / f"{task.task_id}_full_transcript.json"
    with open(full_transcript_path, "w") as f:
        json.dump(agent_result.get("transcript", []), f, indent=2, default=str)

    return metrics, score_result


def run_benchmark(
    config: BenchmarkConfig,
    task_filter: str | None = None,
) -> tuple[AggregateMetrics, list[TaskMetrics], list[dict]]:
    manifest_path = config.project_root / "tasks.json"
    tasks = load_tasks(manifest_path, config.project_root)

    if task_filter:
        tasks = [t for t in tasks if t.task_id == task_filter]
        if not tasks:
            log.error("No task found matching %r", task_filter)
            raise ValueError(f"No task found matching {task_filter!r}")

    log.info("Running %d task(s) with %s/%s", len(tasks), config.provider, config.model)

    all_metrics: list[TaskMetrics] = []
    all_scores: list[dict] = []

    for task in tasks:
        try:
            metrics, score_result = run_single_task(task, config)
            all_metrics.append(metrics)
            all_scores.append(score_result)
        except Exception as e:
            log.error("Task %s failed: %s", task.task_id, e, exc_info=True)
            continue

    # Compute aggregate metrics
    aggregate = compute_aggregate(all_metrics)

    # Print summary via scorer
    sys.path.insert(0, str(config.project_root))
    from scorer import print_summary

    print_summary(all_scores)

    # Save benchmark report
    config.results_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.results_dir / "benchmark_report.json"
    report = {
        "config": {
            "model": config.model,
            "provider": config.provider,
            "max_tool_calls": config.max_tool_calls,
            "use_docker": config.use_docker,
        },
        "aggregate_metrics": aggregate.to_dict(),
        "task_metrics": [m.to_dict() for m in all_metrics],
        "score_results": all_scores,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Report saved to %s", report_path)

    return aggregate, all_metrics, all_scores
