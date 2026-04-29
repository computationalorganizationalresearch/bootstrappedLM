from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    id: str
    env_id: str
    seed: int
    difficulty: float
    initial_state: Any
    goal_spec: dict[str, Any]
    max_steps: int = 32
    generator_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    success: bool
    score: float
    failure_reason: str | None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    obs: Any
    action: int
    reward: float
    next_obs: Any
    done: bool
    env_info: dict[str, Any] = field(default_factory=dict)
    agent_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task: Task
    steps: list[Step] = field(default_factory=list)
    verification: VerificationResult | None = None
