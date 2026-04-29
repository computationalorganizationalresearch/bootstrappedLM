from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CurriculumDecision:
    """Metadata describing a curriculum action for a task."""

    step_index: int
    env_id: str
    task_id: str
    action: str
    reason: str
    success_rate: float
    novelty_score: float
    old_difficulty: float
    new_difficulty: float
    old_seed: int
    new_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CurriculumOutcome:
    """Execution outcome for a sampled task."""

    step_index: int
    env_id: str
    task_id: str
    success: bool
    reward: float
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class CurriculumHistory:
    """Append-only store for curriculum decisions and outcomes."""

    def __init__(self) -> None:
        self.decisions: list[CurriculumDecision] = []
        self.outcomes: list[CurriculumOutcome] = []
        self._outcomes_by_task: dict[tuple[str, str], list[CurriculumOutcome]] = defaultdict(list)

    def record_decision(self, decision: CurriculumDecision) -> None:
        self.decisions.append(decision)

    def record_outcome(self, outcome: CurriculumOutcome) -> None:
        self.outcomes.append(outcome)
        self._outcomes_by_task[(outcome.env_id, outcome.task_id)].append(outcome)

    def task_outcomes(self, env_id: str, task_id: str) -> list[CurriculumOutcome]:
        return list(self._outcomes_by_task.get((env_id, task_id), []))

    def outcomes_for_env(self, env_id: str) -> list[CurriculumOutcome]:
        return [outcome for outcome in self.outcomes if outcome.env_id == env_id]
