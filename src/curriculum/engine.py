from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from random import Random
from typing import Callable

from src.common.types import Task
from src.curriculum.history import CurriculumDecision, CurriculumHistory, CurriculumOutcome

TaskMutator = Callable[[Task, Random], Task]
TaskFactory = Callable[[str, float, int, Random], Task]


@dataclass(slots=True)
class CurriculumConfig:
    window_size: int = 50
    target_success_low: float = 0.10
    target_success_high: float = 0.40
    unsolved_patience: int = 8
    solved_streak_to_mutate: int = 2
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    difficulty_step: float = 0.05
    novelty_min_score: float = 0.10
    novelty_window: int = 200


@dataclass(slots=True)
class _TaskStats:
    results: deque[bool]
    solved_streak: int = 0
    unsolved_streak: int = 0


class CurriculumEngine:
    """Per-environment curriculum controller with rolling success-rate windows."""

    def __init__(
        self,
        *,
        config: CurriculumConfig | None = None,
        seed: int = 0,
        history: CurriculumHistory | None = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self._rng = Random(seed)
        self._history = history or CurriculumHistory()
        self._step_index = 0

        self._env_tasks: dict[str, dict[str, Task]] = defaultdict(dict)
        self._env_windows: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=self.config.window_size))
        self._task_stats: dict[tuple[str, str], _TaskStats] = {}
        self._novelty_windows: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=self.config.novelty_window)
        )

    @property
    def history(self) -> CurriculumHistory:
        return self._history

    def register_task(self, task: Task) -> None:
        self._env_tasks[task.env_id][task.id] = task
        self._task_stats[(task.env_id, task.id)] = _TaskStats(results=deque(maxlen=self.config.window_size))

    def record_outcome(self, outcome: CurriculumOutcome) -> None:
        self._history.record_outcome(outcome)
        self._step_index = max(self._step_index, outcome.step_index + 1)

        key = (outcome.env_id, outcome.task_id)
        if key not in self._task_stats:
            self._task_stats[key] = _TaskStats(results=deque(maxlen=self.config.window_size))

        stats = self._task_stats[key]
        stats.results.append(outcome.success)
        if outcome.success:
            stats.solved_streak += 1
            stats.unsolved_streak = 0
        else:
            stats.unsolved_streak += 1
            stats.solved_streak = 0

        self._env_windows[outcome.env_id].append(outcome.success)

        task = self._env_tasks.get(outcome.env_id, {}).get(outcome.task_id)
        if task is not None:
            self._novelty_windows[outcome.env_id].append((task.difficulty, task.seed))

    def adjust_env(
        self,
        env_id: str,
        *,
        task_mutator: TaskMutator,
        task_factory: TaskFactory,
    ) -> list[CurriculumDecision]:
        decisions: list[CurriculumDecision] = []
        tasks = self._env_tasks[env_id]
        success_rate = self._success_rate(env_id)

        for task_id, task in list(tasks.items()):
            stats = self._task_stats[(env_id, task_id)]
            novelty = self._novelty_score(env_id, task)
            action = "keep"
            reason = "within_target_band"
            new_task = task

            if stats.solved_streak >= self.config.solved_streak_to_mutate:
                new_task = task_mutator(task, self._rng)
                new_task.difficulty = min(self.config.max_difficulty, task.difficulty + self.config.difficulty_step)
                action = "mutate_harder"
                reason = "solved_streak"
                stats.solved_streak = 0
            elif stats.unsolved_streak >= self.config.unsolved_patience:
                replacement = task_factory(
                    env_id,
                    max(self.config.min_difficulty, task.difficulty - self.config.difficulty_step),
                    self._next_seed(),
                    self._rng,
                )
                tasks.pop(task_id, None)
                self._task_stats.pop((env_id, task_id), None)
                tasks[replacement.id] = replacement
                self._task_stats[(env_id, replacement.id)] = _TaskStats(
                    results=deque(maxlen=self.config.window_size)
                )
                action = "drop_unsolved"
                reason = "unsolved_patience_exceeded"
                new_task = replacement
            elif novelty < self.config.novelty_min_score:
                replacement = task_factory(env_id, task.difficulty, self._next_seed(), self._rng)
                tasks[replacement.id] = replacement
                tasks.pop(task_id, None)
                self._task_stats.pop((env_id, task_id), None)
                self._task_stats[(env_id, replacement.id)] = _TaskStats(
                    results=deque(maxlen=self.config.window_size)
                )
                action = "drop_trivial"
                reason = "low_novelty"
                new_task = replacement
            elif success_rate > self.config.target_success_high:
                new_task = task_mutator(task, self._rng)
                new_task.difficulty = min(self.config.max_difficulty, task.difficulty + self.config.difficulty_step)
                action = "mutate_harder"
                reason = "success_rate_too_high"
            elif success_rate < self.config.target_success_low:
                new_task = task
                new_task.difficulty = max(self.config.min_difficulty, task.difficulty - self.config.difficulty_step)
                new_task.seed = self._next_seed()
                action = "ease_task"
                reason = "success_rate_too_low"

            tasks[new_task.id] = new_task
            if new_task.id != task_id:
                tasks.pop(task_id, None)

            decision = CurriculumDecision(
                step_index=self._step_index,
                env_id=env_id,
                task_id=task_id,
                action=action,
                reason=reason,
                success_rate=success_rate,
                novelty_score=novelty,
                old_difficulty=task.difficulty,
                new_difficulty=new_task.difficulty,
                old_seed=task.seed,
                new_seed=new_task.seed,
            )
            self._history.record_decision(decision)
            decisions.append(decision)

        self._step_index += 1
        return decisions

    def _success_rate(self, env_id: str) -> float:
        window = self._env_windows[env_id]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def _next_seed(self) -> int:
        return self._rng.randint(0, 2**31 - 1)

    def _novelty_score(self, env_id: str, task: Task) -> float:
        recent = self._novelty_windows[env_id]
        if not recent:
            return 1.0

        overlap = sum(1 for diff, seed in recent if abs(diff - task.difficulty) < 1e-6 and seed == task.seed)
        return 1.0 - (overlap / len(recent))
