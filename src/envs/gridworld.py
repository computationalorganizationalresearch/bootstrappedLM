from __future__ import annotations

import random

from common.types import Task, Trajectory, VerificationResult
from envs.base import VerifiableEnvironment


class TinyGridWorld(VerifiableEnvironment):
    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.pos = (0, 0)
        self.goal = (size - 1, size - 1)

    def reset(self, task: Task):
        self.goal = tuple(task.goal_spec["goal"])
        self.pos = tuple(task.initial_state["pos"])
        return self._obs()

    def _obs(self):
        x, y = self.pos
        gx, gy = self.goal
        return [x / self.size, y / self.size, gx / self.size, gy / self.size]

    def step(self, action: int):
        x, y = self.pos
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.size - 1, y + 1)
        self.pos = (x, y)
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self._obs(), reward, done, {}

    def available_actions(self):
        return [1, 1, 1, 1]

    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        success = self.pos == self.goal
        return VerificationResult(success=success, score=1.0 if success else 0.0, failure_reason=None if success else "goal_not_reached")

    def generate_task(self, difficulty: float, seed: int) -> Task:
        rng = random.Random(seed)
        goal = (rng.randrange(self.size), rng.randrange(self.size))
        return Task(id=f"grid-{seed}", env_id="gridworld", seed=seed, difficulty=difficulty, initial_state={"pos": (0, 0)}, goal_spec={"goal": goal})
