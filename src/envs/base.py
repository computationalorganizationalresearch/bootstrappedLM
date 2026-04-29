from __future__ import annotations

from abc import ABC, abstractmethod

from common.types import Task, Trajectory, VerificationResult


class VerifiableEnvironment(ABC):
    @abstractmethod
    def reset(self, task: Task):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int):
        raise NotImplementedError

    @abstractmethod
    def available_actions(self):
        raise NotImplementedError

    @abstractmethod
    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult:
        raise NotImplementedError

    @abstractmethod
    def generate_task(self, difficulty: float, seed: int) -> Task:
        raise NotImplementedError
