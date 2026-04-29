from __future__ import annotations


class AdaptiveCurriculum:
    def __init__(self, difficulty: float = 0.1) -> None:
        self.difficulty = difficulty

    def update(self, success_rate: float) -> None:
        if success_rate > 0.8:
            self.difficulty = min(1.0, self.difficulty + 0.05)
        elif success_rate < 0.3:
            self.difficulty = max(0.0, self.difficulty - 0.05)
