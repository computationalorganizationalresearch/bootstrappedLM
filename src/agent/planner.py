from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlannerOutput:
    action: int
    info: dict[str, float]


class Planner:
    def plan(self, logits: list[float], action_mask: list[int]) -> PlannerOutput:
        legal = [i for i, m in enumerate(action_mask) if m]
        action = legal[0] if legal else 0
        return PlannerOutput(action=action, info={"strategy": 1.0})
