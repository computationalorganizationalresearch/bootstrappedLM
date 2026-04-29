from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentOutput:
    logits: list[float]
    value: float


class PolicyValueModel:
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def forward(self, obs: list[float]) -> AgentOutput:
        logits = [0.0 for _ in range(self.action_dim)]
        value = sum(obs) / max(1, len(obs))
        return AgentOutput(logits=logits, value=value)
