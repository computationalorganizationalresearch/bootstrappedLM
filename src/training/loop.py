from __future__ import annotations

from dataclasses import dataclass

from agent.model import PolicyValueModel
from common.config import AppConfig
from replay.buffer import ReplayBuffer


@dataclass
class TrainingComponents:
    model: PolicyValueModel
    replay: ReplayBuffer


def build_components(config: AppConfig) -> TrainingComponents:
    model = PolicyValueModel(obs_dim=4, hidden_dim=32, action_dim=4)
    replay = ReplayBuffer(capacity=config.replay_capacity)
    return TrainingComponents(model=model, replay=replay)


def run_training_loop(config: AppConfig, components: TrainingComponents) -> None:
    if config.dry_run or config.train_steps <= 0:
        print("dry-run: initialized components only")
        return
    print(f"training for {config.train_steps} steps (not implemented)")
