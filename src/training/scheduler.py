from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossCoefficients:
    policy: float = 1.0
    value: float = 1.0
    entropy_bonus: float = 0.0
    transfer_bonus_proxy: float = 0.0
    compression_proxy: float = 0.0
    zipf_regularizer: float = 0.0


class PhaseScheduler:
    """Three-phase MVP schedule for modular loss coefficients."""

    def __init__(self, phase_1_steps: int, phase_2_steps: int) -> None:
        self.phase_1_steps = max(0, phase_1_steps)
        self.phase_2_steps = max(self.phase_1_steps, phase_2_steps)

    def phase_for_step(self, step: int) -> int:
        if step < self.phase_1_steps:
            return 1
        if step < self.phase_2_steps:
            return 2
        return 3

    def coefficients(self, step: int) -> LossCoefficients:
        phase = self.phase_for_step(step)
        if phase == 1:
            return LossCoefficients(policy=1.0, value=0.5, entropy_bonus=0.01)
        if phase == 2:
            return LossCoefficients(
                policy=1.0,
                value=0.5,
                entropy_bonus=0.01,
                transfer_bonus_proxy=0.2,
                compression_proxy=0.1,
            )
        return LossCoefficients(
            policy=1.0,
            value=0.5,
            entropy_bonus=0.01,
            transfer_bonus_proxy=0.2,
            compression_proxy=0.1,
            zipf_regularizer=0.05,
        )
