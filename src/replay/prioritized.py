from __future__ import annotations

from common.types import Trajectory


class PrioritySampler:
    def priority(self, traj: Trajectory) -> float:
        if traj.verification is None:
            return 0.5
        return 1.0 - traj.verification.score
