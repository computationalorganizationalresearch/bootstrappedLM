from __future__ import annotations

from common.types import Trajectory, VerificationResult


def check_step_budget(trajectory: Trajectory, max_steps: int) -> VerificationResult:
    ok = len(trajectory.steps) <= max_steps
    return VerificationResult(success=ok, score=1.0 if ok else 0.0, failure_reason=None if ok else "step_budget_exceeded")
