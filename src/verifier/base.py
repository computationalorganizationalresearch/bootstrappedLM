from __future__ import annotations

from dataclasses import asdict, dataclass, field

from envs.base import EpisodeTrace, JsonValue, TaskSpec


@dataclass(frozen=True)
class RewardBreakdown:
    success: float
    efficiency: float
    transfer: float
    compute_cost: float
    hack_penalty: float

    @property
    def total(self) -> float:
        return self.success + self.efficiency + self.transfer - self.compute_cost - self.hack_penalty


@dataclass(frozen=True)
class VerificationVerdict:
    correct: bool
    reward: RewardBreakdown
    audit_metadata: dict[str, JsonValue] = field(default_factory=dict)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload = asdict(self)
        payload["total_reward"] = self.reward.total
        return payload


def make_verdict(
    *,
    correct: bool,
    success: float,
    efficiency: float = 0.0,
    transfer: float = 0.0,
    compute_cost: float = 0.0,
    hack_penalty: float = 0.0,
    audit_metadata: dict[str, JsonValue] | None = None,
    failure_reason: str | None = None,
) -> VerificationVerdict:
    return VerificationVerdict(
        correct=correct,
        reward=RewardBreakdown(
            success=success,
            efficiency=efficiency,
            transfer=transfer,
            compute_cost=compute_cost,
            hack_penalty=hack_penalty,
        ),
        audit_metadata=audit_metadata or {},
        failure_reason=failure_reason,
    )


def verify_trace_basic(task: TaskSpec, trace: EpisodeTrace) -> VerificationVerdict:
    reached_budget = task.max_steps <= 0 or len(trace.transitions) <= task.max_steps
    terminated = bool(trace.transitions and trace.transitions[-1].done)
    correct = reached_budget and terminated

    efficiency = 0.0
    if correct and task.max_steps > 0:
        efficiency = max(0.0, 1.0 - (len(trace.transitions) / task.max_steps))

    return make_verdict(
        correct=correct,
        success=1.0 if correct else 0.0,
        efficiency=efficiency,
        transfer=float(trace.metadata.get("transfer_score", 0.0) or 0.0),
        compute_cost=float(trace.metadata.get("compute_cost", 0.0) or 0.0),
        hack_penalty=float(trace.metadata.get("hack_penalty", 0.0) or 0.0),
        failure_reason=None if correct else ("step_budget_exceeded" if not reached_budget else "episode_not_terminated"),
        audit_metadata={
            "task_id": task.task_id,
            "num_transitions": len(trace.transitions),
            "max_steps": task.max_steps,
            "terminated": terminated,
            "trace_metadata": trace.metadata,
        },
    )
