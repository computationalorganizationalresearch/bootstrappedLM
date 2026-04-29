from __future__ import annotations

from dataclasses import dataclass

from agent.model import PolicyValueModel
from common.config import AppConfig
from common.types import Step, Task, Trajectory, VerificationResult
from replay.buffer import ReplayBuffer
from training.eval import evaluate_dry_run
from training.losses import LossTerms, build_loss_terms
from training.scheduler import LossCoefficients, PhaseScheduler


@dataclass
class TrainingComponents:
    model: PolicyValueModel
    replay: ReplayBuffer


def build_components(config: AppConfig) -> TrainingComponents:
    model = PolicyValueModel(obs_dim=4, hidden_dim=32, action_dim=4)
    replay = ReplayBuffer(capacity=config.replay_capacity)
    return TrainingComponents(model=model, replay=replay)


def _collect_rollout(step_idx: int) -> Trajectory:
    task = Task(
        id=f"task-{step_idx}",
        env_id="dry-env",
        seed=step_idx,
        difficulty=0.5,
        initial_state=[0.0, 0.0, 0.0, 0.0],
        goal_spec={"target": 1.0},
    )
    rollout = Trajectory(task=task)
    for t in range(4):
        obs = [float(t), 0.0, 0.0, 0.0]
        next_obs = [float(t + 1), 0.0, 0.0, 0.0]
        rollout.steps.append(Step(obs=obs, action=t % 4, reward=0.0, next_obs=next_obs, done=t == 3))
    return rollout


def _verify_outcome(rollout: Trajectory) -> VerificationResult:
    success = len(rollout.steps) >= 4
    return VerificationResult(success=success, score=1.0 if success else 0.0, failure_reason=None if success else "short")


def _compute_rewards(rollout: Trajectory, verification: VerificationResult) -> None:
    terminal_reward = verification.score
    for s in rollout.steps:
        s.reward = terminal_reward / max(1, len(rollout.steps))


def _optimizer_step(model: PolicyValueModel, minibatch: list[Trajectory], coeffs: LossCoefficients) -> dict[str, float]:
    aggregate = LossTerms(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for traj in minibatch:
        first = traj.steps[0]
        out = model.forward(first.obs)
        terms = build_loss_terms(
            logits=out.logits,
            chosen_action=first.action,
            advantage=first.reward,
            predicted_value=out.value,
            target_value=traj.verification.score if traj.verification else 0.0,
            current_success=1.0 if (traj.verification and traj.verification.success) else 0.0,
            historical_success=0.5,
            symbol_count=10,
            useful_symbol_count=7,
            useful_symbol_counts=[7, 4, 2, 1],
        )
        aggregate.policy += terms.policy
        aggregate.value += terms.value
        aggregate.entropy_bonus += terms.entropy_bonus
        aggregate.transfer_bonus_proxy += terms.transfer_bonus_proxy
        aggregate.compression_proxy += terms.compression_proxy
        aggregate.zipf_regularizer += terms.zipf_regularizer

    n = max(1, len(minibatch))
    log_terms = {
        "loss/policy": coeffs.policy * aggregate.policy / n,
        "loss/value": coeffs.value * aggregate.value / n,
        "bonus/entropy": coeffs.entropy_bonus * aggregate.entropy_bonus / n,
        "bonus/transfer_proxy": coeffs.transfer_bonus_proxy * aggregate.transfer_bonus_proxy / n,
        "bonus/compression_proxy": coeffs.compression_proxy * aggregate.compression_proxy / n,
        "regularizer/zipf": coeffs.zipf_regularizer * aggregate.zipf_regularizer / n,
    }
    log_terms["loss/total"] = (
        log_terms["loss/policy"]
        + log_terms["loss/value"]
        - log_terms["bonus/entropy"]
        - log_terms["bonus/transfer_proxy"]
        - log_terms["bonus/compression_proxy"]
        + log_terms["regularizer/zipf"]
    )
    return log_terms


def run_training_loop(config: AppConfig, components: TrainingComponents) -> None:
    if config.dry_run or config.train_steps <= 0:
        print("dry-run: initialized components only")
        return

    scheduler = PhaseScheduler(phase_1_steps=max(1, config.train_steps // 3), phase_2_steps=max(2, (2 * config.train_steps) // 3))

    for step_idx in range(config.train_steps):
        rollout = _collect_rollout(step_idx)
        verification = _verify_outcome(rollout)
        rollout.verification = verification
        _compute_rewards(rollout, verification)
        components.replay.add(rollout)

        minibatch = components.replay.sample(config.batch_size)
        coeffs = scheduler.coefficients(step_idx)
        logs = _optimizer_step(components.model, minibatch, coeffs)
        logs["phase"] = float(scheduler.phase_for_step(step_idx))
        logs["success"] = 1.0 if verification.success else 0.0
        print(f"step={step_idx} logs={logs}")

        if (step_idx + 1) % 10 == 0:
            held_out = evaluate_dry_run()
            print(f"held_out_eval@{step_idx + 1}: {held_out}")
