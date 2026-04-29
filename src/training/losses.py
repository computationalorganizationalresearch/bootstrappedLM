from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LossTerms:
    policy: float
    value: float
    entropy_bonus: float
    transfer_bonus_proxy: float
    compression_proxy: float
    zipf_regularizer: float

    @property
    def total(self) -> float:
        return (
            self.policy
            + self.value
            - self.entropy_bonus
            - self.transfer_bonus_proxy
            - self.compression_proxy
            + self.zipf_regularizer
        )


def _safe_softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exp_vals = [math.exp(x - max_logit) for x in logits]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [x / denom for x in exp_vals]


def policy_loss(logits: list[float], chosen_action: int, advantage: float) -> float:
    probs = _safe_softmax(logits)
    if not probs:
        return 0.0
    idx = max(0, min(chosen_action, len(probs) - 1))
    chosen_prob = max(probs[idx], 1e-9)
    return -math.log(chosen_prob) * advantage


def value_loss(predicted_value: float, target_value: float) -> float:
    delta = predicted_value - target_value
    return delta * delta


def entropy_bonus(logits: list[float]) -> float:
    probs = _safe_softmax(logits)
    return -sum(p * math.log(max(p, 1e-9)) for p in probs)


def transfer_bonus_proxy(current_success: float, historical_success: float) -> float:
    return max(0.0, current_success - historical_success)


def compression_proxy(symbol_count: int, useful_symbol_count: int) -> float:
    if symbol_count <= 0:
        return 0.0
    usage_ratio = useful_symbol_count / symbol_count
    return max(0.0, 1.0 - usage_ratio)


def zipf_regularizer(useful_symbol_counts: list[int]) -> float:
    if not useful_symbol_counts:
        return 0.0
    counts = sorted((max(0, c) for c in useful_symbol_counts), reverse=True)
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    expected = [1.0 / rank for rank in range(1, len(probs) + 1)]
    z = sum(expected)
    expected = [x / z for x in expected]
    return sum((p - e) ** 2 for p, e in zip(probs, expected, strict=False))


def build_loss_terms(
    *,
    logits: list[float],
    chosen_action: int,
    advantage: float,
    predicted_value: float,
    target_value: float,
    current_success: float,
    historical_success: float,
    symbol_count: int,
    useful_symbol_count: int,
    useful_symbol_counts: list[int],
) -> LossTerms:
    return LossTerms(
        policy=policy_loss(logits=logits, chosen_action=chosen_action, advantage=advantage),
        value=value_loss(predicted_value=predicted_value, target_value=target_value),
        entropy_bonus=entropy_bonus(logits=logits),
        transfer_bonus_proxy=transfer_bonus_proxy(
            current_success=current_success,
            historical_success=historical_success,
        ),
        compression_proxy=compression_proxy(
            symbol_count=symbol_count,
            useful_symbol_count=useful_symbol_count,
        ),
        zipf_regularizer=zipf_regularizer(useful_symbol_counts=useful_symbol_counts),
    )
