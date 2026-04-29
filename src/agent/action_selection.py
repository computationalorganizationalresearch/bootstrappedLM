from __future__ import annotations


def select_action(logits: list[float], action_mask: list[int]) -> int:
    for i, m in enumerate(action_mask):
        if m:
            return i
    return 0
