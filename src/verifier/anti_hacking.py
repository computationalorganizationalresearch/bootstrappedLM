from __future__ import annotations


def anti_reward_hacking_flags(env_info: dict) -> dict[str, bool]:
    return {"suspicious_transition": bool(env_info.get("invalid", False))}
