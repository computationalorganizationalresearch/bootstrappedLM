from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    seed: int = 0
    batch_size: int = 4
    replay_capacity: int = 1000
    train_steps: int = 0
    dry_run: bool = True


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    import yaml

    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return AppConfig(**{k: v for k, v in data.items() if hasattr(AppConfig, k)})
