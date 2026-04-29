from __future__ import annotations

import argparse

from common.config import load_config
from training.loop import build_components, run_training_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    components = build_components(config)
    print(f"initialized model={components.model.__class__.__name__} replay_capacity={components.replay.capacity}")
    run_training_loop(config, components)


if __name__ == "__main__":
    main()
