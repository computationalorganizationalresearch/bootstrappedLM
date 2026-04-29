from __future__ import annotations

import random
from collections import deque

from common.types import Trajectory


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: deque[Trajectory] = deque(maxlen=capacity)

    def add(self, traj: Trajectory) -> None:
        self._data.append(traj)

    def sample(self, batch_size: int) -> list[Trajectory]:
        batch_size = min(batch_size, len(self._data))
        return random.sample(list(self._data), batch_size)

    def __len__(self) -> int:
        return len(self._data)
