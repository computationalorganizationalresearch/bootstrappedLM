from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScalarMetric:
    name: str
    value: float
