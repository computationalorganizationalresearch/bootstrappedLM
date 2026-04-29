from __future__ import annotations


class SymbolLibrary:
    def __init__(self) -> None:
        self.symbols: dict[str, int] = {"near_goal": 0, "far_goal": 1}
