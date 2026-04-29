from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConceptSymbol:
    """Reusable concept represented as either macro-action or latent tag."""

    name: str
    symbol_id: int
    kind: str  # "macro_action" | "latent_tag"
    pattern: tuple[int, ...] = ()
    latent_signature: tuple[int, ...] = ()
    usage_count: int = 0
    successes: int = 0
    solve_step_delta: float = 0.0
    search_depth_delta: float = 0.0
    transfer_delta: float = 0.0
    ambiguity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def register_usage(self, successful: bool) -> None:
        self.usage_count += 1
        if successful:
            self.successes += 1

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.successes / self.usage_count


class SymbolLibrary:
    def __init__(self, ambiguity_threshold: float = 0.25, min_usage: int = 2) -> None:
        self.ambiguity_threshold = ambiguity_threshold
        self.min_usage = min_usage
        self.symbols: dict[str, ConceptSymbol] = {}

    def register(self, symbol: ConceptSymbol) -> None:
        self.symbols[symbol.name] = symbol

    def get(self, name: str) -> ConceptSymbol | None:
        return self.symbols.get(name)

    def record_outcome(self, name: str, successful: bool) -> None:
        symbol = self.symbols.get(name)
        if symbol is not None:
            symbol.register_usage(successful)

    def accepted_symbols(self) -> list[ConceptSymbol]:
        accepted: list[ConceptSymbol] = []
        for symbol in self.symbols.values():
            if symbol.usage_count < self.min_usage:
                continue
            if symbol.successes == 0:
                continue
            if symbol.ambiguity > self.ambiguity_threshold:
                continue
            helps_depth_or_steps = (symbol.search_depth_delta < 0) or (symbol.solve_step_delta < 0)
            if not helps_depth_or_steps:
                continue
            if symbol.transfer_delta <= 0:
                continue
            accepted.append(symbol)
        accepted.sort(key=lambda s: (s.transfer_delta, s.success_rate), reverse=True)
        return accepted

    def feature_vector(self) -> list[float]:
        accepted = self.accepted_symbols()
        if not accepted:
            return []
        features: list[float] = []
        for symbol in accepted:
            features.extend(
                [
                    float(symbol.symbol_id),
                    float(symbol.usage_count),
                    symbol.success_rate,
                    symbol.transfer_delta,
                    -symbol.ambiguity,
                ]
            )
        return features
