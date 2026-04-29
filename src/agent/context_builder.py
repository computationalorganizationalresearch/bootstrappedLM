from __future__ import annotations

from src.concepts.library import SymbolLibrary


class PolicyContextBuilder:
    """Builds policy input context from observations and accepted concept symbols."""

    def __init__(self, library: SymbolLibrary) -> None:
        self.library = library

    def build(self, obs: list[float]) -> list[float]:
        concept_features = self.library.feature_vector()
        return [*obs, *concept_features]
