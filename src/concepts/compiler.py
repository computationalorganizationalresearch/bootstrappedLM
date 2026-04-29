from __future__ import annotations


class ConceptCompiler:
    def compile(self, observation: list[float]) -> list[int]:
        return [int(x > 0.5) for x in observation]
