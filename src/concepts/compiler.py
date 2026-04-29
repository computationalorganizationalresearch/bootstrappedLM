from __future__ import annotations

from collections import Counter, defaultdict

from src.common.types import Trajectory
from src.concepts.library import ConceptSymbol, SymbolLibrary


class ConceptCompiler:
    """Mines recurring sub-trajectories and action-state motifs from replay."""

    def __init__(self, library: SymbolLibrary, min_support: int = 2, max_pattern_len: int = 4) -> None:
        self.library = library
        self.min_support = min_support
        self.max_pattern_len = max_pattern_len

    def compile(
        self,
        trajectories: list[Trajectory],
        validation_metrics: dict[str, dict[str, float]],
        transfer_scores: dict[str, float],
    ) -> list[ConceptSymbol]:
        patterns = self._mine_action_patterns(trajectories)
        motifs = self._mine_state_action_motifs(trajectories)

        created: list[ConceptSymbol] = []
        for idx, (pattern, support) in enumerate(patterns.items()):
            symbol = self._build_macro_action_symbol(idx, pattern, support, validation_metrics, transfer_scores)
            self.library.register(symbol)
            created.append(symbol)

        offset = len(created)
        for idx, (motif, support) in enumerate(motifs.items(), start=offset):
            symbol = self._build_latent_tag_symbol(idx, motif, support, validation_metrics, transfer_scores)
            self.library.register(symbol)
            created.append(symbol)

        return self.library.accepted_symbols()

    def _mine_action_patterns(self, trajectories: list[Trajectory]) -> Counter[tuple[int, ...]]:
        counts: Counter[tuple[int, ...]] = Counter()
        for trajectory in trajectories:
            actions = [step.action for step in trajectory.steps]
            for window in range(2, self.max_pattern_len + 1):
                for start in range(0, max(0, len(actions) - window + 1)):
                    counts[tuple(actions[start : start + window])] += 1
        return Counter({k: v for k, v in counts.items() if v >= self.min_support})

    def _mine_state_action_motifs(self, trajectories: list[Trajectory]) -> Counter[tuple[int, ...]]:
        counts: Counter[tuple[int, ...]] = Counter()
        for trajectory in trajectories:
            for step in trajectory.steps:
                obs_bins = tuple(int(float(x) > 0.5) for x in (step.obs[:3] if isinstance(step.obs, list) else []))
                motif = (step.action, *obs_bins)
                counts[motif] += 1
        return Counter({k: v for k, v in counts.items() if v >= self.min_support})

    def _build_macro_action_symbol(
        self,
        symbol_id: int,
        pattern: tuple[int, ...],
        support: int,
        validation_metrics: dict[str, dict[str, float]],
        transfer_scores: dict[str, float],
    ) -> ConceptSymbol:
        deltas = self._aggregate_deltas(validation_metrics)
        transfer_delta = self._aggregate_transfer(transfer_scores)
        ambiguity = self._estimate_ambiguity(pattern)
        return ConceptSymbol(
            name=f"macro_{'_'.join(map(str, pattern))}",
            symbol_id=symbol_id,
            kind="macro_action",
            pattern=pattern,
            usage_count=support,
            successes=max(1, int(support * 0.6)),
            solve_step_delta=deltas["solve_steps"],
            search_depth_delta=deltas["search_depth"],
            transfer_delta=transfer_delta,
            ambiguity=ambiguity,
            metadata={"support": support},
        )

    def _build_latent_tag_symbol(
        self,
        symbol_id: int,
        motif: tuple[int, ...],
        support: int,
        validation_metrics: dict[str, dict[str, float]],
        transfer_scores: dict[str, float],
    ) -> ConceptSymbol:
        deltas = self._aggregate_deltas(validation_metrics)
        transfer_delta = self._aggregate_transfer(transfer_scores)
        ambiguity = self._estimate_ambiguity(motif)
        return ConceptSymbol(
            name=f"latent_{'_'.join(map(str, motif))}",
            symbol_id=symbol_id,
            kind="latent_tag",
            latent_signature=motif,
            usage_count=support,
            successes=max(1, int(support * 0.55)),
            solve_step_delta=deltas["solve_steps"],
            search_depth_delta=deltas["search_depth"],
            transfer_delta=transfer_delta,
            ambiguity=ambiguity,
            metadata={"support": support},
        )

    def _aggregate_deltas(self, validation_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        before_after: dict[str, list[float]] = defaultdict(list)
        for metrics in validation_metrics.values():
            for key, value in metrics.items():
                before_after[key].append(value)
        return {
            "solve_steps": self._pairwise_delta(before_after.get("solve_steps_before", []), before_after.get("solve_steps_after", [])),
            "search_depth": self._pairwise_delta(before_after.get("search_depth_before", []), before_after.get("search_depth_after", [])),
        }

    def _aggregate_transfer(self, transfer_scores: dict[str, float]) -> float:
        if not transfer_scores:
            return 0.0
        old = transfer_scores.get("baseline", 0.0)
        new = transfer_scores.get("with_concepts", 0.0)
        return new - old

    def _pairwise_delta(self, before: list[float], after: list[float]) -> float:
        if not before or not after:
            return 0.0
        return (sum(after) / len(after)) - (sum(before) / len(before))

    def _estimate_ambiguity(self, signature: tuple[int, ...]) -> float:
        if not signature:
            return 1.0
        unique = len(set(signature))
        return 1.0 - (unique / len(signature))
