from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from envs.base import Action, EpisodeTrace, Observation, TaskSpec, VerifyResult, VerifiableEnvironment
from verifier.base import verify_trace_basic

_MOVE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class _GridState:
    pos: tuple[int, int] = (0, 0)
    steps: int = 0
    done: bool = False


class GridPlanningEnv(VerifiableEnvironment):
    env_id = "grid_planning"

    def __init__(self) -> None:
        self._task: TaskSpec | None = None
        self._state = _GridState()

    def sample_task(self, difficulty: float, seed: int) -> TaskSpec:
        rng = random.Random(seed)
        size = 5 + int(round(2 * difficulty))
        wall_budget = max(2, int(size * difficulty))
        start = (0, 0)
        goal = (size - 1, size - 1)
        walls: set[tuple[int, int]] = set()
        while len(walls) < wall_budget:
            c = (rng.randrange(size), rng.randrange(size))
            if c not in {start, goal}:
                walls.add(c)
        max_steps = size * size
        return TaskSpec(
            task_id=f"grid-plan-{seed}",
            env_id=self.env_id,
            difficulty=difficulty,
            seed=seed,
            initial_state={"size": size, "start": list(start), "walls": [list(w) for w in sorted(walls)]},
            goal_spec={"goal": list(goal), "step_cost": 1.0},
            max_steps=max_steps,
            metadata={"generator": "deterministic_v1"},
        )

    def reset(self, task: TaskSpec) -> Observation:
        self._task = task
        self._state = _GridState(pos=tuple(task.initial_state["start"]))
        return Observation(value={"pos": list(self._state.pos), "goal": task.goal_spec["goal"], "size": task.initial_state["size"]})

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, object]]:
        if self._task is None:
            raise RuntimeError("reset() must be called before step()")
        if self._state.done:
            return Observation(value={"status": "done"}), 0.0, True, {"already_done": True}

        name = str(action.value.get("move", "")) if isinstance(action.value, dict) else ""
        dx, dy = _MOVE.get(name, (0, 0))
        size = int(self._task.initial_state["size"])
        walls = {tuple(x) for x in self._task.initial_state["walls"]}

        x, y = self._state.pos
        nx, ny = min(max(0, x + dx), size - 1), min(max(0, y + dy), size - 1)
        blocked = (nx, ny) in walls
        if not blocked:
            self._state.pos = (nx, ny)

        self._state.steps += 1
        goal = tuple(self._task.goal_spec["goal"])
        self._state.done = self._state.pos == goal or self._state.steps >= self._task.max_steps
        reward = -1.0
        if self._state.pos == goal:
            reward = 0.0

        return Observation(value={"pos": list(self._state.pos), "goal": list(goal)}), reward, self._state.done, {"blocked": blocked}

    def verify(self, task: TaskSpec, trace: EpisodeTrace) -> VerifyResult:
        basic = verify_trace_basic(task, trace)
        if not basic.correct:
            return VerifyResult(correct=False, score=0.0, failure_reason=basic.failure_reason, audit=basic.to_dict())

        if not trace.transitions:
            return VerifyResult(correct=False, score=0.0, failure_reason="empty_trace")

        final_obs = trace.transitions[-1].next_observation.value
        reached = tuple(final_obs.get("pos", [])) == tuple(task.goal_spec["goal"])
        optimal = self._shortest_path_len(task)
        used = len(trace.transitions)
        if not reached:
            return VerifyResult(correct=False, score=0.0, failure_reason="goal_not_reached", metrics={"steps_used": used, "optimal_steps": optimal})

        efficiency = 1.0 if optimal <= 0 else max(0.0, min(1.0, optimal / used))
        return VerifyResult(correct=True, score=efficiency, metrics={"steps_used": used, "optimal_steps": optimal})

    def _shortest_path_len(self, task: TaskSpec) -> int:
        size = int(task.initial_state["size"])
        start = tuple(task.initial_state["start"])
        goal = tuple(task.goal_spec["goal"])
        walls = {tuple(x) for x in task.initial_state["walls"]}
        q: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        seen = {start}
        while q:
            (x, y), d = q.popleft()
            if (x, y) == goal:
                return d
            for dx, dy in _MOVE.values():
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= size or ny >= size:
                    continue
                if (nx, ny) in walls or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
        return -1


def generate_eval_tasks(*, seeds: list[int], difficulty: float = 0.5) -> list[TaskSpec]:
    env = GridPlanningEnv()
    return [env.sample_task(difficulty=difficulty, seed=s) for s in seeds]
