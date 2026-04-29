from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from envs import Action, EpisodeTrace, TaskSpec, Transition
from envs.arithmetic_expr import ArithmeticExprEnv
from envs.grid_planning import GridPlanningEnv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tasks(path: Path) -> list[TaskSpec]:
    payload: list[dict[str, Any]] = json.loads(path.read_text())
    return [TaskSpec(**item) for item in payload]


def _run_arithmetic_task(task: TaskSpec) -> float:
    env = ArithmeticExprEnv()
    obs = env.reset(task)
    nums = list(obs.value["numbers"])
    target = int(obs.value["target"])
    expr = f"(({nums[0]}+{nums[1]})*{nums[2]}-{nums[3] if len(nums) > 3 else 0})"
    next_obs, reward, done, info = env.step(Action(value={"expression": expr}))
    trace = EpisodeTrace(
        task=task,
        transitions=[
            Transition(
                observation=obs,
                action=Action(value={"expression": expr}),
                reward=reward,
                next_observation=next_obs,
                done=done,
                env_info=info,
            )
        ],
    )
    result = env.verify(task, trace)
    if int(result.metrics.get("target", target)) != target:
        return 0.0
    return result.score


def _greedy_grid_action(pos: tuple[int, int], goal: tuple[int, int], walls: set[tuple[int, int]], size: int) -> str:
    x, y = pos
    gx, gy = goal
    candidates = []
    if x < gx:
        candidates.append("down")
    elif x > gx:
        candidates.append("up")
    if y < gy:
        candidates.append("right")
    elif y > gy:
        candidates.append("left")
    candidates.extend(["down", "right", "up", "left"])
    for move in candidates:
        dx, dy = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[move]
        nx, ny = min(max(0, x + dx), size - 1), min(max(0, y + dy), size - 1)
        if (nx, ny) not in walls:
            return move
    return "up"


def _run_grid_task(task: TaskSpec) -> float:
    env = GridPlanningEnv()
    obs = env.reset(task)
    goal = tuple(task.goal_spec["goal"])
    walls = {tuple(w) for w in task.initial_state["walls"]}
    size = int(task.initial_state["size"])
    transitions: list[Transition] = []
    for _ in range(task.max_steps):
        pos = tuple(obs.value["pos"])
        move = _greedy_grid_action(pos, goal, walls, size)
        action = Action(value={"move": move})
        next_obs, reward, done, info = env.step(action)
        transitions.append(Transition(observation=obs, action=action, reward=reward, next_observation=next_obs, done=done, env_info=info))
        obs = next_obs
        if done:
            break
    result = env.verify(task, EpisodeTrace(task=task, transitions=transitions))
    return result.score


def evaluate_dry_run() -> dict[str, float]:
    eval_dir = _repo_root() / "data" / "eval_tasks"
    arith_tasks = _load_tasks(eval_dir / "arithmetic_expr_heldout.json")
    grid_tasks = _load_tasks(eval_dir / "grid_planning_heldout.json")

    arith_scores = [_run_arithmetic_task(t) for t in arith_tasks]
    grid_scores = [_run_grid_task(t) for t in grid_tasks]
    all_scores = arith_scores + grid_scores
    return {
        "success_rate": sum(1.0 for s in all_scores if s > 0.0) / max(1, len(all_scores)),
        "mean_score": sum(all_scores) / max(1, len(all_scores)),
        "arithmetic_mean": sum(arith_scores) / max(1, len(arith_scores)),
        "grid_mean": sum(grid_scores) / max(1, len(grid_scores)),
    }
