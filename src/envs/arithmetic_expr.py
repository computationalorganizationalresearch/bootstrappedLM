from __future__ import annotations

import ast
import operator
import random
from dataclasses import dataclass
from fractions import Fraction

from envs.base import Action, EpisodeTrace, Observation, TaskSpec, VerifyResult, VerifiableEnvironment
from verifier.base import verify_trace_basic


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}



def _eval_fraction_expr(expr: str) -> Fraction:
    node = ast.parse(expr, mode="eval")

    def _walk(n: ast.AST) -> Fraction:
        if isinstance(n, ast.Expression):
            return _walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return Fraction(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -_walk(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            lhs = _walk(n.left)
            rhs = _walk(n.right)
            if isinstance(n.op, ast.Div) and rhs == 0:
                raise ZeroDivisionError("division by zero")
            return Fraction(_ALLOWED_BINOPS[type(n.op)](lhs, rhs))
        raise ValueError("unsupported syntax")

    return _walk(node)


@dataclass
class _ArithmeticState:
    done: bool = False


class ArithmeticExprEnv(VerifiableEnvironment):
    env_id = "arithmetic_expr"

    def __init__(self) -> None:
        self._state = _ArithmeticState()
        self._task: TaskSpec | None = None

    def sample_task(self, difficulty: float, seed: int) -> TaskSpec:
        rng = random.Random(seed)
        count = max(3, min(6, 3 + int(round(difficulty * 3))))
        nums = [rng.randint(1, 9) for _ in range(count)]
        target = sum(nums[:2]) * nums[2] - (nums[3] if count > 3 else 0)
        max_steps = 1
        return TaskSpec(
            task_id=f"arith-{seed}",
            env_id=self.env_id,
            difficulty=difficulty,
            seed=seed,
            initial_state={"numbers": nums},
            goal_spec={"target": target, "must_use_all_numbers": False},
            max_steps=max_steps,
            metadata={"generator": "deterministic_v1"},
        )

    def reset(self, task: TaskSpec) -> Observation:
        self._task = task
        self._state = _ArithmeticState(done=False)
        return Observation(value={"numbers": task.initial_state["numbers"], "target": task.goal_spec["target"]})

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, object]]:
        if self._task is None:
            raise RuntimeError("reset() must be called before step()")
        if self._state.done:
            return Observation(value={"status": "done"}), 0.0, True, {"already_done": True}

        expr = str(action.value.get("expression", "")) if isinstance(action.value, dict) else ""
        self._state.done = True
        info = {"submitted_expression": expr}
        return Observation(value={"status": "submitted"}), 0.0, True, info

    def verify(self, task: TaskSpec, trace: EpisodeTrace) -> VerifyResult:
        basic = verify_trace_basic(task, trace)
        if not basic.correct:
            return VerifyResult(correct=False, score=0.0, failure_reason=basic.failure_reason, audit=basic.to_dict())
        if not trace.transitions:
            return VerifyResult(correct=False, score=0.0, failure_reason="empty_trace", audit=basic.to_dict())

        action = trace.transitions[-1].action.value
        expr = str(action.get("expression", "")) if isinstance(action, dict) else ""
        try:
            value = _eval_fraction_expr(expr)
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(correct=False, score=0.0, failure_reason=f"invalid_expression:{exc}", audit={"expression": expr})

        target = Fraction(int(task.goal_spec["target"]))
        correct = value == target
        return VerifyResult(
            correct=correct,
            score=1.0 if correct else 0.0,
            failure_reason=None if correct else "target_mismatch",
            metrics={"target": int(target), "value_num": value.numerator, "value_den": value.denominator},
            audit={"expression": expr, "task_id": task.task_id},
        )


def generate_eval_tasks(*, seeds: list[int], difficulty: float = 0.5) -> list[TaskSpec]:
    env = ArithmeticExprEnv()
    return [env.sample_task(difficulty=difficulty, seed=s) for s in seeds]
