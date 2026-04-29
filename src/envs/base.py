from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class TaskSpec:
    """Deterministic task descriptor used to seed/reset an environment."""

    task_id: str
    env_id: str
    difficulty: float
    seed: int
    initial_state: JsonValue
    goal_spec: dict[str, JsonValue]
    max_steps: int = 0
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Serializable observation payload emitted by an environment."""

    value: JsonValue
    info: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """Serializable action payload consumed by an environment."""

    value: JsonValue
    info: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One transition in an episode trace."""

    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    done: bool
    env_info: dict[str, JsonValue] = field(default_factory=dict)
    agent_info: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeTrace:
    """Full replayable trace for deterministic verification."""

    task: TaskSpec
    transitions: list[Transition] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifyResult:
    """Verification summary returned by env/verifier implementations."""

    correct: bool
    score: float
    failure_reason: str | None = None
    metrics: dict[str, JsonValue] = field(default_factory=dict)
    audit: dict[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class VerifiableEnvironment(Protocol):
    """Interface every task environment must implement."""

    def sample_task(self, difficulty: float, seed: int) -> TaskSpec:
        ...

    def reset(self, task: TaskSpec) -> Observation:
        ...

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, JsonValue]]:
        ...

    def verify(self, task: TaskSpec, trace: EpisodeTrace) -> VerifyResult:
        ...


def to_json_value(value: Any) -> JsonValue:
    """Best-effort conversion into a JSON-safe tree, preserving dict key order by sorting."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list | tuple):
        return [to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    raise TypeError(f"Unsupported value for deterministic JSON conversion: {type(value)!r}")


def _encode_dataclass(instance: Any) -> str:
    payload = to_json_value(asdict(instance))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _decode_dataclass(payload: str) -> dict[str, JsonValue]:
    return json.loads(payload)


def task_to_json(task: TaskSpec) -> str:
    return _encode_dataclass(task)


def task_from_json(payload: str) -> TaskSpec:
    return TaskSpec(**_decode_dataclass(payload))


def trace_to_json(trace: EpisodeTrace) -> str:
    return _encode_dataclass(trace)


def trace_from_json(payload: str) -> EpisodeTrace:
    raw = _decode_dataclass(payload)
    task = TaskSpec(**raw["task"])
    transitions = [
        Transition(
            observation=Observation(**t["observation"]),
            action=Action(**t["action"]),
            reward=float(t["reward"]),
            next_observation=Observation(**t["next_observation"]),
            done=bool(t["done"]),
            env_info=t.get("env_info", {}),
            agent_info=t.get("agent_info", {}),
        )
        for t in raw.get("transitions", [])
    ]
    return EpisodeTrace(task=task, transitions=transitions, metadata=raw.get("metadata", {}))
