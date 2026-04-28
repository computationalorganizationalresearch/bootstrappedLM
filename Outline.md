
# Zipfian World-Model Agent: Complete Implementation Specification

**Document purpose:** This Markdown file is a concrete implementation guide for building a self-training, world-model-style reasoning agent that develops its own internal abstract language through reinforcement learning, procedural environments, objective verification, compression pressure, and Zipfian concept economics.

**Audience:** Lead software developer, ML engineer, research engineer, or technical founder.

**Core design goal:** Build an agent whose intelligence comes solely from acting in procedurally generated objective worlds, not from imitating human-generated text or demonstrations.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Thesis](#2-core-thesis)
3. [Non-Goals and Constraints](#3-non-goals-and-constraints)
4. [System Architecture](#4-system-architecture)
5. [The Main Learning Loop](#5-the-main-learning-loop)
6. [Environment and Verifier System](#6-environment-and-verifier-system)
7. [Agent Core Architecture](#7-agent-core-architecture)
8. [World Model](#8-world-model)
9. [Internal Concept Language](#9-internal-concept-language)
10. [Zipfian Concept Economy](#10-zipfian-concept-economy)
11. [High-Information Objective Layer](#11-high-information-objective-layer)
12. [Taylor-Series Compressibility Objective](#12-taylor-series-compressibility-objective)
13. [Minimum Description Length and Compression](#13-minimum-description-length-and-compression)
14. [Sparsity, Symmetry, Causality, and Predictive Information](#14-sparsity-symmetry-causality-and-predictive-information)
15. [Concept Compiler](#15-concept-compiler)
16. [Planning and Search](#16-planning-and-search)
17. [Curriculum and Self-Play](#17-curriculum-and-self-play)
18. [Replay Buffer and Storage](#18-replay-buffer-and-storage)
19. [Training Losses](#19-training-losses)
20. [Initial Environments](#20-initial-environments)
21. [Later Environments](#21-later-environments)
22. [Evaluation and Metrics](#22-evaluation-and-metrics)
23. [Anti-Reward-Hacking Controls](#23-anti-reward-hacking-controls)
24. [Engineering Stack](#24-engineering-stack)
25. [Repository Layout](#25-repository-layout)
26. [Implementation Milestones](#26-implementation-milestones)
27. [Concrete MVP Specification](#27-concrete-mvp-specification)
28. [Code Skeletons](#28-code-skeletons)
29. [Debugging and Observability](#29-debugging-and-observability)
30. [Expected Failure Modes](#30-expected-failure-modes)
31. [Research Anchors and Justifications](#31-research-anchors-and-justifications)
32. [Final Build Doctrine](#32-final-build-doctrine)

---

# 1. Executive Summary

We are building a system called, for now:

```text
ZWM-Agent = Zipfian World-Model Agent
```

The agent is not a conventional large language model trained on internet-scale text. It is a self-training reinforcement learning system that develops an internal symbolic language by interacting with objective environments.

The system learns from:

- procedural tasks,
- simulators,
- games,
- symbolic rewrite systems,
- generated math problems,
- generated programming tasks,
- formal proof environments,
- causal intervention worlds,
- and other environments where success can be automatically verified.

The agent does **not** learn from:

- human-written solution datasets,
- human demonstrations,
- internet text,
- instruction tuning,
- RLHF preference labels,
- or human-authored explanations.

The desired result is not a human-like conversational assistant. The desired result is an alien but competent reasoning system that develops abstract concepts because those concepts help it predict, control, compress, and solve objective tasks.

The core loop is:

```text
act → observe → predict → verify → compress → abstract → reuse → generate harder tasks
```

The internal language is shaped by Zipfian pressure:

```text
few very common concepts
some medium-frequency reusable concepts
many rare specialized concepts
```

The high-information objective layer further encourages the agent to discover representations that are locally compressible, sparse, causal, symmetric, compositional, and predictive.

---

# 2. Core Thesis

The central thesis is:

> General reasoning can emerge from objective interaction plus compression into a cost-sensitive internal concept language.

A conventional language model asks:

```text
What token would a human write next?
```

This agent asks:

```text
What action changes the world in a way that satisfies a verifiable objective?
```

The agent develops intelligence through experience, not imitation.

It should learn concepts such as:

```text
object
relation
change
quantity
goal
cause
path
symmetry
invariant
composition
decomposition
search
verification
local approximation
compression
```

But it should learn them as internal alien abstractions, not as English words.

The internal concept language should become useful because it helps the agent:

- predict future states,
- plan actions,
- compress experience,
- discover reusable subroutines,
- transfer strategies across environments,
- and generate harder tasks for itself.

---

# 3. Non-Goals and Constraints

## 3.1 Non-Goals

Do not build the first version as a chatbot.

Do not begin by training on:

```text
Common Crawl
Wikipedia
books
Stack Overflow
GitHub code
human-written proofs
human-written solutions
conversation data
preference labels
```

Do not optimize for:

```text
fluency
plausibility
human style
human-likeness
```

Optimize for:

```text
verified success
prediction
compression
causal control
abstraction
transfer
```

## 3.2 Allowed Human Contribution

Humans may design:

- simulators,
- grammars,
- verifiers,
- symbolic rules,
- objective functions,
- procedural task generators,
- environment APIs,
- and software infrastructure.

Humans should not provide solved examples as training data.

This distinction matters:

```text
Allowed: humans design the game rules.
Not allowed: humans provide the winning trajectories.
```

## 3.3 Objective Verification Requirement

Every task must have a verifier.

A verifier must answer:

```text
Did the agent actually satisfy the objective?
```

The verifier should be deterministic whenever possible.

Examples:

| Domain | Verifier |
|---|---|
| Arithmetic | exact symbolic equivalence |
| Algebra | SymPy or custom canonicalizer |
| Logic | SAT solver / truth table |
| Program synthesis | generated hidden tests |
| Formal proof | proof checker |
| Gridworld | goal-state check |
| Physics | simulator state constraints |
| Causal world | intervention outcome match |

---

# 4. System Architecture

## 4.1 Full System Diagram

```text
+------------------------------------------------------------------+
|                       ZWM-Agent System                           |
+------------------------------------------------------------------+
|                                                                  |
|  Procedural World Suite                                          |
|      ├── arithmetic worlds                                       |
|      ├── symbolic rewrite worlds                                 |
|      ├── logic worlds                                            |
|      ├── gridworlds                                              |
|      ├── program synthesis worlds                                |
|      ├── physics worlds                                          |
|      ├── causal worlds                                           |
|      └── self-generated abstract worlds                          |
|                                                                  |
|  Objective Verifier Layer                                        |
|      ├── exact checkers                                          |
|      ├── symbolic solvers                                        |
|      ├── test runners                                            |
|      ├── proof checkers                                          |
|      └── simulator validators                                    |
|                                                                  |
|  Agent Core                                                      |
|      ├── observation encoder                                     |
|      ├── latent state model                                      |
|      ├── world model                                             |
|      ├── policy head                                             |
|      ├── value head                                              |
|      ├── concept bottleneck                                      |
|      ├── internal concept library                                |
|      └── local approximation heads                               |
|                                                                  |
|  Planning and Search                                             |
|      ├── model-predictive planning                               |
|      ├── MCTS-style search                                       |
|      └── verifier-guided search                                  |
|                                                                  |
|  Learning Infrastructure                                         |
|      ├── replay buffer                                           |
|      ├── curriculum engine                                       |
|      ├── concept compiler                                        |
|      ├── high-information objective registry                     |
|      └── evaluation harness                                      |
|                                                                  |
+------------------------------------------------------------------+
```

## 4.2 Core Components

The lead developer should implement these components as cleanly separated modules:

1. `AbstractEnvironment`
2. `Verifier`
3. `TaskGenerator`
4. `Trajectory` and `Step`
5. `ReplayBuffer`
6. `ObservationEncoder`
7. `LatentDynamicsModel`
8. `WorldModel`
9. `PolicyHead`
10. `ValueHead`
11. `ConceptBottleneck`
12. `ConceptLibrary`
13. `ConceptCompiler`
14. `ZipfRegularizer`
15. `HighInformationObjectiveRegistry`
16. `Planner`
17. `CurriculumEngine`
18. `Evaluator`

Do not start with a giant model. Start with clean interfaces.

---

# 5. The Main Learning Loop

## 5.1 Conceptual Loop

The system repeatedly does this:

```text
1. Generate task.
2. Reset environment.
3. Agent observes state.
4. Agent selects internal concepts.
5. Agent plans using policy, value model, and world model.
6. Agent takes action.
7. Environment transitions.
8. Verifier checks progress or final success.
9. Trajectory is stored.
10. World model is trained.
11. Policy and value are trained.
12. Concept usage is updated.
13. Concept compiler proposes merges, splits, or new concepts.
14. Curriculum generator creates harder tasks.
```

## 5.2 Main Pseudocode

```python
while training:
    task_batch = curriculum.sample_batch()
    trajectories = []

    for task in task_batch:
        env = env_registry.get(task.env_id)
        trajectory = run_episode(agent, env, task)
        verification = env.verify(task, trajectory)
        trajectory.verification = verification
        trajectories.append(trajectory)

    replay.add(trajectories)

    train_world_model(agent, replay)
    train_policy_value(agent, replay)
    train_concept_bottleneck(agent, replay)
    train_high_information_objectives(agent, replay)

    concept_compiler.update(trajectories, concept_library)
    curriculum.update(trajectories)

    if should_evaluate():
        evaluator.run(agent)
```

## 5.3 Episode Pseudocode

```python
def run_episode(agent, env, task):
    obs = env.reset(task)
    trajectory = Trajectory(task=task)

    for t in range(task.max_steps):
        action_mask = env.available_actions()

        action, agent_info = agent.act(
            observation=obs,
            action_mask=action_mask,
            use_search=True,
        )

        next_obs, reward, done, env_info = env.step(action)

        trajectory.steps.append(
            Step(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                env_info=env_info,
                agent_info=agent_info,
            )
        )

        obs = next_obs

        if done:
            break

    return trajectory
```

---

# 6. Environment and Verifier System

## 6.1 Environment Interface

Every environment must implement this interface:

```python
class AbstractEnvironment:
    def reset(self, task: "Task") -> "Observation":
        raise NotImplementedError

    def step(self, action: "Action") -> tuple["Observation", float, bool, dict]:
        raise NotImplementedError

    def available_actions(self) -> "ActionMask":
        raise NotImplementedError

    def verify(self, task: "Task", trajectory: "Trajectory") -> "VerificationResult":
        raise NotImplementedError

    def generate_task(self, difficulty: float, seed: int) -> "Task":
        raise NotImplementedError
```

## 6.2 Task Schema

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Task:
    id: str
    env_id: str
    seed: int
    difficulty: float
    initial_state: Any
    goal_spec: Any
    max_steps: int
    generator_params: dict
```

## 6.3 Verification Result Schema

```python
@dataclass
class VerificationResult:
    success: bool
    score: float
    failure_reason: str | None
    metrics: dict
```

## 6.4 Example: Rewrite Environment

Task:

```text
Transform a start expression into a target expression using legal rewrite rules.
```

Example:

```text
start:  ((x + 2) + 3)
target: (x + 5)
actions:
  - associate_add
  - commute_add
  - combine_constants
```

Verifier:

```text
success = final expression exactly equals target expression
```

Additional safety check:

```text
symbolic_equivalence(start, final) == true
```

---

# 7. Agent Core Architecture

## 7.1 Agent Pipeline

At each step:

```text
observation
→ observation encoder
→ latent state model
→ concept bottleneck
→ world model
→ policy/value heads
→ planner
→ action
```

## 7.2 Agent Class Skeleton

```python
class ZWMAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ObservationEncoder(config)
        self.dynamics = LatentDynamicsModel(config)
        self.concepts = ConceptBottleneck(config)
        self.world_model = WorldModel(config)
        self.policy = PolicyHead(config)
        self.value = ValueHead(config)
        self.planner = Planner(config)

    def act(self, observation, action_mask, use_search=True):
        obs_emb = self.encoder(observation)
        latent_state = self.dynamics.update(obs_emb)

        concept_tokens, concept_emb, concept_info = self.concepts(latent_state)

        if use_search:
            action, search_info = self.planner.plan(
                latent_state=latent_state,
                concept_state=concept_emb,
                action_mask=action_mask,
                world_model=self.world_model,
                policy=self.policy,
                value=self.value,
            )
        else:
            logits = self.policy(latent_state, concept_emb, action_mask)
            action = sample_masked(logits, action_mask)
            search_info = {}

        return action, {
            "latent_state": latent_state,
            "concept_tokens": concept_tokens,
            "concept_info": concept_info,
            "search_info": search_info,
        }
```

## 7.3 Observation Encoders

Different environments need different encoders:

| Environment type | Encoder |
|---|---|
| Symbolic expression | tree transformer or token transformer |
| Gridworld | CNN, ViT, or grid transformer |
| Logic formula | syntax tree encoder |
| Program state | AST/graph transformer |
| Numeric simulation | MLP or temporal transformer |
| Proof state | token/tree encoder |

All encoders should produce a common latent size:

```text
z_obs ∈ R^d
```

Recommended MVP:

```text
d = 256 or 512
```

Later:

```text
d = 768, 1024, or larger
```

---

# 8. World Model

## 8.1 Purpose

The world model predicts consequences.

It should estimate:

- next latent state,
- next observation,
- reward,
- termination,
- uncertainty,
- and local dynamics.

This allows the agent to plan inside imagined futures instead of only reacting.

## 8.2 World Model Interface

```python
@dataclass
class WorldModelPrediction:
    next_latent: torch.Tensor
    reward_logits: torch.Tensor
    done_logits: torch.Tensor
    obs_reconstruction: object | None
    uncertainty: torch.Tensor
    local_coefficients: dict | None

class WorldModel(nn.Module):
    def forward(self, latent_state, action, concept_state):
        return WorldModelPrediction(...)
```

## 8.3 World Model Loss

```text
L_world =
    next_state_prediction_loss
  + reward_prediction_loss
  + done_prediction_loss
  + observation_reconstruction_loss
  + uncertainty_calibration_loss
```

## 8.4 Why This Matters

A policy-only agent learns habits.

A world-model agent learns consequences.

The desired system should reason like:

```text
If I apply this rewrite, the expression becomes simpler.
If I push this block, it blocks the door.
If I introduce this lemma, the proof state splits.
If I choose this intervention, I can identify the causal parent.
```

---

# 9. Internal Concept Language

## 9.1 Purpose

The agent needs an internal discrete concept language.

This language is not English. It is a set of internal symbols used to compress and organize experience.

A concept may represent:

- a relation,
- an operator,
- a subgoal,
- a proof tactic,
- a planning macro,
- a causal feature,
- a local law,
- a symmetry,
- an invariant,
- or a reusable subroutine.

## 9.2 Concept Schema

```python
@dataclass
class Concept:
    id: int
    embedding: torch.Tensor

    usage_count: int
    success_credit: float
    transfer_score: float
    compression_gain: float

    local_approximation_gain: float
    causal_prediction_gain: float
    symmetry_reuse_gain: float
    compositional_reuse_gain: float
    future_prediction_gain: float

    ambiguity_penalty: float
    cost: float

    parent_concepts: list[int]
    child_concepts: list[int]
    arity: int
    creation_step: int
    active: bool
```

## 9.3 Concept Bottleneck

The model must pass part of its reasoning through discrete concept tokens.

```text
latent state → concept logits → selected concepts → concept embedding → policy/world model
```

Implementation options:

1. Gumbel-Softmax categorical sampling.
2. Vector-quantized latent codes.
3. Top-k sparse concept selection.
4. Later: dynamic concept creation.

MVP recommendation:

```text
Use Gumbel-Softmax or top-k soft selection.
Start with fixed concept vocabulary size N = 1024.
Allow up to 8 active concepts per timestep.
```

## 9.4 Concept Bottleneck Code Skeleton

```python
class ConceptBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_concepts = config.num_concepts
        self.max_active = config.max_active_concepts
        self.to_logits = nn.Linear(config.latent_dim, self.num_concepts)
        self.embedding = nn.Embedding(self.num_concepts, config.concept_dim)

    def forward(self, latent_state):
        logits = self.to_logits(latent_state)

        # MVP: top-k selection. Later replace or combine with Gumbel-Softmax.
        top_values, top_indices = torch.topk(logits, k=self.max_active, dim=-1)
        weights = torch.softmax(top_values, dim=-1)

        selected_emb = self.embedding(top_indices)
        concept_emb = (selected_emb * weights.unsqueeze(-1)).sum(dim=-2)

        return top_indices, concept_emb, {
            "logits": logits,
            "weights": weights,
        }
```

---

# 10. Zipfian Concept Economy

## 10.1 Purpose

The internal concept language should not be flat.

A good reasoning system should develop:

```text
few extremely common abstractions
some reusable domain abstractions
many rare specialized abstractions
```

This is Zipf-like.

A rank-frequency distribution has the rough form:

```text
frequency(rank r) ∝ 1 / r^s
```

Usually `s` is near 1 for classic Zipf-like behavior, but this system should treat `s` as a tunable regularization target, not a hard law.

## 10.2 Important Warning

Do not optimize for Zipf alone.

A meaningless random process can produce a Zipf-like distribution. Zipfian structure is useful only when the symbols are grounded in task success, prediction, compression, and transfer.

Therefore:

```text
Zipf is a regularizer, not the main reward.
```

## 10.3 Utility-Gated Zipf

Only useful concept usage should count strongly.

Define:

```python
utility_score = (
    alpha_success * success_credit
  + alpha_transfer * transfer_score
  + alpha_compression * compression_gain
  + alpha_taylor * local_approximation_gain
  + alpha_causal * causal_prediction_gain
  + alpha_symmetry * symmetry_reuse_gain
  + alpha_composition * compositional_reuse_gain
  + alpha_predictive * future_prediction_gain
  - alpha_ambiguity * ambiguity_penalty
  - alpha_cost * concept_cost
)
```

Effective usage:

```python
effective_usage[i] = usage_count[i] * max(utility_score[i], 0)
```

## 10.4 Zipf Loss

```python
def zipf_loss(effective_counts, s=1.0, epsilon=1e-8):
    counts = effective_counts.float() + epsilon
    p = counts / counts.sum()
    p_sorted, _ = torch.sort(p, descending=True)

    ranks = torch.arange(1, len(p_sorted) + 1, device=p.device).float()
    q = 1.0 / (ranks ** s)
    q = q / q.sum()

    # KL(p || q)
    return torch.sum(p_sorted * (torch.log(p_sorted + epsilon) - torch.log(q + epsilon)))
```

## 10.5 Concept Cost

Frequent useful concepts should become cheaper.

```python
concept_cost[i] = min_cost + alpha / torch.sqrt(usage_count[i] + 1)
```

This creates an abbreviation-like pressure:

```text
common useful abstraction → cheap
rare specialized abstraction → more expensive
```

## 10.6 Desired Behavior

Example frequent concepts:

```text
goal
object
relation
change
quantity
path
invariant
verify
search
compose
```

Example rare concepts:

```text
specific rewrite pattern #839
rare causal configuration #122
unusual gridworld trap #55
specialized proof macro #9017
```

---

# 11. High-Information Objective Layer

## 11.1 Purpose

The agent should not merely maximize sparse reward.

It should learn world representations that are useful in high-information settings.

Useful high-information properties include:

- local compressibility,
- minimum description length,
- sparsity,
- smoothness,
- symmetry,
- invariance,
- causal structure,
- compositionality,
- predictive information,
- algorithmic simplicity,
- scale-free structure.

These objectives shape the internal representations.

They should not override verified task success.

## 11.2 Objective Registry

Implement high-information objectives as plug-ins.

```python
class Objective:
    name: str
    weight: float

    def compute(self, batch, agent_outputs, replay):
        raise NotImplementedError
```

Registry:

```python
objective_registry = [
    TaskSuccessObjective(),
    WorldModelObjective(),
    PolicyObjective(),
    ValueObjective(),
    ZipfObjective(),
    TaylorCompressibilityObjective(),
    MDLObjective(),
    SparsityObjective(),
    SymmetryObjective(),
    CausalConsistencyObjective(),
    ManifoldObjective(),
    PredictiveInformationObjective(),
    CompositionObjective(),
    AlgorithmicSimplicityObjective(),
]
```

## 11.3 Implementation Rule

Start with a small set:

```text
1. task success
2. world-model prediction
3. concept bottleneck
4. Zipfian concept usage
5. minimum description length
6. Taylor-style local compressibility
```

Add the rest only after the core system works.

---

# 12. Taylor-Series Compressibility Objective

## 12.1 Intuition

Many complex systems are globally hard but locally simple.

A function may be difficult over the whole state space, but near a specific state it may be approximated by:

```text
current value + local slope + local curvature + small residual
```

The agent should prefer internal representations where nearby future states can be predicted with low-order local approximations.

Taylor-style reasoning is especially useful for:

- physics,
- control,
- optimization,
- planning,
- causal intervention,
- continuous simulation,
- and local model-based prediction.

## 12.2 Important Qualification

Taylor-series compressibility applies most directly to continuous or differentiable latent spaces.

For symbolic environments, the same idea should be interpreted as:

```text
Can local transitions be described by simple low-complexity rules?
```

Do not force literal Taylor expansions onto raw symbolic tokens.

Instead, apply the objective to learned latent states.

## 12.3 Local Approximation Head

Add this to the world model:

```python
class LocalApproximationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_head = nn.Linear(config.latent_dim + config.action_dim, config.latent_dim)
        self.first_order_head = nn.Linear(config.latent_dim + config.action_dim, config.latent_dim)
        self.second_order_head = nn.Linear(config.latent_dim + config.action_dim, config.latent_dim)
        self.residual_head = nn.Linear(config.latent_dim + config.action_dim, config.latent_dim)

    def forward(self, latent_state, action_embedding, delta):
        x = torch.cat([latent_state, action_embedding], dim=-1)

        base = self.base_head(x)
        first_order = self.first_order_head(x)
        second_order = self.second_order_head(x)
        residual_scale = torch.nn.functional.softplus(self.residual_head(x))

        z_pred = base + first_order * delta + second_order * (delta ** 2)

        return {
            "base": base,
            "first_order": first_order,
            "second_order": second_order,
            "residual_scale": residual_scale,
            "z_pred": z_pred,
        }
```

## 12.4 Taylor Compressibility Loss

```python
def taylor_compressibility_loss(z_next_actual, local_output):
    prediction_error = mse(local_output["z_pred"], z_next_actual)

    complexity_penalty = (
        local_output["first_order"].abs().mean()
        + local_output["second_order"].abs().mean()
        + local_output["residual_scale"].mean()
    )

    return prediction_error + lambda_complexity * complexity_penalty
```

## 12.5 Desired Effect

The agent should discover concepts that make dynamics simpler.

Example:

```text
Without concept: object trajectories look chaotic.
With concept: velocity + acceleration explains local change.
```

Example:

```text
Without concept: rewrite sequences look arbitrary.
With concept: expression complexity decreases under a known transformation.
```

---

# 13. Minimum Description Length and Compression

## 13.1 Purpose

A good concept should compress many experiences.

The agent should prefer:

```text
short reusable rule + small residuals
```

over:

```text
large memorized lookup table
```

## 13.2 MDL-Style Objective

Define description length as:

```text
description_length =
    cost(model)
  + cost(concepts used)
  + cost(residual errors)
```

Loss:

```python
L_mdl = model_complexity_cost + concept_usage_cost + residual_prediction_cost
```

## 13.3 Compression Gain

For a proposed concept:

```python
compression_gain = loss_without_concept - loss_with_concept
```

If `compression_gain > 0`, the concept helped.

## 13.4 Example

Suppose the agent observes 100 tasks involving expressions like:

```text
(a + b) + c → a + (b + c)
```

Without a concept, it stores many cases.

With a concept, it stores:

```text
associativity-like transformation
```

This concept compresses experience and should gain utility.

---

# 14. Sparsity, Symmetry, Causality, and Predictive Information

## 14.1 Sparsity Objective

Only a few concepts should be active at once.

```python
L_sparsity = active_concept_count_penalty + latent_l1_penalty
```

Desired behavior:

```text
physics task → activate mass, velocity, collision, goal
logic task → activate implication, contradiction, case split
```

Do not activate every concept everywhere.

## 14.2 Symmetry and Invariance Objective

The agent should detect when transformed situations are structurally equivalent.

Examples:

```text
renaming variables should preserve proof structure
translating a maze should preserve path logic
rotating an object may preserve identity
commuting operands may preserve value
```

Equivariance objective:

```python
L_equivariance = mse(
    encode(transform(obs)),
    transform_latent(encode(obs))
)
```

World-model equivariance:

```python
L_world_equivariance = mse(
    world_model(T(state), T(action)),
    T(world_model(state, action))
)
```

## 14.3 Causal Consistency Objective

The agent should distinguish correlation from controllable cause.

Add a causal prediction head:

```python
class CausalHead(nn.Module):
    def forward(self, latent_state, action):
        return causal_adjacency_logits
```

Loss:

```python
L_causal = mse(predicted_intervention_effect, observed_intervention_effect)
L_causal_sparse = l1(causal_adjacency)
L_causal_total = L_causal + lambda_sparse_graph * L_causal_sparse
```

Desired behavior:

```text
press switch → door opens
move block → path changes
rewrite expression → form changes but value preserved
change irrelevant variable → no important effect
```

## 14.4 Predictive Information Objective

The latent state should preserve information that predicts the future.

Approximate this with contrastive future prediction.

```python
L_predictive_info = -contrastive_score(predicted_future_latent, actual_future_latent)
```

The representation should discard noise and preserve future-relevant structure.

## 14.5 Compositionality Objective

The agent should build complex concepts from simpler concepts.

```python
L_composition = mse(
    compose(concept_a, concept_b),
    observed_compound_concept
)
```

Examples:

```text
object + support → stability
quantity + conservation → invariant
goal + obstacle → path planning
loop + invariant → program proof
symmetry + transformation → equivalence class
```

---

# 15. Concept Compiler

## 15.1 Purpose

The concept compiler watches experience and improves the internal language.

It proposes:

- new concepts,
- merged concepts,
- split concepts,
- retired concepts,
- parent-child concept relationships,
- and concept cost updates.

## 15.2 Interface

```python
class ConceptCompiler:
    def analyze(self, trajectories: list[Trajectory]) -> list[ConceptProposal]:
        raise NotImplementedError

    def update_library(
        self,
        proposals: list["ConceptProposal"],
        concept_library: "ConceptLibrary",
    ) -> "ConceptLibrary":
        raise NotImplementedError
```

## 15.3 Concept Proposal Schema

```python
@dataclass
class ConceptProposal:
    concept_type: str
    embedding: torch.Tensor
    source_trajectory_ids: list[str]
    parent_concepts: list[int]
    compression_gain: float
    transfer_score: float
    predicted_utility: float
    suggested_arity: int
    evidence: dict
```

## 15.4 Create a Concept When

```text
pattern recurs across tasks
pattern reduces search depth
pattern improves world-model prediction
pattern compresses experience
pattern transfers to another environment
pattern composes with existing concepts
pattern has low ambiguity
```

Numerical rule:

```python
if (
    compression_gain > threshold_compression
    and transfer_score > threshold_transfer
    and novelty_score > threshold_novelty
    and ambiguity_penalty < max_ambiguity
):
    create_concept()
```

## 15.5 Merge Concepts When

```text
embeddings are close
usage contexts overlap
predictive effects are similar
separate concepts do not improve performance
```

## 15.6 Split Concepts When

```text
concept is frequent but ambiguous
same concept predicts different futures in different domains
concept causes high error variance
subclusters have distinct utility profiles
```

---

# 16. Planning and Search

## 16.1 Purpose

Planning is required because early policies will be weak and many rewards are sparse.

The agent should use its world model to imagine futures.

## 16.2 Planner Inputs

The planner receives:

- latent state,
- concept state,
- action mask,
- world model,
- policy prior,
- value estimate,
- concept costs,
- uncertainty estimate.

## 16.3 MCTS-Style Planner Skeleton

```python
class Planner:
    def plan(self, latent_state, concept_state, action_mask, world_model, policy, value):
        root = SearchNode(latent_state=latent_state, prior=1.0)

        for _ in range(self.num_simulations):
            node = self.select(root)
            prediction = world_model(node.latent_state, node.action, concept_state)
            self.expand(node, prediction, policy, action_mask)
            self.backup(node, prediction)

        return self.best_action(root), self.search_info(root)
```

## 16.4 Planner Selection Score

Use something like:

```text
score = Q + c_puct * prior * sqrt(parent_visits) / (1 + visits)
        - concept_cost_penalty
        - uncertainty_penalty
```

Allow uncertainty bonuses during exploration.

## 16.5 Verifier-Guided Search

For formal environments, the verifier can provide direct feedback.

Examples:

```text
proof tactic accepted or rejected
rewrite legal or illegal
program passes more tests
SAT assignment satisfies more clauses
```

The search tree should store verifier progress.

---

# 17. Curriculum and Self-Play

## 17.1 Purpose

A static procedural generator eventually becomes too easy or too narrow.

The curriculum should keep tasks near the agent's frontier.

Target success range:

```text
20% to 60%
```

Too easy:

```text
increase difficulty
combine skills
lengthen horizon
add distractors
```

Too hard:

```text
simplify task
shorten horizon
isolate subskill
reduce branching factor
```

## 17.2 Task Stats

```python
@dataclass
class TaskStats:
    env_id: str
    generator_params: dict
    attempts: int
    success_rate: float
    avg_steps: float
    avg_return: float
    avg_world_model_error: float
    avg_concept_cost: float
    concept_usage_entropy: float
    transfer_score: float
    last_updated: int
```

## 17.3 Curriculum Sampling

```python
def sample_task(candidates, target_success=0.4):
    weights = []
    for task in candidates:
        frontier_score = 1.0 / (abs(task.success_rate - target_success) + 0.05)
        novelty = novelty_bonus(task)
        transfer = transfer_bonus(task)
        weights.append(frontier_score + novelty + transfer)

    return weighted_sample(candidates, weights)
```

## 17.4 Generator-Solver-Verifier Setup

Long-term architecture:

```text
Generator creates tasks.
Solver attempts tasks.
Verifier checks outcomes.
Concept compiler extracts reusable abstractions.
Generator creates harder variants.
```

Generator reward:

```text
+ valid task
+ novel task
+ frontier difficulty
+ causes useful concept discovery
- impossible task
- trivial task
- invalid task
- task solved by known shortcut only
```

---

# 18. Replay Buffer and Storage

## 18.1 Step Schema

```python
@dataclass
class Step:
    obs: object
    action: object
    reward: float
    next_obs: object
    done: bool
    env_info: dict
    agent_info: dict
```

## 18.2 Trajectory Schema

```python
@dataclass
class Trajectory:
    id: str
    task: Task
    steps: list[Step]
    verification: VerificationResult | None = None
    total_reward: float = 0.0
    concepts_used: list[int] = field(default_factory=list)
    created_concepts: list[int] = field(default_factory=list)
```

## 18.3 Replay Priorities

Prioritize:

```text
successful rare trajectories
near misses
high prediction error
high concept discovery
high transfer examples
frontier-difficulty tasks
reward-hacking suspects
```

## 18.4 Storage Recommendations

Local development:

```text
SQLite + file blobs
```

Medium scale:

```text
PostgreSQL + object storage
```

High-throughput RL:

```text
custom sharded replay
or DeepMind Reverb-style replay system
```

---

# 19. Training Losses

## 19.1 Total Loss

```text
L_total =
    L_policy
  + L_value
  + L_world
  + L_concept
  + λ_zipf L_zipf
  + λ_taylor L_taylor
  + λ_mdl L_mdl
  + λ_sparse L_sparsity
  + λ_symmetry L_symmetry
  + λ_causal L_causal
  + λ_predictive L_predictive
  + λ_composition L_composition
  + λ_algorithmic L_algorithmic
```

## 19.2 Policy Loss

Use PPO, IMPALA, V-trace, or actor-critic.

MVP recommendation:

```text
PPO or simple actor-critic
```

Policy loss:

```text
L_policy = -log π(a_t | s_t, c_t) * advantage_t
```

## 19.3 Value Loss

```text
L_value = (V(s_t, c_t) - return_t)^2
```

## 19.4 World Model Loss

```text
L_world =
    prediction_loss(next_latent)
  + reward_loss
  + done_loss
  + reconstruction_loss
  + uncertainty_calibration_loss
```

## 19.5 Concept Usefulness Loss

A concept is useful if it improves prediction or control.

```python
concept_utility = loss_without_concept - loss_with_concept
```

Positive utility means the concept helped.

## 19.6 Zipf Loss Weight

Start small:

```text
λ_zipf = 0.001 to 0.01
```

Increase only if concept usage remains flat or degenerate.

## 19.7 Taylor Loss Weight

Start small:

```text
λ_taylor = 0.001 to 0.05
```

Use mostly in continuous or latent-dynamics-heavy environments.

---

# 20. Initial Environments

Start with simple environments that are easy to verify.

## 20.1 Arithmetic Environment

Tasks:

```text
solve equations
simplify expressions
compare expressions
find missing values
```

Example:

```text
3x + 7 = 22
```

Actions:

```text
subtract constant
add constant
divide coefficient
combine terms
simplify
```

Verifier:

```text
exact symbolic solution check
```

## 20.2 Symbolic Rewrite Environment

Task:

```text
Transform expression A into expression B using legal rewrite actions.
```

Actions:

```text
associate
commute
distribute
factor
combine constants
substitute
```

Verifier:

```text
exact target reached
symbolic equivalence preserved
```

## 20.3 Logic Environment

Tasks:

```text
derive conclusion from premises
satisfy formula
prove propositional identity
```

Verifier:

```text
truth table
SAT solver
proof checker
```

## 20.4 Gridworld Environment

Tasks:

```text
reach target
collect object
open door
avoid hazard
use key
move block
```

Verifier:

```text
goal reached
constraint satisfied
step count measured
```

## 20.5 Program Synthesis Mini-Language

Create a tiny DSL.

Task:

```text
write a program mapping generated inputs to generated outputs
```

Verifier:

```text
hidden property-based tests
runtime limit
syntax validity
```

---

# 21. Later Environments

## 21.1 Formal Proof Environment

Start with a toy theorem prover, then move to Lean or another formal system.

Agent action:

```text
apply tactic
introduce variable
rewrite
split goal
use lemma
```

Verifier:

```text
proof checker accepts or rejects each state
```

## 21.2 Physics Simulation

Use simple 2D physics first.

Tasks:

```text
move object to target
build stable structure
predict collision outcome
balance object
minimize energy
```

## 21.3 Causal Discovery Environment

Generate hidden causal graphs.

Agent actions:

```text
intervene on variable
observe outcome
predict counterfactual
select next experiment
```

Verifier:

```text
causal graph recovered
intervention predictions correct
fewest interventions rewarded
```

## 21.4 Multi-Agent Games

Use non-linguistic communication only.

Tasks:

```text
coordination
competition
resource gathering
territory control
predator-prey
```

## 21.5 Self-Generated Abstract Worlds

The generator creates new symbolic rule systems.

Verifier checks:

```text
rules are valid
goals are reachable
tasks are not trivial
state transitions are deterministic or well-defined
```

---

# 22. Evaluation and Metrics

## 22.1 Evaluation Philosophy

Evaluation must be procedural and held out.

Do not evaluate on human benchmark examples initially.

Use:

```text
unseen seeds
harder parameters
new compositions
new surface encodings
new environment variants
cross-domain transfer tests
```

## 22.2 Core Metrics

Track:

```text
success rate
average return
sample efficiency
steps to solve
planning depth required
world-model prediction error
value calibration
concept vocabulary size
concept usage entropy
Zipf slope
Zipf R² fit
concept transfer score
compression ratio
reward hacking incidents
```

## 22.3 Zipf Metrics

Compute:

```text
log(rank) vs log(frequency)
```

Track:

```text
estimated slope s
tail mass
top-k dominance
concept entropy
utility-weighted Zipf fit
```

Warning:

```text
Good Zipf fit alone is not success.
```

Zipf only matters if task performance, transfer, and compression also improve.

## 22.4 Concept Ablation

Periodically test:

```text
performance with concepts
performance without concepts
performance with shuffled concepts
performance with top concepts removed
performance with rare concepts removed
```

A useful concept system should show measurable degradation when meaningful concepts are ablated.

---

# 23. Anti-Reward-Hacking Controls

## 23.1 Expected Problem

The agent may exploit the checker instead of solving the intended task.

Examples:

```text
finds simulator bug
exploits unit test weakness
uses invalid but accepted proof encoding
creates degenerate generated tasks
learns shortcut artifacts
```

## 23.2 Controls

Use:

```text
multiple verifiers
hidden randomized tests
metamorphic tests
invariant checks
simulator fuzzing
proof checker isolation
sandboxing
holdout generators
adversarial validation
```

## 23.3 Example: Expression Equivalence

Use several checks:

```text
1. exact canonical form comparison
2. random numeric substitution
3. symbolic simplification
4. transformation history validity
```

## 23.4 Example: Program Synthesis

Use:

```text
visible tests for progress shaping
hidden tests for final reward
random property tests
runtime limits
memory limits
syntax validation
```

---

# 24. Engineering Stack

Recommended initial stack:

```text
Language: Python
ML framework: PyTorch
Environment API: Gymnasium-like custom API
Distributed execution: Ray, optional initially
Experiment tracking: Weights & Biases or MLflow
Config management: Hydra or Pydantic settings
Symbolic math: SymPy
Logic/SAT: Z3
Formal proof later: Lean
Storage: SQLite for MVP, PostgreSQL later
Testing: PyTest + Hypothesis
Packaging: Poetry or uv
```

Use strict reproducibility:

```text
all tasks have seeds
all generators log parameters
all training runs save config snapshots
all evaluations are reproducible
```

---

# 25. Repository Layout

```text
zwm_agent/
  README.md
  pyproject.toml
  configs/
    mvp.yaml
    environments.yaml
    objectives.yaml

  environments/
    base.py
    arithmetic/
      env.py
      generator.py
      verifier.py
    rewrite/
      env.py
      rules.py
      generator.py
      verifier.py
    logic/
      env.py
      generator.py
      verifier.py
    gridworld/
      env.py
      generator.py
      verifier.py
    program_synthesis/
      env.py
      dsl.py
      generator.py
      verifier.py

  agent/
    encoder.py
    dynamics.py
    world_model.py
    local_approximation.py
    policy.py
    value.py
    concept_bottleneck.py
    planner.py
    agent.py

  concepts/
    library.py
    compiler.py
    proposals.py
    zipf.py
    metrics.py

  objectives/
    base.py
    policy.py
    value.py
    world_model.py
    zipf.py
    taylor.py
    mdl.py
    sparsity.py
    symmetry.py
    causal.py
    predictive.py
    composition.py

  training/
    train_loop.py
    replay.py
    curriculum.py
    evaluator.py
    losses.py
    checkpointing.py

  verification/
    base.py
    symbolic.py
    sat.py
    program_tests.py
    metamorphic.py

  infrastructure/
    logging.py
    distributed.py
    config.py
    seeds.py

  tests/
    test_environments.py
    test_verifiers.py
    test_zipf.py
    test_replay.py
    test_training_smoke.py

  scripts/
    train_mvp.py
    evaluate.py
    inspect_concepts.py
```

---

# 26. Implementation Milestones

## Phase 0: Interfaces and Infrastructure

Build:

```text
AbstractEnvironment
Task
Verifier
Trajectory
ReplayBuffer
logging
checkpointing
config system
basic evaluator
```

Success criterion:

```text
Can run random policy through environments and store trajectories.
```

## Phase 1: First Objective Worlds

Build:

```text
arithmetic environment
rewrite environment
gridworld environment
logic environment
```

Success criterion:

```text
Random agent can interact.
Verifier catches valid and invalid outcomes.
Generated tasks are reproducible.
```

## Phase 2: Basic RL Agent

Build:

```text
observation encoder
policy head
value head
actor-critic or PPO trainer
```

Success criterion:

```text
Agent beats random on simple generated tasks.
```

## Phase 3: World Model

Build:

```text
latent dynamics
next-state prediction
reward prediction
done prediction
short imagined rollouts
```

Success criterion:

```text
World-model prediction error decreases.
Planning with model improves performance.
```

## Phase 4: Concept Bottleneck

Build:

```text
fixed concept vocabulary
top-k or Gumbel-Softmax concept selection
concept embeddings
concept usage logging
concept ablation tests
```

Success criterion:

```text
Concept usage becomes non-random.
Ablating concepts hurts performance.
```

## Phase 5: Zipfian Regularizer

Build:

```text
usage counter
utility-weighted counts
Zipf loss
concept cost schedule
Zipf metrics dashboard
```

Success criterion:

```text
Concept distribution develops a useful long tail without performance collapse.
```

## Phase 6: High-Information Objectives

Add:

```text
MDL objective
Taylor compressibility objective
sparsity objective
predictive information objective
```

Success criterion:

```text
Representations become more compressed and predictive.
World-model performance improves.
Concept transfer improves.
```

## Phase 7: Concept Compiler

Add:

```text
concept creation
concept merging
concept splitting
concept retirement
utility scoring
```

Success criterion:

```text
New concepts improve future task performance and compression.
```

## Phase 8: Generator-Solver-Verifier Self-Play

Add:

```text
adaptive task generator
frontier difficulty targeting
novelty scoring
self-generated environment variants
```

Success criterion:

```text
Generator creates valid frontier tasks that drive continued improvement.
```

## Phase 9: Formal Proof and Program Expansion

Add:

```text
toy theorem prover
program synthesis DSL
SAT tasks
Lean integration later
```

Success criterion:

```text
Agent solves generated formal tasks verified by external checkers.
```

---

# 27. Concrete MVP Specification

## 27.1 MVP Name

```text
ZWM-Agent-MVP-1
```

## 27.2 MVP Includes

```text
arithmetic environment
symbolic rewrite environment
gridworld environment
simple actor-critic or PPO agent
small world model
fixed concept bottleneck
concept usage tracker
Zipf regularizer
MDL objective
Taylor-style latent local approximation objective
replay buffer
evaluator
metrics dashboard
```

## 27.3 MVP Excludes

```text
dynamic concept creation
Lean integration
large transformer
human-language interface
multi-agent games
learned task generator
full causal graph learning
```

## 27.4 MVP Success Criteria

The MVP is successful if:

```text
1. agent beats random baseline on generated tasks
2. world model prediction error decreases during training
3. concept bottleneck is used non-randomly
4. concept ablation reduces performance
5. utility-weighted concept distribution shows long-tail structure
6. Zipf regularizer does not destroy performance
7. Taylor/MDL losses improve compression or prediction
8. system can reproduce runs from seed/config
```

---

# 28. Code Skeletons

## 28.1 Verification Result

```python
@dataclass
class VerificationResult:
    success: bool
    score: float
    failure_reason: str | None
    metrics: dict
```

## 28.2 Rewrite Environment

```python
class RewriteEnvironment(AbstractEnvironment):
    def reset(self, task):
        self.task = task
        self.state = task.initial_state
        return self.observe()

    def observe(self):
        return self.state

    def available_actions(self):
        return compute_valid_rewrite_mask(self.state)

    def step(self, action):
        if not is_valid_rewrite(self.state, action):
            return self.observe(), -0.1, False, {"invalid": True}

        self.state = apply_rewrite(self.state, action)
        done = self.state == self.task.goal_spec["target_expr"]
        reward = 1.0 if done else -0.01

        return self.observe(), reward, done, {}

    def verify(self, task, trajectory):
        target = task.goal_spec["target_expr"]
        success = self.state == target
        equivalent = symbolic_equivalence(task.initial_state, self.state)

        if success and equivalent:
            return VerificationResult(True, 1.0, None, {"steps": len(trajectory.steps)})

        return VerificationResult(
            False,
            0.0,
            "target_not_reached_or_equivalence_failed",
            {"equivalent": equivalent, "steps": len(trajectory.steps)},
        )
```

## 28.3 Zipf Tracker

```python
class ConceptUsageTracker:
    def __init__(self, num_concepts):
        self.usage = torch.zeros(num_concepts)
        self.utility = torch.zeros(num_concepts)

    def update(self, concept_ids, utility_delta):
        for cid in concept_ids:
            self.usage[cid] += 1
            self.utility[cid] += utility_delta

    def effective_counts(self):
        utility = torch.relu(self.utility)
        return self.usage * (1.0 + utility)
```

## 28.4 Objective Registry

```python
class ObjectiveRegistry:
    def __init__(self, objectives):
        self.objectives = objectives

    def compute_total_loss(self, batch, outputs, replay):
        losses = {}
        total = 0.0

        for obj in self.objectives:
            loss = obj.compute(batch, outputs, replay)
            weighted = obj.weight * loss
            losses[obj.name] = loss.detach().item()
            total = total + weighted

        return total, losses
```

## 28.5 Training Step

```python
def training_step(agent, replay, objective_registry, optimizer):
    batch = replay.sample()

    outputs = agent.forward_batch(batch)

    total_loss, loss_log = objective_registry.compute_total_loss(
        batch=batch,
        outputs=outputs,
        replay=replay,
    )

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
    optimizer.step()

    return loss_log
```

---

# 29. Debugging and Observability

## 29.1 Required Logs

Log every training run with:

```text
config
random seed
git commit
environment versions
task generator parameters
model checkpoint path
objective weights
```

## 29.2 Required Metrics

Track:

```text
reward
success rate
world-model loss
policy loss
value loss
Zipf loss
Taylor loss
MDL loss
concept entropy
active concepts per step
concept usage top-k
tail mass
ablation performance
invalid action rate
verifier failure reasons
```

## 29.3 Concept Inspection Tool

Build script:

```text
scripts/inspect_concepts.py
```

It should show:

```text
concept id
usage count
utility score
top environments used in
success credit
compression gain
nearest concepts
example trajectories
ablation impact
```

## 29.4 Search Tree Inspection

For planner debugging, log:

```text
selected action
search depth
visit counts
Q values
policy priors
world-model predicted rewards
actual rewards
uncertainty
```

---

# 30. Expected Failure Modes

## 30.1 Zipf Collapse

Problem:

```text
agent overuses a few concepts for everything
```

Fix:

```text
increase ambiguity penalty
increase task-conditioned concept diversity
increase sparsity structure
lower Zipf loss weight
add concept ablation pressure
```

## 30.2 Concept Explosion

Problem:

```text
agent creates or activates too many rare concepts
```

Fix:

```text
increase symbol cost
require compression gain
require transfer score
merge similar concepts
retire unused concepts
```

## 30.3 World-Model Hallucination

Problem:

```text
planner exploits imagined futures that do not work in real environment
```

Fix:

```text
shorter rollout horizon
uncertainty penalty
more real-environment correction
calibration loss
model ensemble
```

## 30.4 Reward Hacking

Problem:

```text
agent exploits verifier bug
```

Fix:

```text
multiple verifiers
hidden tests
metamorphic validation
sandboxing
adversarial generated checks
```

## 30.5 Narrow Intelligence

Problem:

```text
agent becomes good only at one task family
```

Fix:

```text
multi-domain curriculum
cross-domain transfer metrics
held-out generators
environment randomization
concept transfer reward
```

## 30.6 Auxiliary Objective Domination

Problem:

```text
agent optimizes compression or Zipf at the expense of task success
```

Fix:

```text
lower auxiliary weights
use success-gated auxiliary rewards
stop-gradient for some metrics
require task performance threshold before applying strong regularizers
```

---

# 31. Research Anchors and Justifications

This system is speculative, but it is grounded in several known directions.

## 31.1 Experience-Based Agents

David Silver and Richard Sutton argue for an era where agents learn primarily from their own experience rather than only from static human datasets. This supports the core design choice of procedural environments, interaction, and objective reward.

Reference:

```text
Silver, D. and Sutton, R. The Era of Experience.
https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
```

## 31.2 World Models

DreamerV3 is a strong example of learning a world model and improving behavior by imagining future scenarios. It reports broad performance across diverse control tasks using a single configuration, making it a relevant model-based RL precedent.

Reference:

```text
Hafner et al. Mastering Diverse Domains through World Models.
https://arxiv.org/abs/2301.04104
```

## 31.3 Formal Proof With RL

AlphaProof demonstrates that reinforcement learning in a formal proof environment such as Lean can produce verified mathematical reasoning. This supports the plan to use proof checkers and formal environments as objective domains.

Reference:

```text
Hubert et al. Olympiad-level formal mathematical reasoning with reinforcement learning.
https://www.nature.com/articles/s41586-025-09833-y
```

## 31.4 Discrete Latent Concepts

Gumbel-Softmax provides a differentiable way to train categorical latent variables, making it relevant for a discrete concept bottleneck.

Reference:

```text
Jang, Gu, and Poole. Categorical Reparameterization with Gumbel-Softmax.
https://arxiv.org/abs/1611.01144
```

## 31.5 Zipfian Structure

Zipfian distributions appear in language at many levels. For this system, the point is not to imitate human language but to create an efficient internal concept economy where frequent useful abstractions become cheap and rare specialized abstractions remain available.

References:

```text
Zipfian Distributions in Child-Directed Speech.
https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00070/114204/Zipfian-Distributions-in-Child-Directed-Speech

Kanwal et al. Zipf's Law of Abbreviation and the Principle of Least Effort.
https://www.sciencedirect.com/science/article/abs/pii/S0010027717301166
```

## 31.6 Minimum Description Length

MDL supports the principle that better models compress data by balancing model complexity and residual error. This motivates the compression objective.

Reference:

```text
Grünwald. A Tutorial Introduction to the Minimum Description Length Principle.
https://arxiv.org/abs/math/0406077
```

---

# 32. Final Build Doctrine

The system should be built according to these principles:

## 32.1 No Imitation First

Do not begin with human text.

The agent should first learn from objective worlds.

## 32.2 Verification Over Plausibility

A solution is good only if the verifier says it is good.

## 32.3 Compression Over Memorization

The agent should prefer reusable rules over lookup tables.

## 32.4 Concepts Must Earn Their Place

A concept should survive only if it improves at least one of:

```text
task success
prediction
compression
transfer
causal control
local approximation
search efficiency
```

## 32.5 Zipf Is a Regularizer, Not a Religion

Long-tail concept usage is useful only if grounded in performance.

## 32.6 World Models Before Language Interfaces

The agent should first learn consequences, not conversation.

A human-language interface can be added later as a translation layer.

## 32.7 Start Small and Measurable

The first useful prototype is not a giant AGI system.

It is a small agent that can:

```text
solve generated symbolic tasks
learn a predictive world model
use internal concepts
show long-tail concept usage
improve through curriculum
```

## 32.8 Final System Loop

The final system should continuously execute:

```text
generate task
→ act
→ verify
→ predict
→ locally approximate
→ compress
→ abstract
→ compose
→ transfer
→ generate harder task
```

The intended result is an alien but useful reasoner:

```text
not trained to sound human
not trained to imitate human text
not dependent on collected human examples
but grounded in objective success, world modeling, and reusable abstraction
```

