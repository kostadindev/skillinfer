# Core Concepts

## Three vectors

`skillinfer` works with three kinds of vectors that live in the same K-dimensional skill space:

### Skill vector (what someone *can do*)

A **skill vector** is a K-dimensional profile of an agent's capabilities. Each dimension is a measurable skill — a benchmark score, an occupational competency, a subject grade.

```
Agent: GPT-4o
Skill vector: [IFEval=83.2, BBH=72.1, MATH=53.6, GPQA=46.7, MUSR=28.4, MMLU-PRO=68.9]
```

```
Agent: Software Developer
Skill vector: [Programming=0.87, Mathematics=0.72, Writing=0.65, Static Strength=0.14, ...]
```

For AI models, the skill vector is their benchmark scores. For humans, it's their competency ratings. `skillinfer` infers the full skill vector from partial observations — you observe a few dimensions, it predicts the rest.

### Task vector (what something *requires*)

A **task vector** lives in the same skill space but describes *requirements* instead of *capabilities*. Each dimension represents how much of that skill a task demands.

```
Task: "Solve a GPQA-level physics problem"
Task vector: [IFEval=low, BBH=medium, MATH=high, GPQA=high, MUSR=medium, MMLU-PRO=high]
```

```
Task: "Write a technical specification"
Task vector: [Programming=0.5, Writing=0.9, Critical Thinking=0.8, Static Strength=0.0, ...]
```

Task vectors can come from job descriptions (O\*NET), benchmark definitions, or manual specification.

### Agent vector (the posterior belief)

An **agent vector** is what `skillinfer` produces: a probabilistic estimate of an agent's skill vector, with both a **mean** (best guess) and **uncertainty** (how confident we are) for every skill.

```python
state = tax.new_state()               # unknown agent — starts at population mean
state.observe("BBH", 72.1)            # observe one skill

# Predict a specific skill
state.predict("GPQA")
# {'feature': 'GPQA', 'mean': 38.4, 'std': 5.2, 'ci_lower': 28.2, 'ci_upper': 48.6}

# Predict all skills at once
state.predict()
#      feature   mean   std  ci_lower  ci_upper
# 0     IFEval  62.1   8.3     45.8      78.4
# 1        BBH  72.1   1.0     70.1      74.1    ← observed
# 2  MATH Lv 5  35.2   6.1     23.2      47.2
# ...

# Get the agent vector as a labeled Series
state.agent_vector
# IFEval        62.1
# BBH           72.1
# MATH Lvl 5    35.2
# GPQA          38.4
# MUSR          21.8
# MMLU-PRO      54.7
# Name: agent_vector, dtype: float64
```

The agent vector starts as the population average and sharpens with each observation. After enough observations, it converges to the agent's true skill vector.

## How they connect

Because skill vectors, task vectors, and agent vectors all live in the same K-dimensional space, you can directly compare them:

```python
# Get a known agent's skill vector (labeled Series)
tax.skill_vector("GPT-4o")

# How similar is the inferred profile to a known agent?
state.similarity(tax.entity("GPT-4o"))           # 0.987

# How well does this agent match a task's requirements?
state.similarity(task_vector)                      # 0.91

# Which known agent is closest to the inferred profile?
best_match = max(tax.entity_names, key=lambda e: state.similarity(tax.entity(e)))
```

This shared space enables **agent-task matching**: compare the inferred agent vector against task vectors to find the best fit — even when the agent has only been observed on a few skills.

## The inference pipeline

```
Population matrix  →  Taxonomy  →  InferenceState  →  Agent vector
(agents x skills)     (covariance)  (observe skills)    (full profile)
```

1. **Population matrix**: rows are known agents (AI models, workers, students), columns are skills. This is your training data.
2. **Taxonomy**: learns how skills co-vary across the population. Exposes `skill_vector()` for known agents, `covariance_df` and `correlation_df` for the learned structure. `print(tax)` shows a summary.
3. **InferenceState**: the posterior belief about *one new agent*. Call `observe()` to update, `predict()` to get predictions with confidence intervals.
4. **Agent vector**: the output — access via `state.agent_vector` (labeled Series) or `state.predict()` (DataFrame with uncertainties). `print(state)` shows the full profile.

## Skill transfer

The key mechanism: when you observe one skill, the update propagates to *every other skill* via the learned covariance.

- Skills with **positive covariance** move in the same direction (observe high Programming → predict high Analytical Reasoning)
- Skills with **negative covariance** move opposite (observe high Programming → predict lower Static Strength)
- **Independent skills** are unaffected (observe Programming → Art stays at the population mean)

This is why a single benchmark score can predict 37 others — the covariance matrix encodes which skills travel together.

## Observation noise

The `obs_noise` parameter controls how much the filter trusts each observation relative to the prior:

- **Low noise** → the observation dominates, the agent vector snaps to the observed value
- **High noise** → the prior dominates, the agent vector moves gently

This is the standard Kalman filter trade-off. Set `obs_noise` to roughly the standard deviation of measurement error you'd expect.

| Data type | Typical `obs_noise` | Why |
|-----------|-------------------|-----|
| Normalized [0, 1] skills | 0.01 – 0.1 | Precise, bounded measurements |
| Raw benchmark scores | 1.0 – 10.0 | Large scale, noisy single-run evaluations |
| Binary skill assignments | 0.05 – 0.2 | Approximate (not truly Gaussian) |

## When it works well

The model is most powerful when:

- **Skills are correlated.** The more covariance structure exists, the more one observation tells you about everything else. If skills are truly independent, the filter reduces to per-skill updates with no benefit.
- **The population is representative.** The covariance is learned from your population. A model from a very different distribution (e.g., a vision model evaluated on language benchmarks) may not benefit from transfer.
- **Observations are roughly continuous.** The Kalman update is exact for Gaussian observations. For binary or ordinal data, it's approximate but often still useful (see the [ESCO tutorial](../tutorials/european-skills.md)).

## Assumptions and limitations

| Assumption | What it means | When it breaks |
|-----------|---------------|----------------|
| **Linear-Gaussian** | Exact Bayesian update for continuous, normally-distributed skills | Binary/ordinal data (update is approximate) |
| **Stationary skills** | Skills don't change between observations | Long time horizons (skill development over months) |
| **Point-estimate covariance** | Covariance is estimated once and treated as known | Very few agents relative to number of skills |
| **Correlated skills** | Method's value comes from off-diagonal structure | Truly independent skills (reverts to per-skill baseline) |
