# Quickstart

This guide walks through the core workflow: build a taxonomy, observe a few skills, and predict the full profile.

## 1. Build a taxonomy

A **taxonomy** is a population of agents (rows) described by skills (columns). `skillinfer` learns how skills co-vary across the population.

```python
import pandas as pd
import skillinfer

# Rows = agents (AI models, workers, students, ...)
# Columns = skills (benchmarks, competencies, subjects, ...)
df = pd.read_csv("scores.csv", index_col=0)
tax = skillinfer.Taxonomy.from_dataframe(df)
print(tax)
```

```text
Taxonomy(4576 agents x 6 skills, shrinkage=0.0006)
  Condition number: 12.3
  Effective dimensions: ~3 (90% variance)

  Top skill correlations:
       GPQA <-> MMLU-PRO  r = +0.879
        BBH <-> MMLU-PRO  r = +0.871
       GPQA <-> BBH       r = +0.863
```

By default, each column is normalized to [0, 1]. Pass `normalize=False` if your data is already on a meaningful scale.

## 2. Observe and predict

Create an `InferenceState` for a new agent. It starts at the population mean with the full population covariance as uncertainty.

```python
state = tax.new_state(obs_noise=1.0)

# Observe one skill
state.observe("BBH", 32.7)

# Predict another — never observed, inferred via covariance transfer
state.predict("MMLU-PRO")
# {'feature': 'MMLU-PRO', 'mean': 29.93, 'std': 4.21, 'ci_lower': 21.68, 'ci_upper': 38.18}
```

Each call to `observe()` runs a Kalman update: the observed skill propagates to every other skill proportionally to how much they co-vary.

You can chain observations:

```python
state.observe("IFEval", 47.1).observe("MATH Lvl 5", 18.0)
```

Or observe many at once:

```python
state.observe_many({"IFEval": 47.1, "MATH Lvl 5": 18.0})
```

## 3. Get the full profile

```python
# Predict all skills with confidence intervals
state.predict()
```

```text
     feature   mean    std  ci_lower  ci_upper
0     IFEval  47.10   1.00    45.14     49.06
1        BBH  32.70   1.00    30.74     34.66
2  MATH Lv 5  18.00   1.00    16.04     19.96
3       GPQA   8.13   2.78     2.68     13.58  ← inferred
4       MUSR  11.37   4.09     3.35     19.39  ← inferred
5   MMLU-PRO  29.93   4.21    21.68     38.18  ← inferred
```

```python
# The agent vector — the inferred skill profile as a labeled Series
state.agent_vector
# IFEval        47.10
# BBH           32.70
# MATH Lvl 5    18.00
# GPQA           8.13
# MUSR          11.37
# MMLU-PRO      29.93
# Name: agent_vector, dtype: float64

# Compare to a known agent's skill vector
state.similarity(tax.skill_vector("meta-llama/Llama-3-70B"))  # 0.987

# What should we evaluate next?
state.most_uncertain(k=3)
```

```text
     feature    mean   std
0   MMLU-PRO   29.93  4.21  ← highest uncertainty
1       MUSR   11.37  4.09
2       GPQA    8.13  2.78
```

```python
# Pretty print the full agent vector
print(state)
# Agent Vector (3 observations, 6 skills)
# Skill          Mean     ± Std
# -----------------------------
# IFEval      47.1000    1.0000
# BBH         32.7000    1.0000
# MATH Lvl 5  18.0000    1.0000
# GPQA         8.1300    2.7800
# MUSR        11.3700    4.0900
# MMLU-PRO    29.9300    4.2100
```

## 4. Use a specific entity as prior

If you know the new entity is similar to an existing one, use it as the starting point:

```python
# "This new model is based on Llama-3-70B"
state = tax.new_state(prior_entity="meta-llama/Llama-3-70B", obs_noise=1.0)
state.observe("MATH Lvl 5", 25.0)  # fine-tuned for math
# Posterior reflects Llama-3's profile updated with the math observation
```

## Choosing `obs_noise`

The `obs_noise` parameter controls how much the filter trusts each observation relative to the prior.

| `obs_noise` | Trust level | When to use |
|-------------|-------------|-------------|
| **0.01 – 0.1** | High trust | Normalized [0, 1] data, precise measurements |
| **1.0 – 10.0** | Low trust | Raw (unnormalized) data, noisy measurements |

**Rule of thumb:** set `obs_noise` to roughly the standard deviation of measurement error you'd expect for a single observation.

- With `normalize=True` (default): values around `0.05` work well
- With `normalize=False` and raw scores: `1.0` or higher is typical

## Next steps

- [Core Concepts](concepts.md) — understand the model and when it works best
- [LLM Benchmarks tutorial](../tutorials/llm-benchmarks.md) — full end-to-end example
- [API Reference](../api/taxonomy.md) — complete method documentation
