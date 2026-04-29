# Quickstart

This guide walks through the core workflow: build a population, observe a few skills, and predict the full profile.

## 1. Build a population

A **Population** is a matrix of entities (rows) described by skills (columns). `skillinfer` learns how skills co-vary across the population.

```python
import skillinfer

# Built-in O*NET dataset: 894 occupations x 120 skills
pop = skillinfer.datasets.onet()
print(pop)
```

```text
Population(894 entities x 120 skills, shrinkage=0.0054)
  Condition number: 1884.2
  Effective dimensions: ~5 (90% variance)

  Top skill correlations:
    Skill:Equipment Maintenance <-> Skill:Repairing                r = +0.972
    Ability:Arm-Hand Steadiness <-> Ability:Manual Dexterity       r = +0.961
                  Skill:Writing <-> Ability:Written Expression     r = +0.950
```

You can also build from your own data:

```python
import pandas as pd

df = pd.read_csv("scores.csv", index_col=0)  # rows=entities, columns=skills
pop = skillinfer.Population.from_dataframe(df)
```

By default, each column is normalized to [0, 1]. Pass `normalize=False` if your data is already on a meaningful scale.

## 2. Observe and predict

Create a `Profile` for a new entity. It starts at the population mean with the full population covariance as uncertainty.

```python
profile = pop.profile()

# Observe one skill
profile.observe("Skill:Programming", 0.92)

# Predict another — never observed, inferred via covariance
profile.predict("Skill:Mathematics")
# {'feature': 'Skill:Mathematics', 'mean': 0.81, 'std': 0.11, ...}
```

Each call to `observe()` runs a Kalman update: the observed skill propagates to every other skill proportionally to how much they co-vary.

You can chain observations or observe many at once:

```python
profile.observe("Skill:Critical Thinking", 0.85).observe("Skill:Writing", 0.70)

# or equivalently:
profile.observe_many({"Skill:Critical Thinking": 0.85, "Skill:Writing": 0.70})
```

## 3. Get the full profile

```python
print(profile.predict())
```

```text
                           feature   mean    std  ci_lower  ci_upper
           Skill:Active Learning   0.94   0.12      0.71      1.17
          Skill:Active Listening   0.74   0.10      0.55      0.93
   Skill:Complex Problem Solving   1.02   0.09      0.83      1.20
         Skill:Critical Thinking   0.85   0.01      0.83      0.87  ← observed
             Skill:Programming     0.92   0.01      0.91      0.93  ← observed
...
[120 rows]
```

```python
# What should we assess next? (highest remaining uncertainty)
profile.most_uncertain(k=3)
```

```text
                     feature   mean    std
       Knowledge:Mechanical   0.24   0.24
    Skill:Equipment Selection 0.28   0.24
  Ability:Arm-Hand Steadiness 0.01   0.24
```

## 4. Use a specific entity as prior

If you know the new entity is similar to an existing one, use it as the starting point:

```python
# "This person's background is similar to a Software Developer"
profile = pop.profile(prior_entity="Software Developers")
profile.observe("Skill:Writing", 0.90)  # but stronger at writing
```

## Next steps

- [Core Concepts](concepts.md) — understand the model and when it works best
- [LLM Benchmarks tutorial](../tutorials/llm-benchmarks.md) — full end-to-end example with real data
- [API Reference](../api/population.md) — complete method documentation
