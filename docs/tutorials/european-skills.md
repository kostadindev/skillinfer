# ESCO — Profiling EU occupations

Cross-taxonomy validation with the European Skills taxonomy — a completely different data source that validates `skillinfer` generalises beyond any single taxonomy's design.

!!! info "What you'll learn"
    - Working with binary (0/1) data instead of continuous ratings
    - Cross-taxonomy validation: does the method generalise?
    - Handling sparse, independently curated skill assignments

## About ESCO

[ESCO v1.2.1](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations) is curated by the European Commission. It differs from O\*NET in three key ways:

| | O\*NET | ESCO |
|---|--------|------|
| **Source** | U.S. Department of Labor surveys | EU expert panel curation |
| **Feature type** | Continuous ratings (1–5) | Binary assignments (has/doesn't have) |
| **Scale** | 894 occupations x 120 features | 2,999 occupations x 134 skill groups |

These differences make ESCO a strong cross-taxonomy validation: if `skillinfer` works on both, the method generalises beyond any single taxonomy's design choices.

## Step 1: Load the population

```python
import numpy as np
import skillinfer

pop = skillinfer.datasets.esco()
print(pop)
```

```text
Population(2999 entities x 134 skills, shrinkage=0.0211)
  Condition number: 468.3
  Effective dimensions: ~5 (90% variance)
```

The density of the binary matrix is ~10% — most occupations have a small fraction of the skill groups.

## Step 2: Explore the covariance structure

```python
for _, row in pop.top_correlations(k=5).iterrows():
    a = row["feature_a"]
    b = row["feature_b"]
    print(f"  {a:<30} <-> {b:<30}  r = {row['correlation']:+.3f}")
```

```text
  assisting and caring           <-> making decisions                r = +0.771
  assisting and caring           <-> counselling                     r = +0.697
  counselling                    <-> making decisions                r = +0.646
  teaching and training          <-> applying civic skills           r = +0.551
  welfare                        <-> assisting and caring            r = +0.548
```

## Step 3: Observe and predict

With binary data, observations are 0.0 or 1.0:

```python
profile = pop.profile()

# Observe 3 skill groups
profile.observe("education", 1.0)
profile.observe("teaching and training", 1.0)
profile.observe("counselling", 1.0)

# Show top predicted skill groups
df = profile.predict()
df_sorted = df.sort_values("mean", ascending=False).head(10)
for _, row in df_sorted.iterrows():
    observed = " ← observed" if row["std"] < 0.01 else ""
    print(f"  {row['feature']:<45} mean={row['mean']:.3f} ± {row['std']:.3f}{observed}")
```

Even with binary data, the covariance structure captures meaningful skill relationships — observing education-related skills increases predictions for related groups like welfare and social sciences.

## Step 4: Validate transfer helps

```python
results = skillinfer.validation.held_out_evaluation(
    pop, frac_observed=[0.1, 0.3, 0.5], n_splits=10, obs_noise=0.1
)
summary = results.groupby(["frac_observed", "method"])["cosine_similarity"].mean()
print(summary)
```

The Kalman filter outperforms the diagonal baseline on ESCO just as it does on O\*NET, despite the fundamentally different data characteristics (binary vs. continuous, EU curation vs. U.S. surveys).

## Full example

See [`examples/esco.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/esco.py) for the complete script including hierarchy traversal and detailed validation.

## Key takeaway

`skillinfer` works across different data types (binary and continuous), different curation methodologies (expert panels and surveys), and different scales (134 and 120 features). The covariance transfer mechanism is robust to these variations — **it's a property of the math, not the data format**.
