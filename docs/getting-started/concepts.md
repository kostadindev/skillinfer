# Core Concepts

## The inference pipeline

```
Data matrix  →  Population  →  Profile  →  Predictions
(entities x skills)  (covariance)   (observe)    (full profile + uncertainty)
```

1. **Data matrix**: rows are known entities (AI models, workers, occupations), columns are skills. This is your training data.
2. **Population**: learns how skills co-vary. The covariance matrix is the core — it encodes which skills travel together.
3. **Profile**: the posterior belief about *one new entity*. Call `observe()` to update, `predict()` to get predictions with confidence intervals.
4. **Predictions**: a mean vector (best guess for every skill), a covariance matrix (what's still uncertain), and confidence intervals.

## Skill transfer

The key mechanism: when you observe one skill, the update propagates to *every other skill* via the learned covariance.

- Skills with **positive covariance** move in the same direction (observe high Programming → predict high Analytical Reasoning)
- Skills with **negative covariance** move opposite (observe high Programming → predict lower Static Strength)
- **Independent skills** are unaffected (observe Programming → Art stays at the population mean)

This is why a single benchmark score can predict 37 others — the covariance matrix encodes which skills travel together.

## Task matching

A `Task` describes what a job requires as a weighted combination of skills. Because profiles and tasks live in the same skill space, you can directly score how well an entity matches:

```python
from skillinfer import Task

task = Task({"math": 1.0, "reasoning": 0.5})
result = profile.match_score(task)
# result.score  — expected weighted-average performance (normalised)
# result.std    — uncertainty in that score
```

Weights are relative importance — they are normalised internally, so the score stays on the same scale as the underlying skills.

## Observation noise

The `noise` parameter on `pop.profile(noise=...)` controls how much the filter trusts each observation relative to the prior:

- **Low noise** → the observation dominates, the profile snaps to the observed value
- **High noise** → the prior dominates, the profile moves gently

This is the standard Kalman filter trade-off. Set `noise` to roughly the standard deviation of measurement error you'd expect.

| Data type | Typical `noise` | Why |
|-----------|----------------|-----|
| Normalised [0, 1] skills | 0.01 – 0.1 | Precise, bounded measurements |
| Raw benchmark scores | 1.0 – 10.0 | Large scale, noisy single-run evaluations |
| Binary skill assignments | 0.05 – 0.2 | Approximate (not truly Gaussian) |

## When it works well

The model is most powerful when:

- **Skills are correlated.** The more covariance structure exists, the more one observation tells you about everything else. If skills are truly independent, the filter reduces to per-skill updates with no transfer.
- **The population is representative.** The covariance is learned from your population. An entity from a very different distribution may not benefit from transfer.
- **Observations are roughly continuous.** The Kalman update is exact for Gaussian observations. For binary or ordinal data, it's approximate but often still useful (see the [ESCO tutorial](../tutorials/european-skills.md)).

## Assumptions and limitations

| Assumption | What it means | When it breaks |
|-----------|---------------|----------------|
| **Linear-Gaussian** | Exact Bayesian update for continuous, normally-distributed skills | Binary/ordinal data (update is approximate) |
| **Stationary skills** | Skills don't change between observations | Long time horizons (skill development over months) |
| **Point-estimate covariance** | Covariance is estimated once and treated as known | Very few entities relative to number of skills |
| **Correlated skills** | Method's value comes from off-diagonal structure | Truly independent skills (reverts to per-skill baseline) |
