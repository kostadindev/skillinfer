# Profile

A skill profile for one entity. Created via [`Population.profile()`](taxonomy.md), updated via `observe()` calls. Gets sharper with each observation. Carries skill descriptions from the Population.

```python
profile = pop.profile()
profile.observe("BBH", 32.7)
print(profile.predict("MMLU-PRO"))  # predicted value with uncertainty
```

---

## Observation Methods

### `observe`

```python
def observe(self, feature: str | int | Skill, value: float) -> Profile
```

Observe one feature value. Runs a [Kalman update](../how-it-works/kalman-update.md), updating the full profile in place.

**Parameters**

<div class="param-list" markdown>

`feature`
:   Feature name (str), integer index, or `Skill` object.

`value`
:   Observed value.

</div>

**Returns**

`self` — for method chaining.

**Example**

```python
from skillinfer import Skill

# String (simplest)
profile.observe("BBH", 32.7)

# Skill object (with description)
profile.observe(Skill("BBH", "Big-Bench Hard"), 32.7)

# Chained observations
profile.observe("BBH", 32.7).observe("IFEval", 47.1).observe("MATH Lvl 5", 18.0)

# By index
profile.observe(0, 32.7)
```

---

### `observe_many`

```python
def observe_many(self, observations: dict[str | int, float]) -> Profile
```

Observe multiple features at once. Equivalent to calling `observe()` sequentially.

**Parameters**

<div class="param-list" markdown>

`observations`
:   `{feature: value}` mapping.

</div>

**Returns**

`self` — for method chaining.

**Example**

```python
profile.observe_many({
    "BBH": 32.7,
    "IFEval": 47.1,
    "MATH Lvl 5": 18.0,
})
```

---

## Task Matching

### `match_score`

```python
def match_score(
    self,
    task_vector: dict[str, float] | np.ndarray | Task,
    threshold: float | None = None,
    level: float = 0.95,
) -> MatchResult
```

Score this agent against a task. Computes expected weighted performance and propagates uncertainty through the linear combination.

**Parameters**

<div class="param-list" markdown>

`task_vector`
:   `Task` object, dict mapping feature names to importance weights, or a (K,) numpy array. Weights are unitless importance weights, not required performance levels.

`threshold`
:   If given, compute P(score > threshold).

`level`
:   Confidence level for the interval (default 0.95).

</div>

**Returns**

`MatchResult` (named tuple) with fields: `score`, `std`, `ci_lower`, `ci_upper`, `p_above_threshold`.

**Example**

```python
from skillinfer import Task

# With a Task object
task = Task({"MATH Lvl 5": 1.0, "GPQA": 0.5}, "Math-heavy reasoning")
result = profile.match_score(task, threshold=50.0)
print(f"Expected: {result.score:.1f} ± {result.std:.1f}")
print(f"P(score > 50): {result.p_above_threshold:.1%}")

# Or just a dict
result = profile.match_score({"MATH Lvl 5": 1.0, "GPQA": 0.5})
```

---

## Prediction

### `predict`

```python
def predict(
    self,
    feature: str | int | None = None,
    level: float = 0.95,
) -> pd.DataFrame | dict
```

Predict skill values with uncertainty and confidence intervals.

**Parameters**

<div class="param-list" markdown>

`feature`
:   If given, predict one skill. If `None`, predict all skills.

`level`
:   Confidence level for the interval (default 0.95).

</div>

**Returns**

- If `feature` is given: `dict` with keys `feature`, `mean`, `std`, `ci_lower`, `ci_upper`
- If `feature` is `None`: `DataFrame` with all skills and their predictions

**Example**

```python
# Predict a single skill
profile.predict("GPQA")
# {'feature': 'GPQA', 'mean': 8.13, 'std': 2.78, 'ci_lower': 2.68, 'ci_upper': 13.58}

# Predict all skills
profile.predict()
#      feature   mean    std  ci_lower  ci_upper
# 0     IFEval  47.10   1.00    45.14     49.06
# 1        BBH  32.70   1.00    30.74     34.66
# ...
```

---

## Query Methods

### `mean`

```python
def mean(self, feature: str | int | None = None) -> float | np.ndarray
```

Posterior mean. Returns a scalar if `feature` is given, the full (K,) vector otherwise.

---

### `std`

```python
def std(self, feature: str | int | None = None) -> float | np.ndarray
```

Posterior standard deviation (square root of the covariance diagonal).

---

### `confidence_interval`

```python
def confidence_interval(self, feature: str | int, level: float = 0.95) -> tuple[float, float]
```

Gaussian confidence interval. Returns `(lower, upper)`.

---

### `most_uncertain`

```python
def most_uncertain(self, k: int = 10) -> pd.DataFrame
```

Top-k features with highest posterior uncertainty. Returns DataFrame with columns `[feature, mean, std]`.

!!! tip "Active learning"
    Use `most_uncertain()` to decide which feature to observe next. Observing the most uncertain feature maximizes information gain.

---

### `similarity`

```python
def similarity(self, other: np.ndarray) -> float
```

Cosine similarity between the posterior mean and a target vector. Returns float in [-1, 1].

---

### `to_dataframe`

```python
def to_dataframe(self) -> pd.DataFrame
```

Full posterior as a DataFrame with columns `[feature, mean, std]`.

---

### `copy`

```python
def copy(self) -> Profile
```

Deep copy. The returned Profile has independent arrays and preserves `n_observations`.

```python
# Branch: try two different observations
a = profile.copy()
b = profile.copy()
a.observe("GPQA", 15.0)
b.observe("MUSR", 20.0)
```

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `agent_vector` | `pd.Series` | Posterior mean as a labeled Series |
| `covariance_matrix` | `pd.DataFrame` | Posterior covariance as a labeled DataFrame |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu` | `np.ndarray` | (K,) posterior mean vector |
| `Sigma` | `np.ndarray` | (K, K) posterior covariance matrix |
| `feature_names` | `list[str]` | Skill names (from Population) |
| `n_observations` | `int` | Number of `observe()` calls applied |
