# Profile

A skill profile for one entity. Created via [`Population.profile()`](population.md), updated via `observe()` calls. Gets sharper with each observation.

```python
profile = pop.profile()
profile.observe("BBH", 32.7)
print(profile.predict())
```

---

## Observation

### `observe`

```python
def observe(self, feature: str | int | Skill, value: float | None = None) -> Profile
```

Observe one feature value. Runs a [Kalman update](../how-it-works/kalman-update.md), updating the full profile in place. Returns `self` for chaining.

If `feature` is a `Skill` with a score, `value` can be omitted.

**Example**

```python
profile.observe("BBH", 32.7).observe("IFEval", 47.1)

# Or using Skill objects
profile.observe(Skill("BBH", score=32.7))
```

---

### `observe_many`

```python
def observe_many(self, observations: dict[str | int, float] | list[Skill]) -> Profile
```

Observe multiple features at once. Accepts a `{feature: value}` dict or a list of `Skill` objects with scores. Returns `self` for chaining.

**Example**

```python
profile.observe_many({"BBH": 32.7, "IFEval": 47.1, "MATH Lvl 5": 18.0})

# Or using Skill objects
profile.observe_many([Skill("BBH", score=32.7), Skill("IFEval", score=47.1)])
```

---

## Prediction

### `predict`

```python
def predict(
    self,
    feature: str | int | None = None,
    level: float = 0.95,
    detail: bool = False,
) -> pd.DataFrame | dict
```

Predict skill values with confidence intervals.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature` | `str \| int \| None` | `None` | Predict one skill, or all if `None`. |
| `level` | `float` | `0.95` | Confidence level for the interval. |
| `detail` | `bool` | `False` | Include `confidence` (0-1) and `source` (`"observed"` / `"predicted"`) columns. |

**Returns**

- Single feature: `dict` with keys `feature`, `mean`, `std`, `ci_lower`, `ci_upper`
- All features: `pd.DataFrame` with those columns

**Example**

```python
profile.predict("GPQA")
# {'feature': 'GPQA', 'mean': 8.13, 'std': 2.78, 'ci_lower': 2.68, 'ci_upper': 13.58}

profile.predict()
#      feature   mean    std  ci_lower  ci_upper
# 0     IFEval  47.10   1.00    45.14     49.06
# ...
```

---

### `most_uncertain`

```python
def most_uncertain(self, k: int = 10) -> pd.DataFrame
```

Top-k features with highest posterior uncertainty. Returns DataFrame with columns `[feature, mean, std]`.

!!! tip "Active learning"
    Use `most_uncertain()` to decide which feature to observe next — the most uncertain feature gives the most information gain.

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

Score this agent against a task. Computes expected weighted-average performance (normalised by weight sum) and propagates uncertainty.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_vector` | `dict \| np.ndarray \| Task` | — | Skill importance weights. Normalised internally. |
| `threshold` | `float \| None` | `None` | If given, compute P(score > threshold). |
| `level` | `float` | `0.95` | Confidence level for the interval. |

**Returns:** `MatchResult` (named tuple) with fields: `score`, `std`, `ci_lower`, `ci_upper`, `p_above_threshold`.

**Example**

```python
from skillinfer import Task

task = Task({"MATH Lvl 5": 1.0, "GPQA": 0.5})
result = profile.match_score(task, threshold=50.0)
print(f"Expected: {result.score:.1f} ± {result.std:.1f}")
print(f"P(score > 50): {result.p_above_threshold:.1%}")
```

---

### `skillinfer.rank_agents`

```python
skillinfer.rank_agents(
    task_vector: dict[str, float] | np.ndarray | Task,
    profiles: dict[str, Profile],
    threshold: float | None = None,
) -> pd.DataFrame
```

Rank a pool of agents by expected task performance. Calls `match_score` on each profile and sorts descending.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_vector` | `dict \| np.ndarray \| Task` | — | Skill importance weights. |
| `profiles` | `dict[str, Profile]` | — | Map of agent name → Profile. |
| `threshold` | `float \| None` | `None` | If given, include P(score > threshold). |

**Returns:** DataFrame with columns `[agent, expected_score, std, p_above_threshold]`.

**Example**

```python
task = Task({"math": 1.0, "reasoning": 0.5})
ranking = skillinfer.rank_agents(task, {"alice": alice, "gpt-4o": gpt4o})
print(ranking)
#     agent  expected_score    std  p_above_threshold
# 0   alice            0.91   0.03               None
# 1  gpt-4o            0.85   0.03               None
```

---

## Evaluation

### `summary`

```python
def summary(self, true_vector: np.ndarray | None = None) -> dict
```

Summary statistics for this profile.

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `n_features` | `int` | Total feature count |
| `n_observed` | `int` | Number of observed features |
| `n_predicted` | `int` | Number of predicted features |
| `mean_std` | `float` | Average posterior standard deviation |
| `uncertainty_reduction` | `float` | Fraction of prior uncertainty removed (0-1) |
| `top_predicted` | `list[dict]` | Top 3 unobserved features by predicted mean |
| `most_uncertain` | `list[dict]` | Top 3 features by posterior std |

If `true_vector` is given, also includes:

| Key | Type | Description |
|-----|------|-------------|
| `mae` | `float` | Mean absolute error |
| `rmse` | `float` | Root mean squared error |
| `max_error` | `float` | Largest single prediction error |
| `cosine_similarity` | `float` | Cosine similarity to ground truth |
| `coverage_95` | `float` | Fraction of true values inside 95% CIs |

---

### `mae`

```python
def mae(self, true_vector: np.ndarray) -> float
```

Mean absolute error between posterior mean and a ground truth vector.

---

### `rmse`

```python
def rmse(self, true_vector: np.ndarray) -> float
```

Root mean squared error between posterior mean and a ground truth vector.

---

### `similarity`

```python
def similarity(self, other: np.ndarray) -> float
```

Cosine similarity between the posterior mean and a target vector. Returns float in [-1, 1].

---

### `uncertainty_ratio`

```python
def uncertainty_ratio(self, Sigma_0: np.ndarray) -> float
```

Fraction of prior uncertainty remaining: `tr(Sigma) / tr(Sigma_0)`. A value of 0.5 means uncertainty has halved since the prior. Useful for deciding when enough observations have been collected.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Sigma_0` | `np.ndarray` | (K, K) prior covariance, typically `pop.covariance`. |

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

Gaussian confidence interval for a single feature. Returns `(lower, upper)`.

---

### `to_dataframe`

```python
def to_dataframe(self, detail: bool = False) -> pd.DataFrame
```

Full posterior as a DataFrame with columns `[feature, mean, std]`. With `detail=True`, adds `confidence` and `source` columns.

---

### `copy`

```python
def copy(self) -> Profile
```

Deep copy. The returned Profile has independent arrays and preserves all state.

---

## Export / Import

### `to_dict`

```python
def to_dict(self) -> dict
```

Export the profile as a plain dict (JSON-serialisable). Contains `feature_names`, `mean`, `std`, `n_observations`, `observed_features`, `noise`.

### `to_json`

```python
def to_json(self, path: str | None = None) -> str
```

Export the profile as JSON. If `path` is given, writes to file. Always returns the JSON string.

### `Profile.from_dict`

```python
@classmethod
def from_dict(cls, data: dict) -> Profile
```

Reconstruct a Profile from the output of `to_dict()`. Restores the mean vector, observed features, and metadata. The covariance is reconstructed as a diagonal matrix from the exported std values.

### `Profile.from_json`

```python
@classmethod
def from_json(cls, source: str) -> Profile
```

Reconstruct a Profile from a JSON string or file path.

**Example**

```python
# Save
profile.to_json("agent_profile.json")

# Load (from file)
restored = Profile.from_json("agent_profile.json")

# Or round-trip via dict
d = profile.to_dict()
restored = Profile.from_dict(d)
```

!!! note
    Export/import preserves the mean vector and metadata but uses a diagonal covariance approximation. For full covariance fidelity, re-create the profile from a Population and re-apply observations.

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
| `feature_names` | `list[str]` | Feature names (from Population) |
| `n_observations` | `int` | Number of `observe()` calls applied |
| `noise` | `float` | Observation noise standard deviation |
