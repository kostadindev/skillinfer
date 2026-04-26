# InferenceState

::: skillinfer.state.InferenceState

The posterior belief about one entity's feature vector. Created via [`Taxonomy.new_state()`](taxonomy.md#new_state), updated via `observe()` calls.

```python
state = tax.new_state(obs_noise=1.0)
state.observe("BBH", 32.7)
print(state.mean("MMLU-PRO"))  # predicted value
```

---

## Observation Methods

### `observe`

```python
def observe(self, feature: str | int, value: float) -> InferenceState
```

Observe one feature value. Runs a [Kalman update](../how-it-works/kalman-update.md), updating `mu` and `Sigma` in place.

**Parameters**

<div class="param-list" markdown>

`feature`
:   Feature name (str) or integer index.

`value`
:   Observed value.

</div>

**Returns**

`self` — for method chaining.

**Example**

```python
# Single observation
state.observe("BBH", 32.7)

# Chained observations
state.observe("BBH", 32.7).observe("IFEval", 47.1).observe("MATH Lvl 5", 18.0)

# By index
state.observe(0, 32.7)
```

---

### `observe_many`

```python
def observe_many(self, observations: dict[str | int, float]) -> InferenceState
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
state.observe_many({
    "BBH": 32.7,
    "IFEval": 47.1,
    "MATH Lvl 5": 18.0,
})
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
state.predict("GPQA")
# {'feature': 'GPQA', 'mean': 8.13, 'std': 2.78, 'ci_lower': 2.68, 'ci_upper': 13.58}

# Predict all skills
state.predict()
#      feature   mean    std  ci_lower  ci_upper
# 0     IFEval  47.10   1.00    45.14     49.06
# 1        BBH  32.70   1.00    30.74     34.66
# 2  MATH Lv 5  18.00   1.00    16.04     19.96
# 3       GPQA   8.13   2.78     2.68     13.58
# 4       MUSR  11.37   4.09     3.35     19.39
# 5   MMLU-PRO  29.93   4.21    21.68     38.18

# With 99% confidence intervals
state.predict(level=0.99)
```

---

## Query Methods

### `mean`

```python
def mean(self, feature: str | int | None = None) -> float | np.ndarray
```

Posterior mean.

- If `feature` is given: returns a scalar (the mean for that feature)
- If `feature` is `None`: returns the full (K,) mean vector

**Example**

```python
state.mean("MMLU-PRO")     # 29.9
state.mean()                # array([47.1, 32.7, 18.0, 8.1, 11.4, 29.9])
```

---

### `std`

```python
def std(self, feature: str | int | None = None) -> float | np.ndarray
```

Posterior standard deviation (square root of the covariance diagonal).

- If `feature` is given: returns a scalar
- If `feature` is `None`: returns the full (K,) std vector

**Example**

```python
state.std("MMLU-PRO")      # 4.21
state.std()                 # array([1.0, 1.0, 1.0, 2.78, 4.09, 4.21])
```

---

### `confidence_interval`

```python
def confidence_interval(
    self,
    feature: str | int,
    level: float = 0.95,
) -> tuple[float, float]
```

Gaussian confidence interval for a feature.

**Parameters**

<div class="param-list" markdown>

`feature`
:   Feature name or index.

`level`
:   Confidence level (default 0.95 for 95% CI).

</div>

**Returns**

`(lower, upper)` tuple.

**Example**

```python
state.confidence_interval("GPQA", level=0.95)
# (2.7, 13.6)

state.confidence_interval("GPQA", level=0.99)
# (0.9, 15.4)
```

---

### `most_uncertain`

```python
def most_uncertain(self, k: int = 10) -> pd.DataFrame
```

Top-k features with highest posterior standard deviation — the features you know least about.

**Returns**

DataFrame with columns `[feature, mean, std]`, sorted by `std` descending.

**Example**

```python
state.most_uncertain(k=3)
#      feature    mean   std
# 0   MMLU-PRO   29.93  4.21
# 1       MUSR   11.37  4.09
# 2       GPQA    8.13  2.78
```

!!! tip "Active learning"
    Use `most_uncertain()` to decide which feature to observe next. Observing the most uncertain feature maximizes information gain.

---

### `to_dataframe`

```python
def to_dataframe(self) -> pd.DataFrame
```

Full posterior as a DataFrame.

**Returns**

DataFrame with columns `[feature, mean, std]`.

**Example**

```python
state.to_dataframe()
#        feature   mean    std
# 0       IFEval  47.10   1.00
# 1          BBH  32.70   1.00
# 2   MATH Lvl 5  18.00   1.00
# 3         GPQA   8.13   2.78
# 4         MUSR  11.37   4.09
# 5     MMLU-PRO  29.93   4.21
```

---

### `similarity`

```python
def similarity(self, other: np.ndarray) -> float
```

Cosine similarity between the posterior mean and a target vector.

**Parameters**

<div class="param-list" markdown>

`other`
:   (K,) target vector to compare against.

</div>

**Returns**

Float in [-1, 1]. Returns 0.0 if either vector has near-zero norm.

**Example**

```python
# Compare to a known entity
state.similarity(tax.entity("meta-llama/Llama-3-70B"))  # 0.987

# Compare to the population mean
state.similarity(tax.population_mean)  # 0.954
```

---

### `copy`

```python
def copy(self) -> InferenceState
```

Deep copy of the state. The returned `InferenceState` has independent `mu` and `Sigma` arrays.

**Example**

```python
# Branch: try two different observations
state_a = state.copy()
state_b = state.copy()

state_a.observe("GPQA", 15.0)
state_b.observe("MUSR", 20.0)

# state_a and state_b are independent
```

---

## Properties

### `agent_vector`

```python
@property
def agent_vector(self) -> pd.Series
```

The inferred skill profile as a labeled Series (posterior mean). This is the agent vector — the best-guess skill profile after all observations.

**Example**

```python
state.agent_vector
# IFEval        47.10
# BBH           32.70
# MATH Lvl 5    18.00
# GPQA           8.13
# MUSR          11.37
# MMLU-PRO      29.93
# Name: agent_vector, dtype: float64

# Use it for agent-task matching
state.similarity(task_vector)  # compare agent to task requirements
```

---

### `covariance_matrix`

```python
@property
def covariance_matrix(self) -> pd.DataFrame
```

Posterior covariance as a labeled DataFrame (skill names on both axes). Shows how much uncertainty remains and how skills still co-vary after observations.

**Example**

```python
state.covariance_matrix.round(2)
#             IFEval    BBH  MATH Lvl 5   GPQA   MUSR  MMLU-PRO
# IFEval       1.00   0.12        0.08   0.05   0.03      0.09
# BBH          0.12  94.51       18.32  11.24   7.89     15.42
# ...
```

---

## Pretty printing

`print(state)` displays the full agent vector with uncertainties:

```python
print(state)
# Agent Vector (1 observations, 6 skills)
# Skill          Mean     ± Std
# -----------------------------
# IFEval      47.1000    1.0000
# BBH         32.7000    1.0000
# MATH Lvl 5  18.0000    1.0000
# GPQA         8.1300    2.7800
# MUSR        11.3700    4.0900
# MMLU-PRO    29.9300    4.2100
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu` | `np.ndarray` | (K,) posterior mean vector (raw array) |
| `Sigma` | `np.ndarray` | (K, K) posterior covariance matrix (raw array) |
| `agent_vector` | `pd.Series` | Posterior mean as a labeled Series |
| `covariance_matrix` | `pd.DataFrame` | Posterior covariance as a labeled DataFrame |
| `obs_noise` | `float` | Observation noise standard deviation |
| `feature_names` | `list[str]` | Skill names (inherited from Taxonomy) |
| `n_observations` | `int` | Number of `observe()` calls applied |
