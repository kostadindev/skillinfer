# Taxonomy

::: skillinfer.taxonomy.Taxonomy

The population model. Wraps an entity-feature matrix and its learned covariance structure.

```python
import skillinfer

tax = skillinfer.Taxonomy.from_dataframe(df)
```

---

## Constructors

### `Taxonomy.from_dataframe`

```python
@classmethod
def from_dataframe(
    cls,
    df: pd.DataFrame,
    normalize: bool = True,
    covariance: str = "ledoit-wolf",
) -> Taxonomy
```

Build a Taxonomy from a pandas DataFrame.

**Parameters**

<div class="param-list" markdown>

`df`
:   DataFrame with rows = entities, columns = features. All values must be numeric.

`normalize`
:   If `True` (default), scale each column to [0, 1]. Set to `False` if your data is already on a meaningful common scale.

`covariance`
:   Covariance estimation method. `"ledoit-wolf"` (default, recommended) applies shrinkage for numerical stability. `"sample"` uses the unregularized sample covariance with a small ridge.

</div>

**Returns**

A `Taxonomy` instance with the learned covariance structure.

**Example**

```python
import pandas as pd
import skillinfer

df = pd.read_csv("scores.csv", index_col=0)
tax = skillinfer.Taxonomy.from_dataframe(df, normalize=False)
print(tax)
# Taxonomy(4576 entities x 6 features, shrinkage=0.0006)
```

---

### `Taxonomy.from_csv`

```python
@classmethod
def from_csv(
    cls,
    path: str,
    index_col: int | str = 0,
    normalize: bool = True,
    covariance: str = "ledoit-wolf",
) -> Taxonomy
```

Build a Taxonomy from a CSV file. Convenience wrapper around `from_dataframe`.

**Parameters**

<div class="param-list" markdown>

`path`
:   Path to the CSV file.

`index_col`
:   Column to use as entity names (index). Default `0` (first column).

`normalize`
:   If `True`, scale each column to [0, 1].

`covariance`
:   `"ledoit-wolf"` or `"sample"`.

</div>

---

## Methods

### `entity`

```python
def entity(self, name: str) -> np.ndarray
```

Get the raw feature vector for a named entity. Returns a **copy** as a plain numpy array.

**Parameters**

<div class="param-list" markdown>

`name`
:   Entity name (must exist in the index).

</div>

**Returns**

`np.ndarray` of shape `(K,)` — the entity's feature values.

**Example**

```python
vec = tax.entity("meta-llama/Llama-3-70B")
print(vec)  # array([69.15, 63.79, 27.49, 14.09, 17.72, 42.68])
```

---

### `skill_vector`

```python
def skill_vector(self, name: str) -> pd.Series
```

Get a named agent's skill vector as a labeled Series. Like `entity()`, but returns a `pd.Series` with skill names as the index.

**Parameters**

<div class="param-list" markdown>

`name`
:   Agent name (must exist in the index).

</div>

**Returns**

`pd.Series` with skill names as index.

**Example**

```python
sv = tax.skill_vector("meta-llama/Llama-3-70B")
print(sv)
# IFEval        69.15
# BBH           63.79
# MATH Lvl 5    27.49
# GPQA          14.09
# MUSR          17.72
# MMLU-PRO      42.68
# Name: meta-llama/Llama-3-70B, dtype: float64

# Access individual skills
sv["BBH"]      # 63.79
sv["MMLU-PRO"] # 42.68
```

---

### `new_state`

```python
def new_state(
    self,
    prior_entity: str | None = None,
    prior_mean: np.ndarray | None = None,
    obs_noise: float = 0.05,
) -> InferenceState
```

Create an [`InferenceState`](inference-state.md) for a new entity.

**Parameters**

<div class="param-list" markdown>

`prior_entity`
:   If given, use this entity's feature vector as the prior mean.

`prior_mean`
:   If given, use this array directly as the prior mean.

`obs_noise`
:   Observation noise standard deviation. Controls the trust balance between observations and the prior.

</div>

If neither `prior_entity` nor `prior_mean` is given, the population mean is used.

**Returns**

An `InferenceState` initialized with the chosen prior mean and the population covariance.

**Example**

```python
# Default: population mean prior
state = tax.new_state(obs_noise=1.0)

# Entity-specific prior
state = tax.new_state(prior_entity="meta-llama/Llama-3-70B", obs_noise=1.0)

# Custom prior
state = tax.new_state(prior_mean=np.zeros(6), obs_noise=0.5)
```

---

### `pca`

```python
def pca(self, n_components: int = 15) -> dict
```

PCA decomposition of the entity-feature matrix.

**Returns**

Dict with keys:

- `components`: `(n_components, K)` principal component vectors
- `explained_variance_ratio`: `(n_components,)` fraction of variance per component
- `cumulative`: `(n_components,)` cumulative variance explained

**Example**

```python
pca = tax.pca(n_components=3)
print(f"3 components explain {pca['cumulative'][-1]:.1%} of variance")
```

---

### `top_correlations`

```python
def top_correlations(self, k: int = 20) -> pd.DataFrame
```

Top-k strongest feature-feature correlations by absolute value.

**Returns**

DataFrame with columns `[feature_a, feature_b, correlation]`, sorted by `|correlation|` descending.

**Example**

```python
tax.top_correlations(k=5)
#       feature_a     feature_b  correlation
# 0          GPQA      MMLU-PRO        0.879
# 1           BBH      MMLU-PRO        0.871
# 2          GPQA           BBH        0.863
```

---

### `condition_number`

```python
def condition_number(self) -> float
```

Condition number of the covariance matrix ($\lambda_{\max} / \lambda_{\min}$). Lower values indicate better numerical stability.

---

## Properties

### `covariance_df`

```python
@property
def covariance_df(self) -> pd.DataFrame
```

The population covariance matrix as a labeled DataFrame (skill names on both axes).

**Example**

```python
tax.covariance_df
#              Math   Physics  Chemistry  English  History      Art
# Math       157.25   116.22    113.00   -28.32   -17.81    15.05
# Physics    116.22   164.92    102.45   -29.93   -19.49    17.73
# Chemistry  113.00   102.45    167.00   -31.16   -22.35     8.01
# English    -28.32   -29.93    -31.16    88.05     6.06    -7.24
# History    -17.81   -19.49    -22.35     6.06    84.61    14.88
# Art         15.05    17.73      8.01    -7.24    14.88   249.52
```

---

### `correlation_df`

```python
@property
def correlation_df(self) -> pd.DataFrame
```

The Pearson correlation matrix as a labeled DataFrame.

**Example**

```python
tax.correlation_df.round(2)
#            Math  Physics  Chemistry  English  History   Art
# Math       1.00     0.72       0.70    -0.24    -0.15  0.08
# Physics    0.72     1.00       0.62    -0.25    -0.17  0.09
# Chemistry  0.70     0.62       1.00    -0.26    -0.19  0.04
# English   -0.24    -0.25      -0.26     1.00     0.07 -0.05
# History   -0.15    -0.17      -0.19     0.07     1.00  0.10
# Art        0.08     0.09       0.04    -0.05     0.10  1.00
```

---

## Pretty printing

`print(tax)` gives a summary with condition number, effective dimensionality, and top correlations:

```python
print(tax)
# Taxonomy(200 agents x 6 skills, shrinkage=0.0442)
#   Condition number: 9.8
#   Effective dimensions: ~5 (90% variance)
#
#   Top skill correlations:
#          Math <-> Physics    r = +0.722
#          Math <-> Chemistry  r = +0.697
#       Physics <-> Chemistry  r = +0.617
#     Chemistry <-> English    r = -0.257
#       Physics <-> English    r = -0.248
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `matrix` | `pd.DataFrame` | (N, K) raw data |
| `feature_names` | `list[str]` | Skill names |
| `entity_names` | `list[str]` | Agent names |
| `covariance` | `np.ndarray` | (K, K) covariance matrix (raw array) |
| `correlation` | `np.ndarray` | (K, K) Pearson correlation matrix (raw array) |
| `covariance_df` | `pd.DataFrame` | (K, K) covariance matrix (labeled) |
| `correlation_df` | `pd.DataFrame` | (K, K) correlation matrix (labeled) |
| `population_mean` | `np.ndarray` | (K,) mean across all agents |
| `shrinkage` | `float \| None` | Ledoit-Wolf shrinkage coefficient (`None` if sample covariance) |
