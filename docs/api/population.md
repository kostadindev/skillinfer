# Population

The population model. Wraps an entity-feature matrix and its learned covariance structure.

```python
import skillinfer

pop = skillinfer.Population.from_dataframe(df)
```

---

## Constructors

### `Population.from_dataframe`

```python
@classmethod
def from_dataframe(
    cls,
    df: pd.DataFrame,
    normalize: bool = True,
    covariance: str = "ledoit-wolf",
) -> Population
```

Build a Population from a pandas DataFrame.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Rows = entities, columns = features. All values must be numeric. No NaN values. |
| `normalize` | `bool` | `True` | Scale each column to [0, 1]. Set to `False` if data is already on a meaningful scale (e.g., binary 0/1 data, or scores where the raw values matter). |
| `covariance` | `str` | `"ledoit-wolf"` | `"ledoit-wolf"` (recommended) or `"sample"`. |

**Returns:** `Population`

---

### `Population.from_csv`

```python
@classmethod
def from_csv(
    cls,
    path: str,
    index_col: int | str = 0,
    normalize: bool = True,
    covariance: str = "ledoit-wolf",
) -> Population
```

Build a Population from a CSV file. Convenience wrapper around `from_dataframe`.

---

### `Population.from_parquet`

```python
@classmethod
def from_parquet(
    cls,
    path: str,
    normalize: bool = True,
    covariance: str = "ledoit-wolf",
) -> Population
```

Build a Population from a Parquet file. Same parameters as `from_dataframe`.

---

### `Population.from_covariance`

```python
@classmethod
def from_covariance(
    cls,
    covariance: np.ndarray,
    feature_names: list[str],
    population_mean: np.ndarray,
) -> Population
```

Build a Population from a pre-computed covariance matrix. Use this when you have a domain-expert covariance (e.g., from a previous study) rather than estimating from data.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `covariance` | `np.ndarray` | (K, K) covariance matrix. |
| `feature_names` | `list[str]` | List of K feature names. |
| `population_mean` | `np.ndarray` | (K,) mean vector. |

---

## Core Methods

### `profile`

```python
def profile(
    self,
    prior_entity: str | None = None,
    prior_mean: np.ndarray | None = None,
    noise: float | None = None,
    method: str = "kalman",
    rank: int | None = None,
    blocks: list[list[str | int]] | dict[str, str | int] | None = None,
    n_components: int | None = None,
    gmm_random_state: int | None = 0,
) -> Profile
```

Create a [`Profile`](profile.md) for a new entity. Inherits skill descriptions from the Population.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_entity` | `str \| None` | `None` | Use this entity's vector as the prior mean. |
| `prior_mean` | `np.ndarray \| None` | `None` | Use this array directly as the prior mean. |
| `noise` | `float \| None` | `None` | Observation noise (std dev). Default is 5% of average feature spread. |
| `method` | `str` | `"kalman"` | Inference method. One of `"kalman"`, `"diagonal"`, `"block-diagonal"`, `"pmf"`, `"gmm-kalman"`. |
| `rank` | `int \| None` | `None` | Top-r eigencomponents to retain when `method="pmf"`. |
| `blocks` | `list[list] \| dict \| None` | `None` | Block specification for `method="block-diagonal"`: either a list of feature lists or a `{feature: block_label}` dict. |
| `n_components` | `int \| None` | `None` | Mixture size M when `method="gmm-kalman"`. |
| `gmm_random_state` | `int \| None` | `0` | Seed for the EM fit (cached per population). |

If neither `prior_entity` nor `prior_mean` is given, the population mean is used.

**Methods**

- **`"kalman"`** (default) — Full Ledoit–Wolf covariance, propagates evidence to all features.
- **`"diagonal"`** — Off-diagonal entries zeroed; only the observed feature updates (no-transfer ablation).
- **`"block-diagonal"`** — Covariance kept inside each block, zeroed across blocks. Useful for restricting transfer to within Skills, within Knowledge, etc.
- **`"pmf"`** — Rank-`rank` eigentruncation of the covariance (PMF / probabilistic PCA prior). Cheap and serves as a strong linear baseline; variance on directions outside the top-r subspace collapses to zero.
- **`"gmm-kalman"`** — Gaussian-mixture prior fit on this population by EM. Each observation triggers per-component Kalman updates plus mixture re-weighting; this is the only non-linear option (a surprising observation can flip the dominant cluster). Returns a `GMMProfile`.

**Returns:** [`Profile`](profile.md)

**Example**

```python
profile = pop.profile()                                      # population mean prior
profile = pop.profile(prior_entity="Software Developers")    # entity-specific prior
profile = pop.profile(noise=0.1)                             # custom noise level

# Alternative inference methods
profile = pop.profile(method="diagonal")                     # no-transfer baseline
profile = pop.profile(method="pmf", rank=20)                 # rank-20 PMF prior
profile = pop.profile(
    method="block-diagonal",
    blocks=[skill_names, knowledge_names, ability_names],
)                                                            # within-category transfer only
profile = pop.profile(method="gmm-kalman", n_components=10)  # mixture-of-Gaussians prior
```

---

### `entity`

```python
def entity(self, name: str) -> np.ndarray
```

Get the raw feature vector for a named entity. Returns a copy as a numpy array of shape `(K,)`.

---

### `skill_vector`

```python
def skill_vector(self, name: str) -> pd.Series
```

Get a named entity's feature vector as a labeled `pd.Series`.

---

### `describe_skills`

```python
def describe_skills(self, descriptions: dict[str, str] | list[Skill]) -> None
```

Attach descriptions to skill dimensions. Descriptions flow through to Profiles, appearing in `predict()` output.

**Example**

```python
pop.describe_skills({
    "BBH": "Big-Bench Hard: diverse challenging tasks",
    "MMLU-PRO": "Professional-level multitask understanding",
})
```

---

## Analysis Methods

### `summary`

```python
def summary(self) -> dict
```

Summary statistics for this population.

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `n_entities` | `int` | Number of entities (rows) |
| `n_features` | `int` | Number of features (columns) |
| `shrinkage` | `float \| None` | Ledoit-Wolf coefficient |
| `condition_number` | `float` | Covariance matrix condition number |
| `effective_dimensions` | `int` | PCA components needed for 90% variance |
| `mean_correlation` | `float` | Average absolute off-diagonal correlation |
| `sparsity` | `float` | Fraction of \|correlations\| below 0.1 |
| `top_correlations` | `list[dict]` | Top 5 feature pairs by \|correlation\| |

---

### `top_correlations`

```python
def top_correlations(self, k: int = 20) -> pd.DataFrame
```

Top-k strongest feature-feature correlations by absolute value. Returns DataFrame with columns `[feature_a, feature_b, correlation]`.

---

### `pca`

```python
def pca(self, n_components: int = 15) -> dict
```

PCA decomposition. Returns dict with keys: `components`, `explained_variance_ratio`, `cumulative`.

---

### `condition_number`

```python
def condition_number(self) -> float
```

Condition number of the covariance matrix ($\lambda_{\max} / \lambda_{\min}$). Lower = more stable.

---

## Export

### `to_dataframe`

```python
def to_dataframe(self) -> pd.DataFrame
```

The entity-feature matrix as a DataFrame (copy).

---

### `to_csv`

```python
def to_csv(self, path: str) -> None
```

Export the entity-feature matrix to CSV. Re-import with `Population.from_csv(path)`.

---

### `to_parquet`

```python
def to_parquet(self, path: str) -> None
```

Export the entity-feature matrix to Parquet. Re-import with `Population.from_parquet(path)`.

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `covariance_df` | `pd.DataFrame` | (K, K) covariance matrix with feature names |
| `correlation_df` | `pd.DataFrame` | (K, K) correlation matrix with feature names |
| `skills` | `list[Skill]` | Skill objects with descriptions |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `matrix` | `pd.DataFrame` | (N, K) raw data |
| `feature_names` | `list[str]` | Feature/skill names |
| `entity_names` | `list[str]` | Entity names |
| `covariance` | `np.ndarray` | (K, K) covariance matrix |
| `correlation` | `np.ndarray` | (K, K) correlation matrix |
| `population_mean` | `np.ndarray` | (K,) mean across all entities |
| `shrinkage` | `float \| None` | Ledoit-Wolf shrinkage coefficient |
