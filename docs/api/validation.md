# Validation

Held-out evaluation, active-learning curves, and uncertainty diagnostics. Use these to quantify how much covariance transfer helps on your data and to compare observation-selection strategies.

```python
import skillinfer
from skillinfer import validation

pop = skillinfer.datasets.onet()
results = validation.held_out_evaluation(pop, frac_observed=0.3, n_splits=10)
results.groupby("method")[["rmse", "pearson_r", "ndcg_at_5", "crps"]].mean()
```

---

## Functions

### `held_out_evaluation`

```python
def held_out_evaluation(
    pop: Population,
    frac_observed: float | list[float] = 0.3,
    n_splits: int = 10,
    obs_noise: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame
```

Hold out 20% of entities, observe a fraction of features, and predict the rest. The covariance and mean are re-estimated from the training 80% so there is no leakage into the test predictions. Compares three methods:

- **`kalman`** — full-covariance Gaussian conditioning (with transfer).
- **`knn`** — k-nearest-neighbour regression in observed-feature space (`k=10`, inverse-distance weighted).
- **`prior`** — population mean (no observations used).

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pop` | `Population` | — | Population providing the prior mean and covariance. |
| `frac_observed` | `float \| list[float]` | `0.3` | Fraction(s) of features to observe per held-out entity. |
| `n_splits` | `int` | `10` | Independent train/test splits. |
| `obs_noise` | `float` | `0.02` | Gaussian noise added to each observation. |
| `seed` | `int` | `42` | RNG seed. |

**Returns:** `DataFrame` with one row per `(split, entity, frac_observed, method)` and columns:

| Column | Description |
|--------|-------------|
| `cosine_similarity` | Directional alignment between predicted and true profile. |
| `rmse`, `mae`, `mse` | Point error on unobserved features. |
| `r_squared` | Coefficient of determination (1.0 = perfect, 0.0 = no better than the mean). |
| `pearson_r` | Pearson correlation between predicted and true profile. Shape agreement up to a linear rescaling. |
| `spearman_rho` | Spearman rank correlation. Captures rank fidelity. |
| `precision_at_5` | `\|top-5 predicted ∩ top-5 true\| / 5`. Top-strengths recovery. |
| `ndcg_at_5` | Normalised Discounted Cumulative Gain at 5, true values as gains. |
| `calibration_coverage` | Fraction of true values inside the posterior 90% CI. ~0.90 is well-calibrated. *Kalman only.* |
| `mean_log_likelihood` | Mean Gaussian log-likelihood under the posterior. *Kalman only.* |
| `crps` | Mean Continuous Ranked Probability Score under the posterior Gaussian. Lower is better; in the same units as the data. *Kalman only.* |

The last three are `NaN` for `knn` and `prior` because they do not produce a posterior covariance.

---

### `active_learning_curve`

```python
def active_learning_curve(
    pop: Population,
    true_vector: np.ndarray,
    n_steps: int = 20,
    strategies: tuple[str, ...] = ("uncertainty", "random"),
    obs_noise: float = 0.05,
    n_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame
```

Compare observation-selection strategies on a single true profile. For each strategy, repeatedly cold-starts a profile, observes a noisy value at the chosen feature index, and records recovery metrics over all features after each step.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pop` | `Population` | — | Provides the prior mean and covariance. |
| `true_vector` | `np.ndarray` | — | Ground-truth profile to recover, shape `(K,)`. |
| `n_steps` | `int` | `20` | Observations per trial. |
| `strategies` | `tuple[str, ...]` | `("uncertainty", "random")` | Any of `"uncertainty"` (pick the highest-std unobserved feature) or `"random"`. |
| `obs_noise` | `float` | `0.05` | Gaussian noise added to each observation. |
| `n_trials` | `int` | `10` | Independent trials per strategy. |
| `seed` | `int` | `42` | Base RNG seed. |

**Returns:** `DataFrame` with columns `[trial, strategy, step, mae, rmse, recovery]`, where `recovery = 1 - ‖μ - true‖² / ‖prior - true‖²`.

**Example**

```python
true = pop.matrix.iloc[0].values
df = validation.active_learning_curve(pop, true, n_steps=20, n_trials=5)
df.groupby(["strategy", "step"])["recovery"].mean().unstack("strategy")
```

!!! note
    Whether `"uncertainty"` beats `"random"` is dataset-dependent: uncertainty sampling chases high-variance dimensions, which on highly correlated populations are not always the highest-leverage ones through the covariance. Run the comparison on your population before assuming a winner.

---

### `transfer_delta`

```python
def transfer_delta(results: pd.DataFrame, metric: str = "cosine_similarity") -> pd.DataFrame
```

Compute the per-`frac_observed` advantage of `kalman` over the `knn` baseline on the chosen metric. Takes the output of `held_out_evaluation` and returns one row per observation fraction with columns `[frac_observed, kalman, baseline, delta]`.

---

### `uncertainty_shrinkage`

```python
def uncertainty_shrinkage(
    state_or_Sigma: Profile | np.ndarray,
    Sigma_0: np.ndarray,
) -> float
```

Posterior uncertainty as a fraction of prior uncertainty: `tr(Σ) / tr(Σ₀)`. A value of `0.5` means uncertainty has halved since the prior; `0.0` means full collapse. Accepts either a `Profile` or a posterior covariance matrix.
