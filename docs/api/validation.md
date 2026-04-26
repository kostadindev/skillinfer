# validation

::: skillinfer.validation

Held-out evaluation utilities for measuring whether covariance transfer actually improves predictions.

---

## `held_out_evaluation`

```python
skillinfer.validation.held_out_evaluation(
    taxonomy,
    frac_observed: float | list[float] = 0.3,
    n_splits: int = 10,
    obs_noise: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame
```

Hold out entities, observe a fraction of their features (with noise), and predict the rest. Compares three methods:

| Method | Description |
|--------|-------------|
| **`kalman`** | Full covariance Kalman filter — uses off-diagonal covariance to transfer information between features |
| **`diagonal`** | Diagonal-only update — only the observed feature is updated, no cross-feature transfer |
| **`prior`** | Population mean baseline — ignores all observations |

The difference between `kalman` and `diagonal` measures **how much value the covariance structure adds**. If they perform similarly, features are approximately independent.

**Parameters**

<div class="param-list" markdown>

`taxonomy`
:   A [`Taxonomy`](taxonomy.md) instance.

`frac_observed`
:   Fraction of features to observe for each held-out entity. Can be a single float or a list of floats to compare multiple observation budgets.

`n_splits`
:   Number of random train/test splits. Each split holds out ~20% of entities.

`obs_noise`
:   Observation noise standard deviation. Added to the true feature values when creating synthetic observations.

`seed`
:   Random seed for reproducibility.

</div>

**Returns**

DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `split` | int | Split index (0 to n_splits-1) |
| `entity` | str | Entity name |
| `frac_observed` | float | Fraction of features that were observed |
| `method` | str | `"kalman"`, `"diagonal"`, or `"prior"` |
| `cosine_similarity` | float | Cosine similarity between predicted and true feature vectors (on unobserved features only) |
| `mse` | float | Mean squared error on unobserved features |

**Example**

```python
import skillinfer

results = skillinfer.validation.held_out_evaluation(
    tax,
    frac_observed=[0.1, 0.2, 0.3, 0.5],
    n_splits=10,
    obs_noise=0.02,
)

# Summary statistics
summary = results.groupby(["frac_observed", "method"])["cosine_similarity"].mean()
print(summary)
```

```text
frac_observed  method
0.1            diagonal    0.921
               kalman      0.947  ← with transfer
               prior       0.889
0.2            diagonal    0.953
               kalman      0.968
               prior       0.889
0.3            diagonal    0.964
               kalman      0.981
               prior       0.889
0.5            diagonal    0.983
               kalman      0.992
               prior       0.889
```

**Interpreting results**

- `kalman > diagonal` → covariance transfer helps; features are correlated
- `kalman ≈ diagonal` → features are approximately independent; no benefit from transfer
- The gap is largest when `frac_observed` is small — transfer is most valuable when you have few observations
- Both methods converge as `frac_observed → 1.0` (observing everything leaves nothing to predict)

**Detailed analysis**

The returned DataFrame has per-entity results, so you can analyze which entities benefit most from transfer:

```python
# Which entities benefit most from transfer?
kalman = results[results["method"] == "kalman"].set_index(["split", "entity", "frac_observed"])
diag = results[results["method"] == "diagonal"].set_index(["split", "entity", "frac_observed"])

improvement = kalman["cosine_similarity"] - diag["cosine_similarity"]
print(improvement.describe())
```
