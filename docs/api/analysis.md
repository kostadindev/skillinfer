# analysis

::: skillinfer.analysis

Visualization utilities for exploring taxonomy structure and posterior beliefs. Requires `matplotlib`:

```bash
pip install skillinfer[viz]
```

All functions return a matplotlib `Figure` and accept an optional `ax` parameter to draw on an existing Axes.

---

## `correlation_heatmap`

```python
skillinfer.analysis.correlation_heatmap(
    taxonomy,
    cluster: bool = True,
    figsize: tuple = (12, 10),
    ax=None,
) -> Figure
```

Clustered correlation heatmap of features.

**Parameters**

<div class="param-list" markdown>

`taxonomy`
:   A [`Population`](taxonomy.md) instance.

`cluster`
:   If `True` (default), reorder features by hierarchical clustering (Ward's method). Set to `False` to preserve the original feature order.

`figsize`
:   Figure size as `(width, height)`. Ignored if `ax` is provided.

`ax`
:   Matplotlib Axes to draw on. If `None`, creates a new figure.

</div>

**Returns**

Matplotlib `Figure`.

**Example**

```python
import skillinfer

fig = skillinfer.analysis.correlation_heatmap(pop)
fig.savefig("correlation.png", dpi=150)
```

When `cluster=True`, the heatmap reveals block structure — groups of features that co-vary together appear as warm-colored blocks along the diagonal.

---

## `scree_plot`

```python
skillinfer.analysis.scree_plot(
    taxonomy,
    max_components: int = 30,
    ax=None,
) -> Figure
```

PCA variance explained: individual bars + cumulative line.

**Parameters**

<div class="param-list" markdown>

`taxonomy`
:   A [`Population`](taxonomy.md) instance.

`max_components`
:   Maximum number of principal components to show.

`ax`
:   Matplotlib Axes (optional).

</div>

**Returns**

Matplotlib `Figure`.

**Example**

```python
fig = skillinfer.analysis.scree_plot(pop, max_components=10)
fig.savefig("scree.png", dpi=150)
```

The scree plot shows the effective dimensionality of the feature space. If a few components explain most of the variance, there is strong covariance structure for the Kalman filter to exploit.

---

## `posterior_profile`

```python
skillinfer.analysis.posterior_profile(
    state,
    reference: np.ndarray | None = None,
    top_k: int = 20,
    ax=None,
) -> Figure
```

Horizontal bar chart of posterior mean with error bars, optionally overlaying a reference vector.

**Parameters**

<div class="param-list" markdown>

`state`
:   An [`Profile`](profile.md) instance.

`reference`
:   (K,) reference vector for comparison (e.g., `pop.population_mean` or `pop.entity("some_entity")`). Shown as red scatter points.

`top_k`
:   Number of features to show, sorted by posterior mean.

`ax`
:   Matplotlib Axes (optional).

</div>

**Returns**

Matplotlib `Figure`.

**Example**

```python
fig = skillinfer.analysis.posterior_profile(
    state,
    reference=pop.population_mean,
    top_k=15,
)
fig.savefig("profile.png", dpi=150)
```

The error bars show posterior standard deviation — features with wide bars have high uncertainty and are good candidates for the next observation.

---

## Composing plots

All functions accept an `ax` parameter, so you can compose multiple plots on a single figure:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

skillinfer.analysis.scree_plot(pop, ax=axes[0])
skillinfer.analysis.posterior_profile(state, ax=axes[1])

fig.tight_layout()
fig.savefig("combined.png", dpi=150)
```
