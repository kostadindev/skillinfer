"""Visualization and analysis utilities.

Matplotlib is imported lazily so the package works without it installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_heatmap(
    taxonomy,
    cluster: bool = True,
    figsize: tuple = (12, 10),
    ax=None,
):
    """Clustered correlation heatmap of features.

    Parameters
    ----------
    taxonomy : Population instance.
    cluster : if True, reorder features by hierarchical clustering.
    figsize : figure size (ignored if ax is provided).
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    Corr = taxonomy.correlation
    names = taxonomy.feature_names

    if cluster:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = 1 - np.abs(Corr)
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        order = leaves_list(Z)
        Corr = Corr[np.ix_(order, order)]
        names = [names[i] for i in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(Corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def scree_plot(
    taxonomy,
    max_components: int = 30,
    ax=None,
):
    """PCA variance explained (bar + cumulative line).

    Parameters
    ----------
    taxonomy : Population instance.
    max_components : maximum number of components to show.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    pca = taxonomy.pca(n_components=max_components)
    evr = pca["explained_variance_ratio"]
    cum = pca["cumulative"]
    n = len(evr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.bar(range(1, n + 1), evr, alpha=0.6, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cum, "o-", color="tab:red", markersize=4, label="Cumulative")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative variance explained")

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained")
    ax.set_title("PCA Scree Plot")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    return fig


def posterior_profile(
    state,
    reference: np.ndarray | None = None,
    top_k: int = 20,
    ax=None,
):
    """Bar chart of posterior mean with error bars, optionally vs. reference.

    Parameters
    ----------
    state : Profile instance.
    reference : (K,) reference vector for comparison (e.g., prior mean).
    top_k : number of features to show (sorted by posterior mean).
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    mu = state.mu
    stds = np.sqrt(np.diag(state.Sigma))
    names = state.feature_names

    idx = np.argsort(mu)[::-1][:top_k]
    idx = idx[::-1]  # reverse for horizontal bar

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    else:
        fig = ax.figure

    y_pos = np.arange(len(idx))
    ax.barh(y_pos, mu[idx], xerr=stds[idx], alpha=0.7, label="Posterior")

    if reference is not None:
        ax.scatter(reference[idx], y_pos, color="tab:red", zorder=5,
                   s=20, label="Reference")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Value")
    ax.set_title(f"Top-{top_k} Features (Posterior Mean)")
    ax.legend()
    fig.tight_layout()
    return fig
