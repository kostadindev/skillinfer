"""Visualization utilities for skillinfer.

Matplotlib is imported lazily so the package works without it installed.
Install with: pip install skillinfer[viz]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# -- Shared styling --------------------------------------------------------

_PALETTE = {
    "primary": "#4361ee",
    "secondary": "#f72585",
    "accent": "#4cc9f0",
    "muted": "#adb5bd",
    "positive": "#4361ee",
    "negative": "#f72585",
    "bg": "#fafafa",
}


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent styling to an axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _make_fig(ax, figsize):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.figure
    return fig, ax


# -- Population charts -----------------------------------------------------


def correlation_heatmap(
    pop,
    cluster: bool = True,
    figsize: tuple = (11, 9),
    ax=None,
):
    """Clustered correlation heatmap of features.

    Parameters
    ----------
    pop : Population instance.
    cluster : if True, reorder features by hierarchical clustering.
    figsize : figure size (ignored if ax is provided).
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    Corr = pop.correlation
    names = pop.feature_names

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

    fig, ax = _make_fig(ax, figsize)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(Corr, cmap="RdBu_r", norm=norm, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    K = len(names)
    fontsize = max(3, min(7, 120 // K))
    ax.set_xticklabels(names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(names, fontsize=fontsize)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Correlation", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Feature Correlation Matrix", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def scree_plot(
    pop,
    max_components: int = 30,
    ax=None,
):
    """PCA variance explained (bar + cumulative line).

    Parameters
    ----------
    pop : Population instance.
    max_components : maximum number of components to show.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    pca = pop.pca(n_components=max_components)
    evr = pca["explained_variance_ratio"]
    cum = pca["cumulative"]
    n = len(evr)
    x = range(1, n + 1)

    fig, ax = _make_fig(ax, (9, 4.5))

    ax.bar(x, evr, color=_PALETTE["primary"], alpha=0.7, width=0.7,
           label="Individual", zorder=2)
    ax2 = ax.twinx()
    ax2.plot(x, cum, "o-", color=_PALETTE["secondary"], markersize=5,
             linewidth=2, label="Cumulative", zorder=3)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Cumulative variance", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=9)

    # Mark 90% line
    ax2.axhline(0.9, color=_PALETTE["muted"], linestyle="--", linewidth=1,
                alpha=0.7, zorder=1)
    eff = next((i + 1 for i, c in enumerate(cum) if c > 0.9), n)
    ax2.annotate(f"90% at PC{eff}", xy=(eff, 0.9), fontsize=8,
                 color=_PALETTE["muted"], ha="left", va="bottom",
                 xytext=(eff + 0.5, 0.92))

    _style_ax(ax, "PCA Scree Plot", "Principal component", "Variance explained")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right",
              fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def feature_distributions(
    pop,
    top_k: int = 20,
    ax=None,
):
    """Box plots showing the distribution of top-k features across entities.

    Parameters
    ----------
    pop : Population instance.
    top_k : number of features to show (sorted by variance).
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    variances = np.var(pop.matrix.values, axis=0)
    idx = np.argsort(variances)[::-1][:top_k]
    names = [pop.feature_names[i] for i in idx]
    data = [pop.matrix.iloc[:, i].values for i in idx]

    fig, ax = _make_fig(ax, (10, max(4, top_k * 0.3)))

    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color=_PALETTE["secondary"], linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(_PALETTE["primary"])
        patch.set_alpha(0.5)

    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names, fontsize=8)
    _style_ax(ax, f"Feature Distributions (top {top_k} by variance)",
              xlabel="Value")
    fig.tight_layout()
    return fig


# -- Profile charts --------------------------------------------------------


def posterior_profile(
    profile,
    reference: np.ndarray | None = None,
    top_k: int = 20,
    ax=None,
):
    """Bar chart of posterior mean with uncertainty, optionally vs. reference.

    Parameters
    ----------
    profile : Profile instance.
    reference : (K,) reference vector for comparison (e.g., ground truth).
    top_k : number of features to show (sorted by posterior mean).
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    mu = profile.mu
    stds = np.sqrt(np.diag(profile.Sigma))
    names = profile.feature_names
    observed = profile._observed

    idx = np.argsort(mu)[::-1][:top_k]
    idx = idx[::-1]  # reverse for horizontal bar (top at top)

    fig, ax = _make_fig(ax, (9, max(4, top_k * 0.35)))

    y_pos = np.arange(len(idx))
    colors = [_PALETTE["secondary"] if i in observed else _PALETTE["primary"]
              for i in idx]

    ax.barh(y_pos, mu[idx], xerr=stds[idx], alpha=0.7, color=colors,
            capsize=2, error_kw={"linewidth": 1, "alpha": 0.6},
            zorder=2, height=0.7)

    if reference is not None:
        ax.scatter(reference[idx], y_pos, color=_PALETTE["secondary"],
                   zorder=5, s=30, marker="d", label="Ground truth",
                   edgecolors="white", linewidths=0.5)

    ax.set_yticks(y_pos)
    labels = []
    for i in idx:
        name = names[i]
        tag = " ●" if i in observed else ""
        labels.append(f"{name}{tag}")
    ax.set_yticklabels(labels, fontsize=8)

    _style_ax(ax, f"Skill Profile (top {top_k})", xlabel="Predicted value")

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_PALETTE["primary"], alpha=0.7, label="Predicted"),
        Patch(facecolor=_PALETTE["secondary"], alpha=0.7, label="Observed"),
    ]
    if reference is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="d", color="w",
                              markerfacecolor=_PALETTE["secondary"],
                              markersize=7, label="Ground truth"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def prediction_scatter(
    profile,
    true_vector: np.ndarray,
    ax=None,
):
    """Scatter plot of predicted vs. true values for all features.

    Points along the diagonal are accurate predictions. Error bars show
    posterior uncertainty. Observed features cluster tightly on the diagonal;
    predicted features spread out.

    Parameters
    ----------
    profile : Profile instance.
    true_vector : (K,) ground truth vector.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    true_vector = np.asarray(true_vector, dtype=float)
    mu = profile.mu
    stds = np.sqrt(np.diag(profile.Sigma))
    observed = profile._observed

    fig, ax = _make_fig(ax, (6, 6))

    # Diagonal line
    lo = min(true_vector.min(), mu.min()) - 0.05
    hi = max(true_vector.max(), mu.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color=_PALETTE["muted"], linewidth=1,
            zorder=1)

    # Predicted features
    pred_mask = np.array([i not in observed for i in range(len(mu))])
    ax.errorbar(true_vector[pred_mask], mu[pred_mask],
                yerr=stds[pred_mask], fmt="o", color=_PALETTE["primary"],
                alpha=0.6, markersize=5, capsize=2, linewidth=1,
                label="Predicted", zorder=2)

    # Observed features
    obs_mask = ~pred_mask
    if obs_mask.any():
        ax.scatter(true_vector[obs_mask], mu[obs_mask],
                   color=_PALETTE["secondary"], s=50, marker="d",
                   zorder=3, label="Observed", edgecolors="white",
                   linewidths=0.5)

    _style_ax(ax, "Predicted vs. True", "True value", "Predicted value")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def uncertainty_waterfall(
    pop,
    observations: dict[str, float],
    ax=None,
):
    """Show how average uncertainty decreases with each observation.

    Plots the mean posterior std after 0, 1, 2, ... observations,
    revealing the diminishing returns of additional observations.

    Parameters
    ----------
    pop : Population instance.
    observations : ordered dict of {feature: value} to observe sequentially.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    profile = pop.profile()
    steps = ["Prior"]
    mean_stds = [float(np.sqrt(np.diag(profile.Sigma)).mean())]

    for feat, val in observations.items():
        profile.observe(feat, val)
        steps.append(feat)
        mean_stds.append(float(np.sqrt(np.diag(profile.Sigma)).mean()))

    fig, ax = _make_fig(ax, (max(6, len(steps) * 1.2), 4.5))

    x = range(len(steps))
    ax.bar(x, mean_stds, color=_PALETTE["primary"], alpha=0.7, width=0.6,
           zorder=2)
    ax.plot(x, mean_stds, "o-", color=_PALETTE["secondary"], markersize=7,
            linewidth=2, zorder=3)

    # Annotate reduction
    for i in range(1, len(steps)):
        pct = (mean_stds[i - 1] - mean_stds[i]) / mean_stds[0] * 100
        ax.annotate(f"−{pct:.1f}%", xy=(i, mean_stds[i]),
                    fontsize=7, ha="center", va="bottom",
                    color=_PALETTE["secondary"], fontweight="bold",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
    _style_ax(ax, "Uncertainty Reduction per Observation",
              ylabel="Mean posterior std")
    fig.tight_layout()
    return fig


def compare_profiles(
    profiles: dict[str, "Profile"],
    features: list[str] | None = None,
    top_k: int = 10,
    ax=None,
):
    """Side-by-side comparison of multiple profiles on selected features.

    Parameters
    ----------
    profiles : dict mapping names to Profile instances.
    features : features to compare. If None, uses top_k by variance
        across the profiles.
    top_k : number of features if features is not specified.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    names = list(profiles.keys())
    profile_list = list(profiles.values())

    if features is None:
        # Pick features with highest variance across profiles
        all_mu = np.stack([p.mu for p in profile_list])
        var_across = np.var(all_mu, axis=0)
        feat_idx = np.argsort(var_across)[::-1][:top_k]
        features = [profile_list[0].feature_names[i] for i in feat_idx]

    n_features = len(features)
    n_profiles = len(names)

    fig, ax = _make_fig(ax, (10, max(4, n_features * 0.4)))

    y_pos = np.arange(n_features)
    bar_height = 0.7 / n_profiles
    colors = [_PALETTE["primary"], _PALETTE["secondary"], _PALETTE["accent"],
              "#7209b7", "#f77f00"]

    for i, (name, profile) in enumerate(zip(names, profile_list)):
        means = []
        errs = []
        for feat in features:
            pred = profile.predict(feat)
            means.append(pred["mean"])
            errs.append(pred["std"])
        offset = (i - n_profiles / 2 + 0.5) * bar_height
        ax.barh(y_pos + offset, means, xerr=errs, height=bar_height,
                alpha=0.7, color=colors[i % len(colors)], label=name,
                capsize=1.5, error_kw={"linewidth": 0.8}, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    _style_ax(ax, "Profile Comparison", xlabel="Predicted value")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def skill_embedding(
    pop,
    ax=None,
):
    """2D PCA embedding of features, colored by category prefix.

    Features are projected into the first two principal component
    directions of the covariance matrix. Each point is a feature
    (not an entity). Category prefixes like ``Skill:``, ``Knowledge:``,
    ``Ability:`` are used for coloring.

    Parameters
    ----------
    pop : Population instance.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    # Each feature is a row: transpose so (K, N)
    coords = pca.fit_transform(pop.matrix.values.T)
    ev = pca.explained_variance_ratio_

    fig, ax = _make_fig(ax, (10, 8))

    # Group by category prefix
    categories = {}
    for i, name in enumerate(pop.feature_names):
        cat = name.split(":")[0] if ":" in name else "Other"
        categories.setdefault(cat, []).append(i)

    cat_colors = {
        "Skill": _PALETTE["primary"],
        "Knowledge": "#2a9d8f",
        "Ability": _PALETTE["secondary"],
    }

    for cat, indices in sorted(categories.items()):
        color = cat_colors.get(cat, _PALETTE["muted"])
        idx = np.array(indices)
        ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=40,
                   alpha=0.7, label=cat, edgecolors="white", linewidths=0.4,
                   zorder=2)

    # Label a subset of points (most extreme on each axis)
    n_labels = min(12, len(pop.feature_names))
    label_idx = set()
    for dim in [0, 1]:
        ranked = np.argsort(coords[:, dim])
        label_idx.update(ranked[:n_labels // 4])
        label_idx.update(ranked[-n_labels // 4:])

    for i in label_idx:
        name = pop.feature_names[i]
        short = name.split(":")[-1] if ":" in name else name
        ax.annotate(short, (coords[i, 0], coords[i, 1]),
                    fontsize=6, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")

    _style_ax(ax, "Skill Embedding (2D PCA)",
              f"PC1 ({ev[0]:.1%} var)", f"PC2 ({ev[1]:.1%} var)")
    ax.legend(fontsize=9, framealpha=0.9, markerscale=1.2)
    ax.axhline(0, color=_PALETTE["muted"], linewidth=0.5, alpha=0.3)
    ax.axvline(0, color=_PALETTE["muted"], linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    return fig


def convergence_curve(
    pop,
    n_entities: int = 20,
    max_observations: int | None = None,
    seed: int = 42,
    ax=None,
):
    """Prediction quality vs. number of observations.

    Holds out ``n_entities``, then for each observation count (1, 2, ...),
    randomly observes that many features and measures MAE on the rest.
    Compares the Kalman filter (full covariance) against the diagonal
    baseline (no transfer).

    Parameters
    ----------
    pop : Population instance.
    n_entities : number of entities to hold out and average over.
    max_observations : maximum observations to plot (default: K-1).
    seed : random seed.
    ax : matplotlib Axes (optional).

    Returns
    -------
    matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    K = len(pop.feature_names)
    if max_observations is None:
        max_observations = min(K - 1, 15)

    entity_idx = rng.choice(len(pop.entity_names), size=min(n_entities, len(pop.entity_names)),
                            replace=False)
    obs_counts = list(range(1, max_observations + 1))

    kalman_maes = {k: [] for k in obs_counts}
    diag_maes = {k: [] for k in obs_counts}

    for ei in entity_idx:
        true_vec = pop.matrix.iloc[ei].values
        perm = rng.permutation(K)
        for n_obs in obs_counts:
            obs_idx = perm[:n_obs]

            # Kalman (full covariance)
            profile = pop.profile()
            for j in obs_idx:
                profile.observe(int(j), float(true_vec[j]))
            unobs = [j for j in range(K) if j not in obs_idx]
            errs = np.abs(profile.mu[unobs] - true_vec[unobs])
            kalman_maes[n_obs].append(float(errs.mean()))

            # Diagonal baseline
            diag_mu = pop.population_mean.copy()
            diag_mu[obs_idx] = true_vec[obs_idx]
            errs_d = np.abs(diag_mu[unobs] - true_vec[unobs])
            diag_maes[n_obs].append(float(errs_d.mean()))

    k_mean = [np.mean(kalman_maes[k]) for k in obs_counts]
    k_std = [np.std(kalman_maes[k]) / np.sqrt(len(kalman_maes[k])) for k in obs_counts]
    d_mean = [np.mean(diag_maes[k]) for k in obs_counts]
    d_std = [np.std(diag_maes[k]) / np.sqrt(len(diag_maes[k])) for k in obs_counts]

    fig, ax = _make_fig(ax, (8, 5))

    ax.fill_between(obs_counts,
                    [m - s for m, s in zip(k_mean, k_std)],
                    [m + s for m, s in zip(k_mean, k_std)],
                    alpha=0.15, color=_PALETTE["primary"])
    ax.plot(obs_counts, k_mean, "o-", color=_PALETTE["primary"],
            markersize=5, linewidth=2, label="Kalman (transfer)", zorder=3)

    ax.fill_between(obs_counts,
                    [m - s for m, s in zip(d_mean, d_std)],
                    [m + s for m, s in zip(d_mean, d_std)],
                    alpha=0.15, color=_PALETTE["muted"])
    ax.plot(obs_counts, d_mean, "s--", color=_PALETTE["muted"],
            markersize=4, linewidth=1.5, label="Diagonal (no transfer)", zorder=2)

    _style_ax(ax, "Prediction Quality vs. Observations",
              "Number of observations", "MAE on unobserved features")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig
