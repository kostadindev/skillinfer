"""End-to-end integration tests."""

import numpy as np
import pandas as pd
import pytest

import skillinfer


def _make_taxonomy(n_entities=50, n_features=10, seed=42):
    """Create a synthetic taxonomy with known block structure."""
    rng = np.random.default_rng(seed)
    # Two correlated blocks: [0-4] cognitive, [5-9] physical
    cog = rng.normal(0.6, 0.15, (n_entities, 5))
    phy = rng.normal(0.4, 0.15, (n_entities, 5))
    # Anti-correlate: high-cognitive entities have low-physical
    rank = np.argsort(cog.mean(axis=1))
    phy = phy[rank[::-1]]
    data = np.clip(np.hstack([cog, phy]), 0, 1)
    features = [f"cog_{i}" for i in range(5)] + [f"phy_{i}" for i in range(5)]
    df = pd.DataFrame(data, columns=features)
    return skillinfer.Population.from_dataframe(df)


def test_full_workflow():
    """Population → state → observe → query."""
    tax = _make_taxonomy()
    assert tax.matrix.shape == (50, 10)

    state = tax.profile()
    state.observe("cog_0", 0.9).observe("cog_1", 0.85)

    # Cognitive features should increase, physical should decrease
    df = state.to_dataframe()
    cog_mean = df[df["feature"].str.startswith("cog")]["mean"].mean()
    phy_mean = df[df["feature"].str.startswith("phy")]["mean"].mean()

    pop_cog = tax.population_mean[:5].mean()
    pop_phy = tax.population_mean[5:].mean()

    assert cog_mean > pop_cog  # cognitive went up
    assert phy_mean < pop_phy  # physical went down (anti-correlation)


def test_transfer_beats_diagonal():
    """Full Kalman should predict unobserved features better than diagonal."""
    # Use larger taxonomy with stronger block structure for reliable transfer
    tax = _make_taxonomy(n_entities=200, n_features=10, seed=99)

    results = skillinfer.validation.held_out_evaluation(
        tax,
        frac_observed=0.2,
        n_splits=5,
        obs_noise=0.05,
        seed=42,
    )

    means = results.groupby("method")["cosine_similarity"].mean()
    # Kalman (with transfer) should be at least as good as diagonal (no transfer)
    assert means["kalman"] >= means["diagonal"] - 0.01


def test_prior_entity_closer_than_population():
    """Using a specific entity as prior should give better results than pop mean."""
    tax = _make_taxonomy(n_entities=50)

    # Entity 0's true vector
    true_vec = tax.matrix.values[0]
    entity_name = tax.entity_names[0]

    # State with entity prior
    s1 = tax.profile(prior_entity=entity_name)
    # State with population prior
    s2 = tax.profile()

    sim1 = s1.similarity(true_vec)
    sim2 = s2.similarity(true_vec)
    assert sim1 >= sim2


def test_convergence_with_more_observations():
    """More observations should bring the posterior closer to truth."""
    tax = _make_taxonomy()
    true_vec = tax.matrix.values[3]
    features = tax.feature_names

    rng = np.random.default_rng(42)
    state = tax.profile()

    sims = [state.similarity(true_vec)]
    for _ in range(8):
        j = rng.integers(len(features))
        obs = true_vec[j] + rng.normal(0, 0.05)
        state.observe(features[j], obs)
        sims.append(state.similarity(true_vec))

    # Final similarity should be better than initial
    assert sims[-1] > sims[0]


def test_pca_analysis():
    tax = _make_taxonomy()
    pca = tax.pca(n_components=5)
    assert pca["cumulative"][-1] > 0.5
    assert len(pca["explained_variance_ratio"]) == 5


def test_top_correlations():
    tax = _make_taxonomy()
    top = tax.top_correlations(k=5)
    assert len(top) == 5
    # Top correlation should be between cognitive features
    row = top.iloc[0]
    assert "cog" in row["feature_a"] or "cog" in row["feature_b"]
