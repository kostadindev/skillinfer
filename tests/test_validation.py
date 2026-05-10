"""Tests for skillinfer.validation metrics and helpers."""

import numpy as np
import pandas as pd
import pytest

import skillinfer
from skillinfer import Population
from skillinfer.validation import (
    _crps_gaussian,
    _ndcg_at_k,
    _pearson_r,
    _precision_at_k,
    _spearman_rho,
    active_learning_curve,
    held_out_evaluation,
)


@pytest.fixture
def pop():
    # Synthetic population with strong block structure: a shared latent
    # factor drives all features, so off-diagonal covariance carries real
    # signal and Kalman transfer can outperform the prior baseline.
    rng = np.random.default_rng(0)
    n, k = 200, 10
    latent = rng.normal(0.5, 0.2, (n, 1))
    loadings = rng.uniform(0.4, 0.9, (1, k))
    noise = rng.normal(0.0, 0.05, (n, k))
    data = np.clip(latent @ loadings + noise + 0.2, 0.0, 1.0)
    cols = [f"f{i}" for i in range(k)]
    return Population.from_dataframe(pd.DataFrame(data, columns=cols))


# ---------------------------------------------------------------------------
# Pearson / Spearman
# ---------------------------------------------------------------------------

def test_pearson_perfect():
    x = np.linspace(0, 1, 20)
    assert _pearson_r(x, 2 * x + 1) == pytest.approx(1.0)


def test_pearson_anti():
    x = np.linspace(0, 1, 20)
    assert _pearson_r(x, -x) == pytest.approx(-1.0)


def test_pearson_constant_is_nan():
    assert np.isnan(_pearson_r(np.ones(10), np.linspace(0, 1, 10)))


def test_spearman_monotone_invariant():
    x = np.linspace(0.1, 1.0, 20)
    # Spearman is rank-based, so any monotone transform of one side is rank=1.
    assert _spearman_rho(x, np.exp(x)) == pytest.approx(1.0)
    assert _spearman_rho(x, np.log(x)) == pytest.approx(1.0)


def test_spearman_constant_is_nan():
    assert np.isnan(_spearman_rho(np.zeros(10), np.linspace(0, 1, 10)))


# ---------------------------------------------------------------------------
# Top-k metrics
# ---------------------------------------------------------------------------

def test_precision_at_k_perfect():
    true = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.6])
    pred = true.copy()
    assert _precision_at_k(pred, true, k=3) == 1.0


def test_precision_at_k_zero():
    # Predicted top-3 disjoint from true top-3.
    true = np.array([1.0, 0.9, 0.8, 0.0, 0.0, 0.0])
    pred = np.array([0.0, 0.0, 0.0, 1.0, 0.9, 0.8])
    assert _precision_at_k(pred, true, k=3) == 0.0


def test_precision_at_k_clamps_to_n():
    # Fewer than k items: should still return a finite value, not crash.
    true = np.array([0.5, 0.9])
    pred = np.array([0.5, 0.9])
    assert _precision_at_k(pred, true, k=5) == 1.0


def test_ndcg_perfect_ranking_is_one():
    true = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
    assert _ndcg_at_k(true, true, k=5) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_is_lower():
    true = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
    rev = -true  # rank exactly inverted
    assert _ndcg_at_k(rev, true, k=5) < 1.0


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def test_crps_zero_when_perfect_and_certain():
    # Tiny variance + truth at the mean → CRPS ≈ 0.
    crps = _crps_gaussian(
        true=np.array([0.5]),
        mu=np.array([0.5]),
        var=np.array([1e-10]),
    )
    assert crps == pytest.approx(0.0, abs=1e-3)


def test_crps_increases_with_error():
    mu = np.array([0.5])
    var = np.array([0.1])
    near = _crps_gaussian(np.array([0.5]), mu, var)
    far = _crps_gaussian(np.array([2.0]), mu, var)
    assert far > near


def test_crps_increases_with_variance_when_off():
    # When the truth is far from the mean, a wider posterior is *better*
    # (less surprise), so CRPS should drop. Standard property of CRPS.
    true = np.array([2.0])
    mu = np.array([0.0])
    tight = _crps_gaussian(true, mu, np.array([0.01]))
    wide = _crps_gaussian(true, mu, np.array([1.0]))
    assert wide < tight


# ---------------------------------------------------------------------------
# held_out_evaluation columns and behaviour
# ---------------------------------------------------------------------------

def test_held_out_evaluation_returns_new_columns(pop):
    df = held_out_evaluation(pop, frac_observed=0.3, n_splits=2, seed=0)
    expected = {
        "cosine_similarity", "rmse", "mae", "mse", "r_squared",
        "calibration_coverage", "mean_log_likelihood", "crps",
        "pearson_r", "spearman_rho", "precision_at_5", "ndcg_at_5",
    }
    assert expected.issubset(df.columns)


def test_held_out_baselines_have_nan_for_posterior_metrics(pop):
    df = held_out_evaluation(pop, frac_observed=0.3, n_splits=2, seed=0)
    for method in ("knn", "prior"):
        rows = df[df["method"] == method]
        assert rows["calibration_coverage"].isna().all()
        assert rows["mean_log_likelihood"].isna().all()
        assert rows["crps"].isna().all()
        # Point-prediction metrics are still computed for baselines.
        assert rows["pearson_r"].notna().any()
        assert rows["precision_at_5"].notna().any()


def test_kalman_beats_prior_on_rmse(pop):
    df = held_out_evaluation(pop, frac_observed=0.5, n_splits=3, seed=0)
    means = df.groupby("method")["rmse"].mean()
    assert means.loc["kalman"] < means.loc["prior"]


def test_correlation_metrics_in_valid_ranges(pop):
    df = held_out_evaluation(pop, frac_observed=0.3, n_splits=2, seed=0)
    finite = df.dropna(subset=["pearson_r", "spearman_rho"])
    assert ((finite["pearson_r"] >= -1.0) & (finite["pearson_r"] <= 1.0)).all()
    assert ((finite["spearman_rho"] >= -1.0) & (finite["spearman_rho"] <= 1.0)).all()
    assert ((df["precision_at_5"].dropna() >= 0.0) & (df["precision_at_5"].dropna() <= 1.0)).all()
    ndcg = df["ndcg_at_5"].dropna()
    assert ((ndcg >= 0.0) & (ndcg <= 1.0)).all()


# ---------------------------------------------------------------------------
# active_learning_curve
# ---------------------------------------------------------------------------

def test_active_learning_curve_columns(pop):
    true = pop.matrix.iloc[0].values
    df = active_learning_curve(pop, true, n_steps=5, n_trials=2, seed=0)
    assert set(df.columns) == {"trial", "strategy", "step", "mae", "rmse", "recovery"}
    assert set(df["strategy"]) == {"uncertainty", "random"}
    assert df["step"].max() == 5
    # Two trials × two strategies × five steps.
    assert len(df) == 2 * 2 * 5


def test_active_learning_recovery_rises(pop):
    true = pop.matrix.iloc[0].values
    df = active_learning_curve(
        pop, true, n_steps=8, n_trials=4, strategies=("random",), seed=0,
    )
    by_step = df.groupby("step")["recovery"].mean()
    # Mean recovery at the last step should exceed the first.
    assert by_step.loc[8] > by_step.loc[1]


def test_active_learning_rejects_mismatched_truth(pop):
    bad = np.zeros(pop.matrix.shape[1] + 1)
    with pytest.raises(ValueError, match="length"):
        active_learning_curve(pop, bad, n_steps=2, n_trials=1)


def test_active_learning_rejects_population_mean(pop):
    with pytest.raises(ValueError, match="undefined"):
        active_learning_curve(pop, pop.population_mean, n_steps=2, n_trials=1)


def test_active_learning_unknown_strategy(pop):
    true = pop.matrix.iloc[0].values
    with pytest.raises(ValueError, match="Unknown strategy"):
        active_learning_curve(
            pop, true, n_steps=2, n_trials=1, strategies=("ouija",),
        )
