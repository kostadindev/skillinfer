"""Tests for core Kalman conditioning math."""

import numpy as np
import pytest

from skillinfer._kalman import (
    block_diagonal_covariance,
    condition,
    diagonal_covariance,
    diagonal_update,
    low_rank_covariance,
    posterior_covariance,
)


def test_condition_moves_toward_observed():
    """Observing a feature should move posterior mean toward the observed value."""
    K = 5
    mu = np.ones(K) * 0.5
    Sigma = np.eye(K) * 0.1

    mu_new = condition(mu, Sigma, np.array([2]), np.array([0.8]), obs_noise=0.05)

    assert abs(mu_new[2] - 0.8) < abs(mu[2] - 0.8)


def test_condition_transfers_via_covariance():
    """Off-diagonal covariance should propagate updates to unobserved features."""
    K = 3
    mu = np.array([0.5, 0.5, 0.5])
    Sigma = np.array([
        [0.10, 0.08, 0.00],
        [0.08, 0.10, 0.00],
        [0.00, 0.00, 0.10],
    ])

    mu_new = condition(mu, Sigma, np.array([0]), np.array([0.9]), obs_noise=0.05)

    # Feature 0 should move toward 0.9
    assert mu_new[0] > 0.5
    # Feature 1 should also increase (positive correlation)
    assert mu_new[1] > 0.5
    # Feature 2 should stay the same (zero correlation)
    assert np.isclose(mu_new[2], 0.5, atol=1e-10)


def test_condition_anticorrelation():
    """Negative covariance should push unobserved features in opposite direction."""
    K = 2
    mu = np.array([0.5, 0.5])
    Sigma = np.array([
        [0.10, -0.05],
        [-0.05, 0.10],
    ])

    mu_new = condition(mu, Sigma, np.array([0]), np.array([0.9]), obs_noise=0.05)

    assert mu_new[0] > 0.5  # observed, moves up
    assert mu_new[1] < 0.5  # anti-correlated, moves down


def test_condition_batch_order_independent():
    """Batch conditioning should give the same result regardless of observation order."""
    K = 5
    rng = np.random.default_rng(123)
    A = rng.normal(size=(K, K))
    Sigma = A @ A.T + np.eye(K) * 0.1
    mu = np.clip(rng.normal(0.5, 0.1, K), 0, 1)

    obs_idx = np.array([0, 2, 4])
    obs_val = np.array([0.3, 0.7, 0.1])

    # Order 1
    mu1 = condition(mu, Sigma, obs_idx, obs_val, 0.05)

    # Reversed order
    mu2 = condition(mu, Sigma, obs_idx[::-1], obs_val[::-1], 0.05)

    np.testing.assert_allclose(mu1, mu2, atol=1e-12)


def test_posterior_covariance_reduces_variance():
    """Observing a feature should reduce its posterior variance."""
    K = 5
    Sigma = np.eye(K) * 0.1

    Sigma_post = posterior_covariance(Sigma, np.array([2]), obs_noise=0.05)

    assert Sigma_post[2, 2] < Sigma[2, 2]


def test_posterior_covariance_symmetric():
    """Posterior covariance should remain symmetric."""
    K = 10
    rng = np.random.default_rng(42)
    A = rng.normal(size=(K, K))
    Sigma = A @ A.T + np.eye(K) * 0.1

    Sigma_post = posterior_covariance(Sigma, np.array([3, 7]), obs_noise=0.1)

    np.testing.assert_allclose(Sigma_post, Sigma_post.T, atol=1e-12)


def test_posterior_covariance_positive_diagonal():
    """Diagonal of posterior covariance should remain positive."""
    K = 5
    Sigma = np.eye(K) * 0.01

    Sigma_post = posterior_covariance(Sigma, np.arange(K), obs_noise=0.01)

    assert np.all(np.diag(Sigma_post) > 0)


def test_population_covariance_not_mutated():
    """Conditioning should not modify the population covariance."""
    K = 3
    mu = np.array([0.5, 0.5, 0.5])
    Sigma = np.eye(K) * 0.1
    Sigma_orig = Sigma.copy()
    mu_orig = mu.copy()

    condition(mu, Sigma, np.array([0]), np.array([0.9]), obs_noise=0.05)
    posterior_covariance(Sigma, np.array([0]), obs_noise=0.05)

    np.testing.assert_array_equal(Sigma, Sigma_orig)
    np.testing.assert_array_equal(mu, mu_orig)


def test_no_observations_returns_prior():
    """With no observations, posterior should equal the prior."""
    K = 5
    mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    Sigma = np.eye(K) * 0.1

    mu_post = condition(mu, Sigma, np.array([], dtype=int), np.array([]), 0.05)
    Sigma_post = posterior_covariance(Sigma, np.array([], dtype=int), 0.05)

    np.testing.assert_array_equal(mu_post, mu)
    np.testing.assert_array_equal(Sigma_post, Sigma)


def test_diagonal_update_no_transfer():
    """Diagonal update should only change the observed feature."""
    K = 5
    mu = np.zeros(K)
    var = np.ones(K) * 0.1

    mu_new, var_new = diagonal_update(mu, var, j=2, y_j=0.8, obs_noise=0.05)

    assert mu_new[2] != mu[2]
    assert var_new[2] < var[2]
    # All other features unchanged
    for i in [0, 1, 3, 4]:
        assert mu_new[i] == mu[i]
        assert var_new[i] == var[i]


def test_diagonal_covariance_zeros_off_diagonal():
    Sigma = np.array([
        [0.10, 0.08, 0.02],
        [0.08, 0.10, 0.03],
        [0.02, 0.03, 0.10],
    ])
    D = diagonal_covariance(Sigma)
    np.testing.assert_array_equal(np.diag(D), np.diag(Sigma))
    off = D - np.diag(np.diag(D))
    assert np.all(off == 0)
    # Original is not mutated.
    assert Sigma[0, 1] == 0.08


def test_block_diagonal_covariance_keeps_within_blocks():
    Sigma = np.array([
        [0.10, 0.08, 0.02, 0.01],
        [0.08, 0.10, 0.03, 0.04],
        [0.02, 0.03, 0.10, 0.07],
        [0.01, 0.04, 0.07, 0.10],
    ])
    B = block_diagonal_covariance(Sigma, [[0, 1], [2, 3]])
    # Within-block entries preserved.
    assert B[0, 1] == 0.08
    assert B[2, 3] == 0.07
    # Across-block entries zeroed.
    assert B[0, 2] == 0.0
    assert B[1, 3] == 0.0
    # Symmetric.
    np.testing.assert_allclose(B, B.T)


def test_block_diagonal_unassigned_index_is_diagonal():
    Sigma = np.array([
        [0.10, 0.08, 0.02],
        [0.08, 0.10, 0.03],
        [0.02, 0.03, 0.10],
    ])
    # Index 2 is not in any block — only its diagonal survives.
    B = block_diagonal_covariance(Sigma, [[0, 1]])
    assert B[2, 2] == 0.10
    assert B[0, 2] == 0.0
    assert B[1, 2] == 0.0


def test_block_diagonal_rejects_out_of_range():
    Sigma = np.eye(3) * 0.1
    with pytest.raises(ValueError, match="out of range"):
        block_diagonal_covariance(Sigma, [[0, 5]])


def test_low_rank_covariance_rank_full_recovers_input():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    Sigma = A @ A.T + np.eye(6) * 0.1
    Sigma_r = low_rank_covariance(Sigma, rank=6)
    np.testing.assert_allclose(Sigma_r, Sigma, atol=1e-10)


def test_low_rank_covariance_truncates_spectrum():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 8))
    Sigma = A @ A.T + np.eye(8) * 0.05
    rank = 3
    Sigma_r = low_rank_covariance(Sigma, rank=rank)
    eigvals = np.linalg.eigvalsh(Sigma_r)
    # Exactly `rank` eigenvalues should be non-trivial; the rest near zero.
    nonzero = np.sum(eigvals > 1e-8)
    assert nonzero == rank
    # Symmetric.
    np.testing.assert_allclose(Sigma_r, Sigma_r.T, atol=1e-12)


def test_low_rank_covariance_rejects_bad_rank():
    Sigma = np.eye(4) * 0.1
    with pytest.raises(ValueError):
        low_rank_covariance(Sigma, rank=0)
    with pytest.raises(ValueError):
        low_rank_covariance(Sigma, rank=5)
