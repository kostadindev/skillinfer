"""Tests for core Kalman update math."""

import numpy as np
import pytest

from bayeskal._kalman import kalman_update, kalman_update_batch, diagonal_update


def test_kalman_update_reduces_variance():
    """Observing a feature should reduce its posterior variance."""
    K = 5
    mu = np.zeros(K)
    Sigma = np.eye(K) * 0.1
    mu_new, Sigma_new = kalman_update(mu, Sigma, j=2, y_j=0.8, obs_noise=0.05)

    assert Sigma_new[2, 2] < Sigma[2, 2]
    assert np.isclose(mu_new[2], 0.8, atol=0.05)


def test_kalman_update_transfers_via_covariance():
    """Off-diagonal covariance should propagate updates to unobserved features."""
    K = 3
    mu = np.array([0.5, 0.5, 0.5])
    # Feature 0 and 1 are positively correlated
    Sigma = np.array([
        [0.10, 0.08, 0.00],
        [0.08, 0.10, 0.00],
        [0.00, 0.00, 0.10],
    ])

    mu_new, _ = kalman_update(mu, Sigma, j=0, y_j=0.9, obs_noise=0.05)

    # Feature 0 should move toward 0.9
    assert mu_new[0] > 0.5
    # Feature 1 should also increase (positive correlation)
    assert mu_new[1] > 0.5
    # Feature 2 should stay the same (zero correlation)
    assert np.isclose(mu_new[2], 0.5, atol=1e-10)


def test_kalman_update_anticorrelation():
    """Negative covariance should push unobserved features in opposite direction."""
    K = 2
    mu = np.array([0.5, 0.5])
    Sigma = np.array([
        [0.10, -0.05],
        [-0.05, 0.10],
    ])

    mu_new, _ = kalman_update(mu, Sigma, j=0, y_j=0.9, obs_noise=0.05)

    assert mu_new[0] > 0.5  # observed, moves up
    assert mu_new[1] < 0.5  # anti-correlated, moves down


def test_kalman_update_preserves_symmetry():
    """Covariance should remain symmetric after update."""
    K = 10
    rng = np.random.default_rng(42)
    A = rng.normal(size=(K, K))
    Sigma = A @ A.T + np.eye(K) * 0.1
    mu = rng.normal(size=K)

    _, Sigma_new = kalman_update(mu, Sigma, j=3, y_j=0.5, obs_noise=0.1)

    np.testing.assert_allclose(Sigma_new, Sigma_new.T, atol=1e-12)


def test_kalman_update_positive_diagonal():
    """Diagonal of covariance should remain positive."""
    K = 5
    mu = np.zeros(K)
    Sigma = np.eye(K) * 0.01  # small variance

    for _ in range(50):
        mu, Sigma = kalman_update(mu, Sigma, j=0, y_j=0.5, obs_noise=0.01)

    assert np.all(np.diag(Sigma) > 0)


def test_kalman_batch_equals_sequential():
    """Batch update should match sequential single updates."""
    K = 5
    rng = np.random.default_rng(123)
    A = rng.normal(size=(K, K))
    Sigma = A @ A.T + np.eye(K) * 0.1
    mu = rng.normal(size=K)

    obs_idx = np.array([0, 2, 4])
    obs_val = np.array([0.3, 0.7, 0.1])

    # Batch
    mu_b, Sigma_b = kalman_update_batch(mu, Sigma, obs_idx, obs_val, 0.05)

    # Sequential
    mu_s, Sigma_s = mu.copy(), Sigma.copy()
    for j, y in zip(obs_idx, obs_val):
        mu_s, Sigma_s = kalman_update(mu_s, Sigma_s, j, y, 0.05)

    np.testing.assert_allclose(mu_b, mu_s, atol=1e-12)
    np.testing.assert_allclose(Sigma_b, Sigma_s, atol=1e-12)


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


def test_does_not_mutate_inputs():
    """Update functions should not modify input arrays."""
    K = 3
    mu = np.array([0.5, 0.5, 0.5])
    Sigma = np.eye(K) * 0.1
    mu_orig = mu.copy()
    Sigma_orig = Sigma.copy()

    kalman_update(mu, Sigma, j=0, y_j=0.9, obs_noise=0.05)

    np.testing.assert_array_equal(mu, mu_orig)
    np.testing.assert_array_equal(Sigma, Sigma_orig)
