"""Core Kalman filter update math (pure numpy, no classes)."""

import numpy as np


def kalman_update(
    mu: np.ndarray,
    Sigma: np.ndarray,
    j: int,
    y_j: float,
    obs_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-feature Kalman update.

    Observing feature j with value y_j updates the full mean vector and
    covariance matrix via the off-diagonal covariance structure.

    Parameters
    ----------
    mu : (K,) posterior mean
    Sigma : (K, K) posterior covariance
    j : index of observed feature
    y_j : observed value
    obs_noise : observation noise standard deviation

    Returns
    -------
    mu_new, Sigma_new : updated mean and covariance (copies)
    """
    mu = mu.copy()
    Sigma = Sigma.copy()

    S_j = Sigma[:, j]
    denom = max(Sigma[j, j] + obs_noise ** 2, 1e-8)
    K_gain = S_j / denom
    mu += K_gain * (y_j - mu[j])
    Sigma -= np.outer(K_gain, S_j)

    # Numerical stability: enforce symmetry and positive diagonal
    Sigma = (Sigma + Sigma.T) * 0.5
    np.maximum(np.diag(Sigma), 1e-10, out=Sigma[np.diag_indices_from(Sigma)])

    return mu, Sigma


def kalman_update_batch(
    mu: np.ndarray,
    Sigma: np.ndarray,
    obs_indices: np.ndarray,
    obs_values: np.ndarray,
    obs_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential Kalman updates for multiple observations.

    Parameters
    ----------
    mu : (K,) posterior mean
    Sigma : (K, K) posterior covariance
    obs_indices : (n_obs,) which features were observed
    obs_values : (n_obs,) observed values
    obs_noise : observation noise standard deviation

    Returns
    -------
    mu_new, Sigma_new : updated mean and covariance (copies)
    """
    mu = mu.copy()
    Sigma = Sigma.copy()

    for j, y_j in zip(obs_indices, obs_values):
        S_j = Sigma[:, j]
        denom = max(Sigma[j, j] + obs_noise ** 2, 1e-8)
        K_gain = S_j / denom
        mu += K_gain * (y_j - mu[j])
        Sigma -= np.outer(K_gain, S_j)

    # Numerical stability
    Sigma = (Sigma + Sigma.T) * 0.5
    np.maximum(np.diag(Sigma), 1e-10, out=Sigma[np.diag_indices_from(Sigma)])

    return mu, Sigma


def diagonal_update(
    mu: np.ndarray,
    var: np.ndarray,
    j: int,
    y_j: float,
    obs_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal-only Kalman update (no cross-feature transfer).

    Only the observed feature j is updated; all other features remain
    at their prior values. This is the no-transfer baseline.

    Parameters
    ----------
    mu : (K,) posterior mean
    var : (K,) diagonal variances
    j : index of observed feature
    y_j : observed value
    obs_noise : observation noise standard deviation

    Returns
    -------
    mu_new, var_new : updated mean and variance (copies)
    """
    mu = mu.copy()
    var = var.copy()

    gain = var[j] / (var[j] + obs_noise ** 2)
    mu[j] += gain * (y_j - mu[j])
    var[j] *= (1 - gain)

    return mu, var
