"""Core Kalman conditioning math (pure numpy, no classes).

The population covariance is always used as a read-only reference.
Observations condition the mean and covariance via the Gaussian
conditioning formula — no sequential mutation of the covariance.
"""

import numpy as np


def condition(
    prior_mean: np.ndarray,
    pop_cov: np.ndarray,
    obs_indices: np.ndarray,
    obs_values: np.ndarray,
    obs_noise: float,
) -> np.ndarray:
    """Compute posterior mean by conditioning on observed features.

    Uses the Gaussian conditioning formula with the population covariance
    (which is not modified). Each observation propagates to all features
    via the learned covariance structure.

    Parameters
    ----------
    prior_mean : (K,) prior mean vector (population mean, entity, or custom).
    pop_cov : (K, K) population covariance matrix (read-only).
    obs_indices : (n_obs,) indices of observed features.
    obs_values : (n_obs,) observed values.
    obs_noise : observation noise standard deviation.

    Returns
    -------
    mu_post : (K,) posterior mean.
    """
    if len(obs_indices) == 0:
        return prior_mean.copy()

    J = np.asarray(obs_indices, dtype=int)
    y = np.asarray(obs_values, dtype=float)

    S_J = pop_cov[:, J]                           # (K, n_obs)
    S_JJ = pop_cov[np.ix_(J, J)]                  # (n_obs, n_obs)
    M = S_JJ + obs_noise ** 2 * np.eye(len(J))    # (n_obs, n_obs)

    innovation = y - prior_mean[J]                 # (n_obs,)
    alpha = np.linalg.solve(M, innovation)         # (n_obs,)
    delta = S_J @ alpha                            # (K,)

    mu = prior_mean + delta
    return mu


def posterior_covariance(
    pop_cov: np.ndarray,
    obs_indices: np.ndarray,
    obs_noise: float,
) -> np.ndarray:
    """Compute posterior covariance after conditioning on observed features.

    Parameters
    ----------
    pop_cov : (K, K) population covariance matrix (read-only).
    obs_indices : (n_obs,) indices of observed features.
    obs_noise : observation noise standard deviation.

    Returns
    -------
    Sigma_post : (K, K) posterior covariance matrix.
    """
    if len(obs_indices) == 0:
        return pop_cov.copy()

    J = np.asarray(obs_indices, dtype=int)
    S_J = pop_cov[:, J]                           # (K, n_obs)
    S_JJ = pop_cov[np.ix_(J, J)]                  # (n_obs, n_obs)
    M = S_JJ + obs_noise ** 2 * np.eye(len(J))

    beta = np.linalg.solve(M, S_J.T)              # (n_obs, K)
    Sigma_post = pop_cov - S_J @ beta              # (K, K)

    # Numerical stability: enforce symmetry and positive diagonal.
    Sigma_post = (Sigma_post + Sigma_post.T) * 0.5
    np.maximum(np.diag(Sigma_post), 1e-10,
               out=Sigma_post[np.diag_indices_from(Sigma_post)])

    return Sigma_post


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
