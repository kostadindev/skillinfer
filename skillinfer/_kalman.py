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


def diagonal_covariance(pop_cov: np.ndarray) -> np.ndarray:
    """Zero out all off-diagonal entries of pop_cov.

    Used by the Diagonal ablation: each observation only updates the
    observed feature; no cross-feature transfer.
    """
    return np.diag(np.diag(pop_cov)).astype(float, copy=True)


def block_diagonal_covariance(
    pop_cov: np.ndarray,
    blocks: list[list[int]],
) -> np.ndarray:
    """Restrict pop_cov to within-block entries; zero across blocks.

    Used by the Block-diagonal ablation: covariance transfer is allowed
    inside each block (e.g. within Skills, within Knowledge) but
    zeroed across blocks. Indices not assigned to any block become
    diagonal-only.

    Parameters
    ----------
    pop_cov : (K, K) population covariance.
    blocks : list of index lists. Each inner list is one block.

    Returns
    -------
    Sigma : (K, K) restricted covariance.
    """
    K = pop_cov.shape[0]
    mask = np.eye(K, dtype=bool)
    for block in blocks:
        idx = np.asarray(list(block), dtype=int)
        if idx.size == 0:
            continue
        if idx.min() < 0 or idx.max() >= K:
            raise ValueError(
                f"Block index out of range for K={K}: {idx.tolist()}"
            )
        mask[np.ix_(idx, idx)] = True
    Sigma = np.where(mask, pop_cov, 0.0).astype(float, copy=True)
    Sigma = (Sigma + Sigma.T) * 0.5
    return Sigma


def low_rank_covariance(
    pop_cov: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Rank-r eigentruncation of pop_cov (PMF / probabilistic PCA prior).

    Computes the eigendecomposition of pop_cov and keeps only the top
    ``rank`` components: Sigma_r = V_r Lambda_r V_r^T. This is the
    population prior used by PMF rank-r in Section 4.4.1 of the
    skillinfer chapter. Variance on directions outside the top-r
    eigenspace collapses to zero, which is the structural reason PMF
    intervals are narrower than full-Sigma Kalman.

    Parameters
    ----------
    pop_cov : (K, K) symmetric positive-(semi-)definite covariance.
    rank : number of eigencomponents to retain (1 <= rank <= K).

    Returns
    -------
    Sigma_r : (K, K) rank-``rank`` truncation, symmetric.
    """
    K = pop_cov.shape[0]
    if not (1 <= rank <= K):
        raise ValueError(f"rank must be in [1, {K}], got {rank}")

    Sigma_sym = (pop_cov + pop_cov.T) * 0.5
    # eigh returns eigenvalues in ascending order; take the top-rank tail.
    eigvals, eigvecs = np.linalg.eigh(Sigma_sym)
    eigvals = np.maximum(eigvals, 0.0)
    V_r = eigvecs[:, -rank:]
    L_r = eigvals[-rank:]
    Sigma_r = (V_r * L_r) @ V_r.T
    Sigma_r = (Sigma_r + Sigma_r.T) * 0.5
    return Sigma_r
