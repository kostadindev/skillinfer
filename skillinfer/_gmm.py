"""GMM-Kalman: sequential Bayesian inference under a Gaussian-mixture prior.

Replaces the single-Gaussian prior of ``_kalman.py`` with a mixture
``p(c) = sum_m pi_m N(mu_m, Sigma_m)``. Each per-skill observation
triggers (i) a per-component Kalman update on (mu_m, Sigma_m) and
(ii) a mixture re-weighting via the per-component marginal predictive
likelihood. The posterior is read out as the mixture marginal mean and
covariance.

Refs: skillinfer chapter, Section 4.5 (Algorithm 2).
"""

from __future__ import annotations

import numpy as np


def fit_gmm(
    R: np.ndarray,
    n_components: int,
    random_state: int | None = None,
    reg_covar: float = 1e-6,
    max_iter: int = 200,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Fit a Gaussian mixture on the entity-feature matrix via EM.

    Parameters
    ----------
    R : (N, K) entity-feature matrix.
    n_components : number of mixture components M.
    random_state : seed for EM initialisation.
    reg_covar : non-negative regulariser added to component covariances
        (sklearn default 1e-6). Bumped if EM struggles on near-singular
        populations.
    max_iter : EM iteration cap.

    Returns
    -------
    means : list of (K,) component means, length M.
    covariances : list of (K, K) component covariances, length M.
    weights : (M,) mixture weights summing to 1.
    """
    from sklearn.mixture import GaussianMixture

    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"R must be 2-D; got shape {R.shape}")
    N, K = R.shape
    if not (1 <= n_components <= N):
        raise ValueError(
            f"n_components must be in [1, {N}], got {n_components}"
        )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        reg_covar=reg_covar,
        max_iter=max_iter,
    )
    gmm.fit(R)
    means = [gmm.means_[m].copy() for m in range(n_components)]
    covariances = [gmm.covariances_[m].copy() for m in range(n_components)]
    weights = gmm.weights_.astype(float).copy()
    return means, covariances, weights


def gmm_condition(
    means: list[np.ndarray],
    covariances: list[np.ndarray],
    weights: np.ndarray,
    obs_indices: np.ndarray,
    obs_values: np.ndarray,
    obs_noise: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Sequential Bayesian conditioning under a Gaussian-mixture prior.

    For each observation y_j on dimension j:
      For each component m:
        - Save current mu_{m,j} and predictive variance v = Sigma_{m,jj} + obs_noise^2.
        - Per-component Kalman update on mu_m, Sigma_m.
        - Multiply log-weight by N(y_j; mu_{m,j}, v) (the marginal
          predictive likelihood under m, evaluated at the *prior* state
          for this observation).
      Renormalise mixture weights.

    Parameters
    ----------
    means : list of (K,) prior component means.
    covariances : list of (K, K) prior component covariances.
    weights : (M,) prior mixture weights.
    obs_indices : (n_obs,) observed feature indices.
    obs_values : (n_obs,) observed values.
    obs_noise : observation noise standard deviation.

    Returns
    -------
    means_post, covariances_post, weights_post.
    """
    M = len(means)
    if len(covariances) != M or len(weights) != M:
        raise ValueError("means / covariances / weights have inconsistent lengths.")

    means_post = [m.copy() for m in means]
    covariances_post = [S.copy() for S in covariances]
    log_weights = np.log(np.maximum(np.asarray(weights, dtype=float), 1e-300))

    obs_indices = np.asarray(obs_indices, dtype=int)
    obs_values = np.asarray(obs_values, dtype=float)
    if obs_indices.size == 0:
        weights_out = np.exp(log_weights - log_weights.max())
        return means_post, covariances_post, weights_out / weights_out.sum()

    sigma2 = float(obs_noise) ** 2

    for j, y in zip(obs_indices, obs_values):
        for m in range(M):
            mu_m = means_post[m]
            S_m = covariances_post[m]

            mu_j_prior = float(mu_m[j])
            var_j_prior = float(S_m[j, j]) + sigma2
            var_j_prior = max(var_j_prior, 1e-12)

            # Mixture re-weighting at the prior state.
            log_lik = -0.5 * (
                np.log(2.0 * np.pi * var_j_prior)
                + (float(y) - mu_j_prior) ** 2 / var_j_prior
            )
            log_weights[m] += log_lik

            # Per-component Kalman update.
            S_col = S_m[:, j].copy()
            K = S_col / var_j_prior
            means_post[m] = mu_m + K * (float(y) - mu_j_prior)
            S_new = S_m - np.outer(K, S_col)
            S_new = (S_new + S_new.T) * 0.5
            np.maximum(
                np.diag(S_new), 1e-10,
                out=S_new[np.diag_indices_from(S_new)],
            )
            covariances_post[m] = S_new

        # Renormalise after each observation to keep weights numerically sane.
        log_weights -= log_weights.max()
        w = np.exp(log_weights)
        w_sum = w.sum()
        if w_sum < 1e-300:
            # All components ruled out by the observation; fall back to uniform.
            w = np.ones(M) / M
        else:
            w = w / w_sum
        log_weights = np.log(np.maximum(w, 1e-300))

    weights_post = np.exp(log_weights - log_weights.max())
    weights_post = weights_post / weights_post.sum()
    return means_post, covariances_post, weights_post


def gmm_marginal(
    means: list[np.ndarray],
    covariances: list[np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mixture marginal mean and covariance.

    mu_hat   = sum_m pi_m mu_m
    Sigma_hat = sum_m pi_m [ Sigma_m + (mu_m - mu_hat)(mu_m - mu_hat)^T ]

    Returns
    -------
    mu_hat : (K,) marginal mean.
    Sigma_hat : (K, K) marginal covariance.
    """
    weights = np.asarray(weights, dtype=float)
    K = means[0].shape[0]

    mu_hat = np.zeros(K)
    for w, m in zip(weights, means):
        mu_hat += w * m

    Sigma_hat = np.zeros((K, K))
    for w, m, S in zip(weights, means, covariances):
        diff = m - mu_hat
        Sigma_hat += w * (S + np.outer(diff, diff))
    Sigma_hat = (Sigma_hat + Sigma_hat.T) * 0.5
    return mu_hat, Sigma_hat
