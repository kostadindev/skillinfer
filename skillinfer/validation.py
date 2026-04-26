"""Held-out evaluation: does covariance transfer help?

Metrics reported by held_out_evaluation():

From the thesis:
  - cosine_similarity: directional alignment between predicted and true
    profiles on unobserved features (primary metric in all experiments).
  - rmse: root mean squared error on unobserved features; in percentage
    points when using raw (unnormalised) benchmark scores.
  - mae: mean absolute error on unobserved features.
  - mse: mean squared error on unobserved features.

Beyond the thesis:
  - r_squared: coefficient of determination on unobserved features.
    1.0 = perfect prediction, 0.0 = no better than the mean, <0 = worse.
  - calibration_coverage: fraction of unobserved features whose true
    value falls within the posterior 90% confidence interval. A well-
    calibrated model should score ~0.90. Values below indicate
    overconfidence; values above indicate underconfidence.
  - mean_log_likelihood: average log-likelihood of unobserved true values
    under the posterior Gaussian (proper scoring rule that rewards both
    accuracy and calibrated uncertainty). Only computed for the kalman
    method (diagonal and prior lack per-feature posterior variances from
    the full covariance update).
"""

import numpy as np
import pandas as pd

from skillinfer._kalman import kalman_update, diagonal_update


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    """Coefficient of determination (R²) on unobserved features."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _calibration_coverage(
    true: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    level: float = 0.90,
) -> float:
    """Fraction of true values inside the posterior CI at the given level."""
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)
    std = np.sqrt(np.maximum(var, 1e-15))
    inside = np.abs(true - mu) <= z * std
    return float(inside.mean())


def _mean_log_likelihood(
    true: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> float:
    """Mean Gaussian log-likelihood of true values under the posterior."""
    var = np.maximum(var, 1e-15)
    ll = -0.5 * (np.log(2 * np.pi * var) + (true - mu) ** 2 / var)
    return float(ll.mean())


def held_out_evaluation(
    pop,
    frac_observed: float | list[float] = 0.3,
    n_splits: int = 10,
    obs_noise: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Hold out entities, observe a fraction of features, predict the rest.

    For each held-out entity, a random subset of features is "observed"
    (with noise), and the remaining features are predicted. Compares:
      - **kalman**: full-covariance Kalman filter (with transfer)
      - **diagonal**: diagonal-only update (no transfer)
      - **prior**: population mean (no observations used)

    Parameters
    ----------
    pop : Population instance.
    frac_observed : fraction(s) of features to observe (scalar or list).
    n_splits : number of random train/test splits.
    obs_noise : observation noise standard deviation.
    seed : random seed.

    Returns
    -------
    DataFrame with columns:
        [split, entity, frac_observed, method,
         cosine_similarity, rmse, mae, mse, r_squared,
         calibration_coverage, mean_log_likelihood]

    The last two columns (calibration_coverage, mean_log_likelihood) are
    NaN for the diagonal and prior methods, which lack full posterior
    covariance on unobserved features.
    """
    if isinstance(frac_observed, (int, float)):
        frac_observed = [frac_observed]

    R = pop.matrix.values
    N, K = R.shape
    Sigma = pop.covariance
    pop_mean = pop.population_mean
    entity_names = pop.entity_names

    rng = np.random.default_rng(seed)
    rows = []

    for split in range(n_splits):
        # Hold out ~20% of entities
        perm = rng.permutation(N)
        n_test = max(1, N // 5)
        test_idx = perm[:n_test]

        for frac in frac_observed:
            n_obs = max(1, int(K * frac))

            for ei in test_idx:
                true_vec = R[ei]
                obs_dims = rng.choice(K, size=n_obs, replace=False)
                obs_vals = true_vec[obs_dims] + rng.normal(0, obs_noise, n_obs)
                unobs_dims = np.setdiff1d(np.arange(K), obs_dims)

                if len(unobs_dims) == 0:
                    continue

                # Method 1: Full Kalman (transfer)
                mu_k = pop_mean.copy()
                Sigma_k = Sigma.copy()
                for j, y in zip(obs_dims, obs_vals):
                    mu_k, Sigma_k = kalman_update(mu_k, Sigma_k, int(j), float(y), obs_noise)

                # Method 2: Diagonal (no transfer)
                mu_d = pop_mean.copy()
                var_d = np.diag(Sigma).copy()
                for j, y in zip(obs_dims, obs_vals):
                    mu_d, var_d = diagonal_update(mu_d, var_d, int(j), float(y), obs_noise)

                # Evaluate on unobserved features
                true = true_vec[unobs_dims]

                # Kalman: has full posterior covariance
                pred_k = mu_k[unobs_dims]
                var_k = np.diag(Sigma_k)[unobs_dims]
                mse_k = float(np.mean((pred_k - true) ** 2))
                rows.append({
                    "split": split,
                    "entity": entity_names[ei],
                    "frac_observed": frac,
                    "method": "kalman",
                    "cosine_similarity": _cosine_sim(pred_k, true),
                    "rmse": float(np.sqrt(mse_k)),
                    "mae": float(np.mean(np.abs(pred_k - true))),
                    "mse": mse_k,
                    "r_squared": _r_squared(pred_k, true),
                    "calibration_coverage": _calibration_coverage(true, pred_k, var_k),
                    "mean_log_likelihood": _mean_log_likelihood(true, pred_k, var_k),
                })

                # Diagonal and prior: no full posterior covariance
                for method, mu_pred in [("diagonal", mu_d), ("prior", pop_mean)]:
                    pred = mu_pred[unobs_dims]
                    mse_val = float(np.mean((pred - true) ** 2))
                    rows.append({
                        "split": split,
                        "entity": entity_names[ei],
                        "frac_observed": frac,
                        "method": method,
                        "cosine_similarity": _cosine_sim(pred, true),
                        "rmse": float(np.sqrt(mse_val)),
                        "mae": float(np.mean(np.abs(pred - true))),
                        "mse": mse_val,
                        "r_squared": _r_squared(pred, true),
                        "calibration_coverage": np.nan,
                        "mean_log_likelihood": np.nan,
                    })

    return pd.DataFrame(rows)


def uncertainty_shrinkage(state_or_Sigma, Sigma_0: np.ndarray) -> float:
    """Posterior uncertainty as a fraction of prior uncertainty.

    Computes tr(Sigma_k) / tr(Sigma_0), measuring how much total
    uncertainty remains after observations. A value of 0.5 means
    uncertainty has halved; 0.2 means 80% of initial uncertainty
    has been eliminated.

    This metric is used in the thesis (Figure 12) to determine when
    enough observations have been collected to transition oversight
    levels: high values warrant more human oversight, low values
    permit more autonomy.

    Parameters
    ----------
    state_or_Sigma : Profile or (K, K) ndarray.
        The current posterior covariance. If an Profile, its
        .Sigma attribute is used.
    Sigma_0 : (K, K) ndarray.
        The prior covariance (e.g., pop.covariance).

    Returns
    -------
    float : tr(Sigma_k) / tr(Sigma_0), in [0, 1].
    """
    if hasattr(state_or_Sigma, "Sigma"):
        Sigma_k = state_or_Sigma.Sigma
    else:
        Sigma_k = state_or_Sigma

    tr_0 = np.trace(Sigma_0)
    if tr_0 < 1e-15:
        return 0.0
    return float(np.trace(Sigma_k) / tr_0)


def transfer_delta(results: pd.DataFrame, metric: str = "cosine_similarity") -> pd.DataFrame:
    """Compute the Kalman advantage over the diagonal baseline.

    Takes the output of held_out_evaluation() and returns the
    per-frac_observed difference: kalman - diagonal.

    Parameters
    ----------
    results : DataFrame from held_out_evaluation().
    metric : column name to compare (default: "cosine_similarity").

    Returns
    -------
    DataFrame with columns [frac_observed, kalman, diagonal, delta].
    """
    means = results.groupby(["frac_observed", "method"])[metric].mean().unstack()
    out = pd.DataFrame({
        "frac_observed": means.index,
        "kalman": means["kalman"].values,
        "diagonal": means["diagonal"].values,
    })
    out["delta"] = out["kalman"] - out["diagonal"]
    return out.reset_index(drop=True)
