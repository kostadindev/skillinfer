"""Held-out evaluation: does covariance transfer help on this data?

Hold out a fraction of entities, observe a fraction of features per
held-out entity, predict the rest, and compare the full Kalman update
against a kNN baseline and the population mean. The output DataFrame
lets you check whether the off-diagonal covariance is buying anything
on your population before relying on it for routing or matching.

Metrics reported by held_out_evaluation():

  - cosine_similarity: directional alignment between predicted and true
    profiles on unobserved features.
  - rmse: root mean squared error on unobserved features.
  - mae: mean absolute error on unobserved features.
  - mse: mean squared error on unobserved features.
  - r_squared: coefficient of determination on unobserved features.
    1.0 = perfect prediction, 0.0 = no better than the mean, <0 = worse.
  - calibration_coverage: fraction of unobserved features whose true
    value falls within the posterior 90% confidence interval. A well-
    calibrated model should score ~0.90. Values below indicate
    overconfidence; values above indicate underconfidence.
  - mean_log_likelihood: average log-likelihood of unobserved true values
    under the posterior Gaussian. Proper scoring rule that rewards both
    accuracy and calibrated uncertainty. Kalman method only.
  - crps: mean Continuous Ranked Probability Score under the posterior
    Gaussian. Lower is better; in the same units as the data. Like MLL
    it is a proper scoring rule but is bounded and well-defined when
    the truth lies near the tails of the posterior. Kalman method only.
  - pearson_r: Pearson correlation between predicted and true profiles
    over the unobserved features. Captures shape agreement up to a
    linear rescaling.
  - spearman_rho: Spearman rank correlation between predicted and true
    profiles over the unobserved features. Captures rank fidelity.
  - precision_at_5: |top-5 predicted ∩ top-5 true| / 5 over the
    unobserved features. Top-strengths recovery for routing.
  - ndcg_at_5: Normalised Discounted Cumulative Gain at 5, with the
    true value used as the gain for each ranked feature. 1.0 = perfect
    top-5 ranking; ~0.0 = pessimal.
"""

import numpy as np
import pandas as pd

from skillinfer._kalman import condition, posterior_covariance


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


def _pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation; NaN if either side is constant."""
    if pred.size < 2:
        return float("nan")
    sp, st = pred.std(), true.std()
    if sp < 1e-15 or st < 1e-15:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def _spearman_rho(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation; NaN if either side is constant."""
    if pred.size < 2:
        return float("nan")
    if pred.std() < 1e-15 or true.std() < 1e-15:
        return float("nan")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(pred, true)
    return float(rho) if np.isfinite(rho) else float("nan")


def _precision_at_k(pred: np.ndarray, true: np.ndarray, k: int = 5) -> float:
    """|top-k by prediction ∩ top-k by truth| / k."""
    n = pred.size
    if n == 0:
        return float("nan")
    kk = min(k, n)
    top_pred = set(np.argpartition(-pred, kk - 1)[:kk].tolist())
    top_true = set(np.argpartition(-true, kk - 1)[:kk].tolist())
    return len(top_pred & top_true) / kk


def _ndcg_at_k(pred: np.ndarray, true: np.ndarray, k: int = 5) -> float:
    """NDCG@k using the true values as gains. NaN if all gains are zero."""
    n = pred.size
    if n == 0:
        return float("nan")
    kk = min(k, n)
    order = np.argsort(-pred)[:kk]
    gains = true[order]
    ideal = np.sort(true)[::-1][:kk]
    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = float(np.dot(gains, discounts))
    idcg = float(np.dot(ideal, discounts))
    if abs(idcg) < 1e-15:
        return float("nan")
    return dcg / idcg


def _crps_gaussian(
    true: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> float:
    """Mean CRPS of true values under a Gaussian posterior (closed form).

    For x ~ N(mu, sigma^2), CRPS = sigma * [z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]
    with z = (x - mu)/sigma. Lower is better.
    """
    from scipy.stats import norm

    sigma = np.sqrt(np.maximum(var, 1e-15))
    z = (true - mu) / sigma
    crps = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(crps.mean())


def held_out_evaluation(
    pop,
    frac_observed: float | list[float] = 0.3,
    n_splits: int = 10,
    obs_noise: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Hold out entities, observe a fraction of features, predict the rest.

    For each split, 20% of entities are held out. The covariance and mean
    are re-estimated from the remaining 80% (training set only), ensuring
    no data leakage into the test predictions.

    Compares three methods:
      - **kalman**: full-covariance Gaussian conditioning (with transfer)
      - **knn**: k-nearest-neighbor regression in observed-feature space
        (k=10, inverse-distance weighted — non-parametric baseline)
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
    NaN for the knn and prior methods (only kalman produces a posterior
    covariance).
    """
    from skillinfer._covariance import ledoit_wolf_covariance

    if isinstance(frac_observed, (int, float)):
        frac_observed = [frac_observed]

    R = pop.matrix.values
    N, K = R.shape
    entity_names = pop.entity_names

    rng = np.random.default_rng(seed)
    rows = []

    for split in range(n_splits):
        # Hold out ~20% of entities
        perm = rng.permutation(N)
        n_test = max(1, N // 5)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        # Re-estimate covariance and mean from TRAINING entities only
        R_train = R[train_idx]
        train_mean = R_train.mean(axis=0)
        train_cov, _ = ledoit_wolf_covariance(R_train)

        for frac in frac_observed:
            n_obs = max(1, int(K * frac))

            for ei in test_idx:
                true_vec = R[ei]
                obs_dims = rng.choice(K, size=n_obs, replace=False)
                obs_vals = true_vec[obs_dims] + rng.normal(0, obs_noise, n_obs)
                unobs_dims = np.setdiff1d(np.arange(K), obs_dims)

                if len(unobs_dims) == 0:
                    continue

                # Method 1: Full conditioning (transfer)
                mu_k = condition(
                    train_mean, train_cov, obs_dims, obs_vals, obs_noise,
                )
                Sigma_k = posterior_covariance(train_cov, obs_dims, obs_noise)

                # Method 2: k-NN (k=10, inverse-distance weighted)
                dists = np.linalg.norm(
                    R_train[:, obs_dims] - obs_vals, axis=1,
                )
                k_neighbors = min(10, len(R_train))
                topk = np.argpartition(dists, k_neighbors)[:k_neighbors]
                weights = 1.0 / (dists[topk] + 1e-8)
                weights /= weights.sum()
                mu_knn = (R_train[topk] * weights[:, None]).sum(axis=0)

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
                    "crps": _crps_gaussian(true, pred_k, var_k),
                    "pearson_r": _pearson_r(pred_k, true),
                    "spearman_rho": _spearman_rho(pred_k, true),
                    "precision_at_5": _precision_at_k(pred_k, true, 5),
                    "ndcg_at_5": _ndcg_at_k(pred_k, true, 5),
                })

                # k-NN and prior
                for method, mu_pred in [("knn", mu_knn),
                                        ("prior", train_mean)]:
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
                        "crps": np.nan,
                        "pearson_r": _pearson_r(pred, true),
                        "spearman_rho": _spearman_rho(pred, true),
                        "precision_at_5": _precision_at_k(pred, true, 5),
                        "ndcg_at_5": _ndcg_at_k(pred, true, 5),
                    })

    return pd.DataFrame(rows)


def uncertainty_shrinkage(state_or_Sigma, Sigma_0: np.ndarray) -> float:
    """Posterior uncertainty as a fraction of prior uncertainty.

    Computes tr(Sigma_k) / tr(Sigma_0), measuring how much total
    uncertainty remains after observations. A value of 0.5 means
    uncertainty has halved; 0.2 means 80% of initial uncertainty
    has been eliminated. Useful as a stopping criterion for
    observation collection.

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
    """Compute the Kalman advantage over the baseline.

    Takes the output of held_out_evaluation() and returns the
    per-frac_observed difference: kalman - baseline.

    Parameters
    ----------
    results : DataFrame from held_out_evaluation().
    metric : column name to compare (default: "cosine_similarity").

    Returns
    -------
    DataFrame with columns [frac_observed, kalman, baseline, delta].
    """
    means = results.groupby(["frac_observed", "method"])[metric].mean().unstack()
    out = pd.DataFrame({
        "frac_observed": means.index,
        "kalman": means["kalman"].values,
        "baseline": means["knn"].values,
    })
    out["delta"] = out["kalman"] - out["baseline"]
    return out.reset_index(drop=True)


def active_learning_curve(
    pop,
    true_vector: np.ndarray,
    n_steps: int = 20,
    strategies: tuple[str, ...] = ("uncertainty", "random"),
    obs_noise: float = 0.05,
    n_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare observation-selection strategies on a single true profile.

    For each strategy, repeatedly cold-starts a profile, observes a noisy
    value at the chosen feature index, and records recovery metrics over
    all features after each step. Use this to quantify the lift from
    uncertainty-based active selection over random sampling.

    Parameters
    ----------
    pop : Population
        Provides the prior mean and covariance.
    true_vector : (K,) array
        Ground-truth profile to recover.
    n_steps : observations per trial.
    strategies : any of {"uncertainty", "random"}.
    obs_noise : Gaussian noise added to each observation.
    n_trials : independent trials per strategy (different RNG draws).
    seed : base RNG seed.

    Returns
    -------
    DataFrame with columns [trial, strategy, step, mae, rmse, recovery]
    where recovery = 1 - ||mu - true||^2 / ||prior - true||^2.
    """
    true = np.asarray(true_vector, dtype=float)
    K = len(true)
    if K != pop.matrix.shape[1]:
        raise ValueError(
            f"true_vector length {K} does not match population K={pop.matrix.shape[1]}"
        )

    prior_diff = pop.population_mean - true
    baseline_sse = float(np.dot(prior_diff, prior_diff))
    if baseline_sse < 1e-15:
        raise ValueError("true_vector equals the population mean; recovery is undefined.")

    rows = []
    for trial in range(n_trials):
        for strategy in strategies:
            rng = np.random.default_rng(seed + trial * 1000 + hash(strategy) % 1000)
            profile = pop.profile()
            observed: set[int] = set()
            for step in range(1, n_steps + 1):
                if strategy == "uncertainty":
                    stds = np.sqrt(np.diag(profile.Sigma))
                    candidate_order = np.argsort(stds)[::-1]
                    j = next((int(i) for i in candidate_order if int(i) not in observed), None)
                elif strategy == "random":
                    available = [i for i in range(K) if i not in observed]
                    j = int(rng.choice(available))
                else:
                    raise ValueError(f"Unknown strategy {strategy!r}")
                if j is None:
                    break
                noisy = float(true[j] + rng.normal(0.0, obs_noise))
                profile.observe(j, noisy)
                observed.add(j)

                err = profile.mu - true
                sse = float(np.dot(err, err))
                rows.append({
                    "trial": trial,
                    "strategy": strategy,
                    "step": step,
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(sse / K)),
                    "recovery": 1.0 - sse / baseline_sse,
                })
    return pd.DataFrame(rows)
