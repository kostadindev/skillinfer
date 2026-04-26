"""Held-out evaluation: does covariance transfer help?"""

import numpy as np
import pandas as pd

from bayeskal._kalman import kalman_update, diagonal_update


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def held_out_evaluation(
    taxonomy,
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
    taxonomy : Taxonomy instance.
    frac_observed : fraction(s) of features to observe (scalar or list).
    n_splits : number of random train/test splits.
    obs_noise : observation noise standard deviation.
    seed : random seed.

    Returns
    -------
    DataFrame with columns:
        [split, entity, frac_observed, method, cosine_similarity, mse]
    """
    if isinstance(frac_observed, (int, float)):
        frac_observed = [frac_observed]

    R = taxonomy.matrix.values
    N, K = R.shape
    Sigma = taxonomy.covariance
    pop_mean = taxonomy.population_mean
    entity_names = taxonomy.entity_names

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
                for method, mu_pred in [("kalman", mu_k), ("diagonal", mu_d), ("prior", pop_mean)]:
                    pred = mu_pred[unobs_dims]
                    true = true_vec[unobs_dims]
                    rows.append({
                        "split": split,
                        "entity": entity_names[ei],
                        "frac_observed": frac,
                        "method": method,
                        "cosine_similarity": _cosine_sim(pred, true),
                        "mse": float(np.mean((pred - true) ** 2)),
                    })

    return pd.DataFrame(rows)
