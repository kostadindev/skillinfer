"""Profile: a skill profile that gets sharper with observations."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from skillinfer._kalman import condition, posterior_covariance
from skillinfer.types import Skill, Task


class MatchResult(NamedTuple):
    """Result of matching an agent to a task vector."""

    score: float
    std: float
    ci_lower: float
    ci_upper: float
    p_above_threshold: float | None


class Profile:
    """Posterior belief about one entity's feature vector.

    Created via ``Population.profile()``. Updated via ``observe()`` calls.
    The population covariance is shared and never mutated — observations
    update only the posterior mean via the Gaussian conditioning formula.

    Attributes
    ----------
    mu : np.ndarray
        (K,) posterior mean (computed from observations via conditioning).
    Sigma : np.ndarray
        (K, K) posterior covariance (derived from population covariance
        and observed feature set).
    feature_names : list[str]
        Feature names inherited from Population.
    n_observations : int
        Number of observe() calls applied.
    """

    def __init__(
        self,
        prior_mean: np.ndarray,
        pop_cov: np.ndarray,
        feature_names: list[str],
        noise: float = 1e-6,
    ):
        self._prior_mean = prior_mean
        self._pop_cov = pop_cov          # shared reference, never mutated
        self.feature_names = feature_names
        self.noise = noise
        self.n_observations = 0
        self._observed: dict[int, float] = {}   # index -> observed value
        self._name_to_idx = {name: i for i, name in enumerate(feature_names)}
        self._skills: dict[str, Skill] = {}

        # Caches (invalidated on observe)
        self._mu_cache: np.ndarray | None = None
        self._sigma_cache: np.ndarray | None = None

    @property
    def mu(self) -> np.ndarray:
        """(K,) posterior mean vector."""
        if self._mu_cache is None:
            if not self._observed:
                self._mu_cache = self._prior_mean.copy()
            else:
                obs_idx = np.array(list(self._observed.keys()), dtype=int)
                obs_val = np.array(list(self._observed.values()), dtype=float)
                self._mu_cache = condition(
                    self._prior_mean, self._pop_cov,
                    obs_idx, obs_val, self.noise,
                )
        return self._mu_cache

    @property
    def Sigma(self) -> np.ndarray:
        """(K, K) posterior covariance matrix."""
        if self._sigma_cache is None:
            if not self._observed:
                self._sigma_cache = self._pop_cov.copy()
            else:
                obs_idx = np.array(list(self._observed.keys()), dtype=int)
                self._sigma_cache = posterior_covariance(
                    self._pop_cov, obs_idx, self.noise,
                )
        return self._sigma_cache

    @property
    def _prior_var(self) -> np.ndarray:
        """Diagonal of population covariance (prior variance)."""
        return np.diag(self._pop_cov)

    def _invalidate_cache(self) -> None:
        self._mu_cache = None
        self._sigma_cache = None

    @classmethod
    def from_dict(cls, data: dict, population=None) -> Profile:
        """Reconstruct a Profile from the output of ``to_dict()``.

        Parameters
        ----------
        data : dict as returned by ``to_dict()``.
        population : Population used to create the original profile.
            If given, observations are replayed for full fidelity.
            If None, a lossy reconstruction is used (diagonal covariance
            only — no cross-feature propagation on future observations).
        """
        if population is not None:
            noise = data.get("noise", 1e-6)
            profile = population.profile(noise=noise)
            obs = data.get("observations", {})
            if obs:
                profile.observe_many(obs)
            return profile

        # Lossy fallback: no population available
        mu = np.array(data["mean"], dtype=float)
        stds = np.array(data["std"], dtype=float)
        Sigma = np.diag(stds ** 2)
        feature_names = list(data["feature_names"])
        noise = data.get("noise", 1e-6)
        profile = cls(
            prior_mean=mu, pop_cov=Sigma,
            feature_names=feature_names, noise=noise,
        )
        profile.n_observations = data.get("n_observations", 0)
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for feat, val in data.get("observations", {}).items():
            if feat in name_to_idx:
                profile._observed[name_to_idx[feat]] = val
        # Pre-fill mu cache since we already have the conditioned mean
        profile._mu_cache = mu.copy()
        return profile

    @classmethod
    def from_json(cls, source: str, population=None) -> Profile:
        """Reconstruct a Profile from a JSON string or file path.

        Parameters
        ----------
        source : JSON string, or a path to a JSON file.
        population : Population to reconstruct from (see from_dict).
        """
        import json
        try:
            data = json.loads(source)
        except json.JSONDecodeError:
            with open(source) as f:
                data = json.load(f)
        return cls.from_dict(data, population)

    def _resolve_index(self, feature: str | int | Skill) -> int:
        if isinstance(feature, Skill):
            feature = feature.name
        if isinstance(feature, str):
            if feature not in self._name_to_idx:
                raise KeyError(
                    f"Unknown feature: {feature!r}. "
                    f"Available features: {self.feature_names}"
                )
            return self._name_to_idx[feature]
        idx = int(feature)
        if idx < 0 or idx >= len(self._prior_mean):
            raise IndexError(
                f"Feature index {idx} out of range for "
                f"{len(self._prior_mean)} features."
            )
        return idx

    def observe(self, feature: str | int | Skill, value: float | None = None) -> Profile:
        """Observe one feature value. Updates the posterior mean.

        Parameters
        ----------
        feature : feature name, integer index, or Skill with a score.
        value : observed value. Optional if feature is a Skill with a score.

        Returns self for method chaining.
        """
        if isinstance(feature, Skill) and value is None:
            if feature.score is None:
                raise ValueError(
                    f"Skill {feature.name!r} has no score. "
                    "Pass a value or set skill.score first."
                )
            value = feature.score
        if value is None:
            raise TypeError("value is required when feature is a str or int.")
        j = self._resolve_index(feature)
        self._observed[j] = float(value)
        self.n_observations += 1
        self._invalidate_cache()
        return self

    def observe_many(
        self, observations: dict[str | int, float] | list[Skill]
    ) -> Profile:
        """Observe multiple features at once.

        Parameters
        ----------
        observations : {feature: value} mapping, or list of Skills with scores.

        Returns self for method chaining.
        """
        if isinstance(observations, list):
            obs_dict: dict[str | int, float] = {}
            for skill in observations:
                if not isinstance(skill, Skill) or skill.score is None:
                    raise ValueError(
                        "When passing a list, each element must be a Skill with a score."
                    )
                obs_dict[skill.name] = skill.score
            observations = obs_dict
        for f, v in observations.items():
            j = self._resolve_index(f)
            self._observed[j] = float(v)
        self.n_observations += len(observations)
        self._invalidate_cache()
        return self

    def mean(self, feature: str | int | None = None) -> float | np.ndarray:
        """Posterior mean (clipped to [0, 1]). If feature given, return scalar; else full vector."""
        if feature is None:
            return np.clip(self.mu, 0.0, 1.0).copy()
        return float(np.clip(self.mu[self._resolve_index(feature)], 0.0, 1.0))

    def std(self, feature: str | int | None = None) -> float | np.ndarray:
        """Posterior standard deviation (sqrt of diagonal of Sigma)."""
        diag = np.sqrt(np.diag(self.Sigma))
        if feature is None:
            return diag
        return float(diag[self._resolve_index(feature)])

    def confidence_interval(
        self, feature: str | int, level: float = 0.95
    ) -> tuple[float, float]:
        """Gaussian confidence interval for a feature."""
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2)
        mu = self.mean(feature)
        s = self.std(feature)
        return (mu - z * s, mu + z * s)

    def most_uncertain(self, k: int = 10) -> pd.DataFrame:
        """Top-k features with highest posterior variance."""
        stds = np.sqrt(np.diag(self.Sigma))
        idx = np.argsort(stds)[::-1][:k]
        return pd.DataFrame({
            "feature": [self.feature_names[i] for i in idx],
            "mean": [float(np.clip(self.mu[i], 0.0, 1.0)) for i in idx],
            "std": [stds[i] for i in idx],
        })

    def _build_dataframe(self, include_ci: bool = False, detail: bool = False, level: float = 0.95) -> pd.DataFrame:
        """Build the output DataFrame."""
        stds = np.sqrt(np.diag(self.Sigma))
        mu = np.clip(self.mu, 0.0, 1.0)

        data: dict = {"feature": self.feature_names}
        if self._skills:
            descs = [self._skills.get(n, Skill(n)).description for n in self.feature_names]
            if any(descs):
                data["description"] = descs
        data["mean"] = mu
        data["std"] = stds
        if include_ci:
            from scipy.stats import norm
            z = norm.ppf(0.5 + level / 2)
            data["ci_lower"] = np.clip(mu - z * stds, 0.0, 1.0)
            data["ci_upper"] = np.clip(mu + z * stds, 0.0, 1.0)
        if detail:
            prior_stds = np.sqrt(np.maximum(self._prior_var, 1e-15))
            data["confidence"] = np.clip(1.0 - stds / prior_stds, 0.0, 1.0)
            data["source"] = [
                "observed" if i in self._observed else "predicted"
                for i in range(len(self.mu))
            ]
        return pd.DataFrame(data)

    def to_dataframe(self, detail: bool = False) -> pd.DataFrame:
        """Full posterior as a DataFrame.

        Parameters
        ----------
        detail : if True, include confidence and source columns.
            confidence is 0-1 (how much uncertainty was reduced).
            source is "observed" or "predicted".
        """
        return self._build_dataframe(include_ci=False, detail=detail)

    def to_dict(self) -> dict:
        """Export the profile as a plain dict (JSON-serialisable).

        Returns
        -------
        dict with keys: feature_names, mean, std, n_observations,
            observed_features, observations, noise.
        """
        stds = np.sqrt(np.diag(self.Sigma)).tolist()
        return {
            "feature_names": list(self.feature_names),
            "mean": np.clip(self.mu, 0.0, 1.0).tolist(),
            "std": stds,
            "n_observations": self.n_observations,
            "observed_features": [self.feature_names[i] for i in sorted(self._observed)],
            "observations": {
                self.feature_names[i]: v for i, v in self._observed.items()
            },
            "noise": self.noise,
        }

    def to_json(self, path: str | None = None) -> str:
        """Export the profile as JSON.

        Parameters
        ----------
        path : if given, write to this file. Otherwise return the JSON string.

        Returns
        -------
        JSON string.
        """
        import json
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(s)
        return s

    def similarity(self, other: np.ndarray) -> float:
        """Cosine similarity between posterior mean and a target vector."""
        other = np.asarray(other, dtype=float)
        dot = np.dot(self.mu, other)
        norm_a = np.linalg.norm(self.mu)
        norm_b = np.linalg.norm(other)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def uncertainty_ratio(self, Sigma_0: np.ndarray) -> float:
        """Fraction of prior uncertainty remaining: tr(Sigma) / tr(Sigma_0).

        A value of 0.5 means uncertainty has halved since the prior.
        Useful for deciding when enough observations have been collected.

        Parameters
        ----------
        Sigma_0 : (K, K) ndarray, typically pop.covariance.
        """
        tr_0 = np.trace(Sigma_0)
        if tr_0 < 1e-15:
            return 0.0
        return float(np.trace(self.Sigma) / tr_0)

    def metrics_by_category(
        self,
        true_vector: np.ndarray,
        categories: dict[str, str] | None = None,
        sep: str = ":",
    ) -> pd.DataFrame:
        """Per-category prediction metrics on a known truth vector.

        Splits features into categories (by ``sep`` in feature names by
        default, e.g. ``Skill:Programming`` -> ``Skill``) and reports
        MAE, RMSE, and recovery per group. Recovery is the share of
        squared error eliminated relative to predicting the prior mean.

        Parameters
        ----------
        true_vector : (K,) ground-truth profile.
        categories : optional mapping {feature_name: category}. If None,
            categories are derived by splitting feature names on ``sep``;
            features without ``sep`` go into ``"uncategorised"``.
        sep : separator used to derive categories from feature names.

        Returns
        -------
        DataFrame with columns
        [category, n_features, n_observed, mae, rmse, recovery].
        """
        true = np.asarray(true_vector, dtype=float)
        if true.shape != self.mu.shape:
            raise ValueError(
                f"true_vector shape {true.shape} does not match profile {self.mu.shape}"
            )

        if categories is None:
            cats = [
                n.split(sep, 1)[0] if sep in n else "uncategorised"
                for n in self.feature_names
            ]
        else:
            cats = [categories.get(n, "uncategorised") for n in self.feature_names]

        mu = self.mu
        prior = self._prior_mean
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cats):
            groups.setdefault(c, []).append(i)

        rows = []
        for cat, idxs in groups.items():
            idx = np.array(idxs, dtype=int)
            err = mu[idx] - true[idx]
            prior_err = prior[idx] - true[idx]
            sse = float(np.dot(err, err))
            prior_sse = float(np.dot(prior_err, prior_err))
            recovery = (
                1.0 - sse / prior_sse if prior_sse > 1e-15 else float("nan")
            )
            rows.append({
                "category": cat,
                "n_features": len(idx),
                "n_observed": sum(1 for i in idx if int(i) in self._observed),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(sse / len(idx))),
                "recovery": recovery,
            })
        return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)

    def rmse(self, true_vector: np.ndarray) -> float:
        """Root mean squared error between posterior mean and a target."""
        true_vector = np.asarray(true_vector, dtype=float)
        return float(np.sqrt(np.mean((self.mu - true_vector) ** 2)))

    def mae(self, true_vector: np.ndarray) -> float:
        """Mean absolute error between posterior mean and a target."""
        true_vector = np.asarray(true_vector, dtype=float)
        return float(np.mean(np.abs(self.mu - true_vector)))

    def predict(
        self, feature: str | int | None = None, level: float = 0.95, detail: bool = False
    ) -> pd.DataFrame | dict:
        """Predict skill values with uncertainty.

        Parameters
        ----------
        feature : if given, predict one skill. If None, predict all.
        level : confidence level for the interval (default 0.95).
        detail : if True, include confidence and source columns.

        Returns
        -------
        If feature is given: dict with keys mean, std, ci_lower, ci_upper.
        If feature is None: DataFrame with all skills and CIs.
        """
        from scipy.stats import norm

        if feature is not None:
            j = self._resolve_index(feature)
            mu_j = float(np.clip(self.mu[j], 0.0, 1.0))
            std_j = float(np.sqrt(self.Sigma[j, j]))
            z = norm.ppf(0.5 + level / 2)
            result = {
                "feature": self.feature_names[j] if isinstance(feature, int) else feature,
                "mean": mu_j,
                "std": std_j,
                "ci_lower": float(np.clip(mu_j - z * std_j, 0.0, 1.0)),
                "ci_upper": float(np.clip(mu_j + z * std_j, 0.0, 1.0)),
            }
            if detail:
                prior_std = float(np.sqrt(max(self._prior_var[j], 1e-15)))
                result["confidence"] = float(np.clip(1.0 - std_j / prior_std, 0.0, 1.0))
                result["source"] = "observed" if j in self._observed else "predicted"
            return result

        return self._build_dataframe(include_ci=True, detail=detail, level=level)

    @property
    def agent_vector(self) -> pd.Series:
        """The inferred skill profile as a named Series (posterior mean)."""
        return pd.Series(self.mu, index=self.feature_names, name="agent_vector")

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """Posterior covariance as a labeled DataFrame."""
        return pd.DataFrame(
            self.Sigma,
            index=self.feature_names,
            columns=self.feature_names,
        )

    def summary(self, true_vector: np.ndarray | None = None) -> dict:
        """Summary statistics for this profile.

        Returns a dict with:
        - n_features, n_observed, n_predicted: counts
        - mean_std: average posterior standard deviation
        - uncertainty_reduction: fraction of prior uncertainty removed (0-1)
        - top_predicted: top 3 predicted features by mean
        - most_uncertain: top 3 features by posterior std

        If ``true_vector`` is given, also includes:
        - mae, rmse, max_error, cosine_similarity: accuracy vs. ground truth
        - coverage_95: fraction of true values inside 95% confidence intervals

        Parameters
        ----------
        true_vector : optional (K,) ground truth for evaluation.
        """
        stds = np.sqrt(np.diag(self.Sigma))
        prior_stds = np.sqrt(np.maximum(self._prior_var, 1e-15))
        reduction = float(np.mean(np.clip(1.0 - stds / prior_stds, 0.0, 1.0)))

        # Top predicted (unobserved) by mean
        unobserved_idx = [i for i in range(len(self.mu)) if i not in self._observed]
        top_pred_idx = sorted(unobserved_idx, key=lambda i: -self.mu[i])[:3]
        top_predicted = [
            {"feature": self.feature_names[i], "mean": float(self.mu[i]), "std": float(stds[i])}
            for i in top_pred_idx
        ]

        # Most uncertain
        uncertain_idx = np.argsort(stds)[::-1][:3]
        most_uncertain = [
            {"feature": self.feature_names[i], "std": float(stds[i])}
            for i in uncertain_idx
        ]

        result = {
            "n_features": len(self.mu),
            "n_observed": len(self._observed),
            "n_predicted": len(self.mu) - len(self._observed),
            "mean_std": float(stds.mean()),
            "uncertainty_reduction": reduction,
            "top_predicted": top_predicted,
            "most_uncertain": most_uncertain,
        }

        if true_vector is not None:
            true_vector = np.asarray(true_vector, dtype=float)
            errors = np.abs(self.mu - true_vector)
            result["mae"] = float(errors.mean())
            result["rmse"] = float(np.sqrt(np.mean(errors ** 2)))
            result["max_error"] = float(errors.max())
            result["cosine_similarity"] = self.similarity(true_vector)

            # Coverage: fraction of true values inside 95% CI
            z = 1.96
            lo = self.mu - z * stds
            hi = self.mu + z * stds
            inside = (true_vector >= lo) & (true_vector <= hi)
            result["coverage_95"] = float(inside.mean())

        return result

    def copy(self) -> Profile:
        """Deep copy of the state."""
        new = Profile(
            prior_mean=self._prior_mean,
            pop_cov=self._pop_cov,
            feature_names=list(self.feature_names),
            noise=self.noise,
        )
        new.n_observations = self.n_observations
        new._observed = dict(self._observed)
        new._skills = dict(self._skills)
        if self._mu_cache is not None:
            new._mu_cache = self._mu_cache.copy()
        if self._sigma_cache is not None:
            new._sigma_cache = self._sigma_cache.copy()
        return new

    def match_score(
        self,
        task_vector: dict[str, float] | np.ndarray | Task,
        threshold: float | None = None,
        level: float = 0.95,
    ) -> MatchResult:
        """Score this agent against a task vector.

        Computes the expected weighted-average performance and propagates
        uncertainty through the linear combination. Weights are normalised
        so the score stays on the same scale as the underlying skills.

        Parameters
        ----------
        task_vector : Task, dict mapping feature names to importance
            weights, or a (K,) numpy array. Weights are relative
            importance — they are normalised internally.
        threshold : if given, compute P(score > threshold).
        level : confidence level for the interval (default 0.95).

        Returns
        -------
        MatchResult with fields: score, std, ci_lower, ci_upper,
            p_above_threshold (None if no threshold given).
        """
        from scipy.stats import norm

        if isinstance(task_vector, Task):
            task_vector = task_vector.weights
        if isinstance(task_vector, dict):
            w = np.zeros(len(self.mu))
            for feat, weight in task_vector.items():
                w[self._resolve_index(feat)] = weight
        else:
            w = np.asarray(task_vector, dtype=float)
            if w.shape != self.mu.shape:
                raise ValueError(
                    f"task_vector shape {w.shape} doesn't match "
                    f"feature count {self.mu.shape}"
                )

        w_sum = float(np.sum(np.abs(w)))
        if w_sum > 1e-15:
            w = w / w_sum

        score = float(w @ self.mu)
        var = float(w @ self.Sigma @ w)
        std = float(np.sqrt(max(var, 0.0)))

        z = norm.ppf(0.5 + level / 2)
        ci_lower = score - z * std
        ci_upper = score + z * std

        p_above = None
        if threshold is not None and std > 1e-15:
            p_above = float(1.0 - norm.cdf((threshold - score) / std))

        return MatchResult(
            score=score,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_above_threshold=p_above,
        )

    def __repr__(self) -> str:
        K = len(self.mu)
        mean_std = float(np.sqrt(np.diag(self.Sigma)).mean())
        return f"Profile(K={K}, n_obs={self.n_observations}, mean_std={mean_std:.4f})"

    def __str__(self) -> str:
        stds = np.sqrt(np.diag(self.Sigma))
        max_name = max(len(n) for n in self.feature_names)
        lines = [f"Agent Vector ({self.n_observations} observations, {len(self.mu)} skills)"]
        lines.append(f"{'Skill':<{max_name}}  {'Mean':>8}  {'± Std':>8}")
        lines.append("-" * (max_name + 20))
        for i, name in enumerate(self.feature_names):
            lines.append(f"{name:<{max_name}}  {self.mu[i]:>8.4f}  {stds[i]:>8.4f}")
        return "\n".join(lines)


class GMMProfile(Profile):
    """Posterior under a Gaussian-mixture prior (GMM-Kalman).

    The prior is ``p(c) = sum_m pi_m N(mu_m, Sigma_m)``. Each observation
    triggers per-component Kalman updates plus mixture re-weighting via
    the per-component marginal predictive likelihood. The reported mean
    and covariance are the mixture marginal:

        mu_hat    = sum_m pi_m mu_m
        Sigma_hat = sum_m pi_m [Sigma_m + (mu_m - mu_hat)(mu_m - mu_hat)^T]

    See ``_gmm.py`` for the math and the chapter Section 4.5 for the
    motivation. The ``Profile`` interface (observe, predict, match_score,
    etc.) is unchanged.
    """

    def __init__(
        self,
        prior_means: list[np.ndarray],
        prior_covariances: list[np.ndarray],
        prior_weights: np.ndarray,
        feature_names: list[str],
        noise: float = 1e-6,
    ):
        from skillinfer._gmm import gmm_marginal

        prior_means = [np.asarray(m, dtype=float).copy() for m in prior_means]
        prior_covariances = [np.asarray(S, dtype=float).copy() for S in prior_covariances]
        prior_weights = np.asarray(prior_weights, dtype=float).copy()
        prior_weights = prior_weights / prior_weights.sum()

        # Prior marginal mean / covariance, used to satisfy the parent
        # contract (so things like uncertainty_ratio against Sigma_0 still
        # make sense). The actual GMM math goes through gmm_state below.
        prior_mean_marg, prior_cov_marg = gmm_marginal(
            prior_means, prior_covariances, prior_weights,
        )
        super().__init__(
            prior_mean=prior_mean_marg,
            pop_cov=prior_cov_marg,
            feature_names=feature_names,
            noise=noise,
        )
        self._prior_means = prior_means
        self._prior_covariances = prior_covariances
        self._prior_weights = prior_weights
        self._gmm_state_cache: tuple[
            list[np.ndarray], list[np.ndarray], np.ndarray
        ] | None = None

    @property
    def n_components(self) -> int:
        """Number of mixture components M."""
        return len(self._prior_means)

    @property
    def gmm_state(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Posterior (component means, covariances, mixture weights).

        Recomputed lazily from ``self._observed`` so observation order
        is irrelevant.
        """
        if self._gmm_state_cache is None:
            from skillinfer._gmm import gmm_condition

            if not self._observed:
                self._gmm_state_cache = (
                    [m.copy() for m in self._prior_means],
                    [S.copy() for S in self._prior_covariances],
                    self._prior_weights.copy(),
                )
            else:
                obs_idx = np.array(list(self._observed.keys()), dtype=int)
                obs_val = np.array(list(self._observed.values()), dtype=float)
                self._gmm_state_cache = gmm_condition(
                    self._prior_means,
                    self._prior_covariances,
                    self._prior_weights,
                    obs_idx, obs_val, self.noise,
                )
        return self._gmm_state_cache

    def _compute_marginal(self) -> None:
        from skillinfer._gmm import gmm_marginal

        means, covs, weights = self.gmm_state
        mu_hat, Sigma_hat = gmm_marginal(means, covs, weights)
        self._mu_cache = mu_hat
        self._sigma_cache = Sigma_hat

    @property
    def mu(self) -> np.ndarray:
        if self._mu_cache is None:
            self._compute_marginal()
        return self._mu_cache

    @property
    def Sigma(self) -> np.ndarray:
        if self._sigma_cache is None:
            self._compute_marginal()
        return self._sigma_cache

    @property
    def weights(self) -> np.ndarray:
        """Posterior mixture weights given the observations so far."""
        return self.gmm_state[2].copy()

    def _invalidate_cache(self) -> None:
        super()._invalidate_cache()
        self._gmm_state_cache = None

    def copy(self) -> GMMProfile:
        new = GMMProfile(
            prior_means=self._prior_means,
            prior_covariances=self._prior_covariances,
            prior_weights=self._prior_weights,
            feature_names=list(self.feature_names),
            noise=self.noise,
        )
        new.n_observations = self.n_observations
        new._observed = dict(self._observed)
        new._skills = dict(self._skills)
        return new

    def __repr__(self) -> str:
        K = len(self.mu)
        mean_std = float(np.sqrt(np.diag(self.Sigma)).mean())
        return (
            f"GMMProfile(K={K}, M={self.n_components}, "
            f"n_obs={self.n_observations}, mean_std={mean_std:.4f})"
        )
