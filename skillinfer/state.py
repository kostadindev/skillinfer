"""Profile: a skill profile that gets sharper with observations."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from skillinfer._kalman import kalman_update, kalman_update_batch
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

    Attributes
    ----------
    mu : np.ndarray
        (K,) posterior mean.
    Sigma : np.ndarray
        (K, K) posterior covariance.
    feature_names : list[str]
        Feature names inherited from Population.
    n_observations : int
        Number of observe() calls applied.
    """

    def __init__(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        feature_names: list[str],
        noise: float = 1e-6,
    ):
        self.mu = mu
        self.Sigma = Sigma
        self._prior_var = np.diag(Sigma).copy()
        self.feature_names = feature_names
        self.noise = noise
        self.n_observations = 0
        self._observed: set[int] = set()
        self._name_to_idx = {name: i for i, name in enumerate(feature_names)}
        self._skills: dict[str, Skill] = {}

    @classmethod
    def from_dict(cls, data: dict) -> Profile:
        """Reconstruct a Profile from the output of ``to_dict()``.

        Note: this restores the mean vector and metadata but not the
        full posterior covariance. The covariance is set to a diagonal
        matrix using the exported std values. For full covariance
        round-tripping, re-create the profile from a Population.

        Parameters
        ----------
        data : dict as returned by ``to_dict()``.
        """
        mu = np.array(data["mean"], dtype=float)
        stds = np.array(data["std"], dtype=float)
        Sigma = np.diag(stds ** 2)
        feature_names = list(data["feature_names"])
        noise = data.get("noise", 1e-6)
        profile = cls(mu=mu, Sigma=Sigma, feature_names=feature_names, noise=noise)
        profile.n_observations = data.get("n_observations", 0)
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for feat in data.get("observed_features", []):
            if feat in name_to_idx:
                profile._observed.add(name_to_idx[feat])
        return profile

    @classmethod
    def from_json(cls, source: str) -> Profile:
        """Reconstruct a Profile from a JSON string or file path.

        Parameters
        ----------
        source : JSON string, or a path to a JSON file.
        """
        import json
        try:
            data = json.loads(source)
        except json.JSONDecodeError:
            with open(source) as f:
                data = json.load(f)
        return cls.from_dict(data)

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
        if idx < 0 or idx >= len(self.mu):
            raise IndexError(
                f"Feature index {idx} out of range for {len(self.mu)} features."
            )
        return idx

    def observe(self, feature: str | int | Skill, value: float | None = None) -> Profile:
        """Observe one feature value. Updates mu and Sigma in place.

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
        self.mu, self.Sigma = kalman_update(
            self.mu, self.Sigma, j, value, self.noise
        )
        self._observed.add(j)
        self.n_observations += 1
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
        indices = np.array([self._resolve_index(f) for f in observations])
        values = np.array(list(observations.values()), dtype=float)
        self.mu, self.Sigma = kalman_update_batch(
            self.mu, self.Sigma, indices, values, self.noise
        )
        for j in indices:
            self._observed.add(int(j))
        self.n_observations += len(observations)
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
            observed_features, noise.
        """
        stds = np.sqrt(np.diag(self.Sigma)).tolist()
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mu.tolist(),
            "std": stds,
            "n_observations": self.n_observations,
            "observed_features": [self.feature_names[i] for i in sorted(self._observed)],
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
            mu=self.mu.copy(),
            Sigma=self.Sigma.copy(),
            feature_names=list(self.feature_names),
            noise=self.noise,
        )
        new.n_observations = self.n_observations
        new._observed = set(self._observed)
        new._prior_var = self._prior_var.copy()
        new._skills = dict(self._skills)
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
