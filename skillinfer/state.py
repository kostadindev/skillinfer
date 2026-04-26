"""InferenceState: mutable posterior for one entity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skillinfer._kalman import kalman_update, kalman_update_batch


class InferenceState:
    """Posterior belief about one entity's feature vector.

    Created via ``Taxonomy.new_state()``. Updated via ``observe()`` calls.

    Attributes
    ----------
    mu : np.ndarray
        (K,) posterior mean.
    Sigma : np.ndarray
        (K, K) posterior covariance.
    obs_noise : float
        Observation noise standard deviation.
    feature_names : list[str]
        Feature names inherited from Taxonomy.
    n_observations : int
        Number of observe() calls applied.
    """

    def __init__(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        feature_names: list[str],
        obs_noise: float = 0.05,
    ):
        self.mu = mu
        self.Sigma = Sigma
        self.feature_names = feature_names
        self.obs_noise = obs_noise
        self.n_observations = 0
        self._name_to_idx = {name: i for i, name in enumerate(feature_names)}

    def _resolve_index(self, feature: str | int) -> int:
        if isinstance(feature, str):
            if feature not in self._name_to_idx:
                raise KeyError(f"Unknown feature: {feature!r}")
            return self._name_to_idx[feature]
        return int(feature)

    def observe(self, feature: str | int, value: float) -> InferenceState:
        """Observe one feature value. Updates mu and Sigma in place.

        Parameters
        ----------
        feature : feature name or integer index.
        value : observed value.

        Returns self for method chaining.
        """
        j = self._resolve_index(feature)
        self.mu, self.Sigma = kalman_update(
            self.mu, self.Sigma, j, value, self.obs_noise
        )
        self.n_observations += 1
        return self

    def observe_many(
        self, observations: dict[str | int, float]
    ) -> InferenceState:
        """Observe multiple features at once.

        Parameters
        ----------
        observations : {feature: value} mapping.

        Returns self for method chaining.
        """
        indices = np.array([self._resolve_index(f) for f in observations])
        values = np.array(list(observations.values()), dtype=float)
        self.mu, self.Sigma = kalman_update_batch(
            self.mu, self.Sigma, indices, values, self.obs_noise
        )
        self.n_observations += len(observations)
        return self

    def mean(self, feature: str | int | None = None) -> float | np.ndarray:
        """Posterior mean. If feature given, return scalar; else full vector."""
        if feature is None:
            return self.mu.copy()
        return float(self.mu[self._resolve_index(feature)])

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
            "mean": [self.mu[i] for i in idx],
            "std": [stds[i] for i in idx],
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Full posterior as a DataFrame with columns [feature, mean, std]."""
        stds = np.sqrt(np.diag(self.Sigma))
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean": self.mu,
            "std": stds,
        })

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
        Sigma_0 : (K, K) ndarray, typically taxonomy.covariance.
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
        self, feature: str | int | None = None, level: float = 0.95
    ) -> pd.DataFrame | dict:
        """Predict skill values with uncertainty.

        Parameters
        ----------
        feature : if given, predict one skill. If None, predict all.
        level : confidence level for the interval (default 0.95).

        Returns
        -------
        If feature is given: dict with keys mean, std, ci_lower, ci_upper.
        If feature is None: DataFrame with all skills and their predictions.
        """
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2)

        if feature is not None:
            j = self._resolve_index(feature)
            mu_j = float(self.mu[j])
            std_j = float(np.sqrt(self.Sigma[j, j]))
            return {
                "feature": self.feature_names[j] if isinstance(feature, int) else feature,
                "mean": mu_j,
                "std": std_j,
                "ci_lower": mu_j - z * std_j,
                "ci_upper": mu_j + z * std_j,
            }

        stds = np.sqrt(np.diag(self.Sigma))
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean": self.mu,
            "std": stds,
            "ci_lower": self.mu - z * stds,
            "ci_upper": self.mu + z * stds,
        })

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

    def copy(self) -> InferenceState:
        """Deep copy of the state."""
        return InferenceState(
            mu=self.mu.copy(),
            Sigma=self.Sigma.copy(),
            feature_names=list(self.feature_names),
            obs_noise=self.obs_noise,
        )

    def __repr__(self) -> str:
        K = len(self.mu)
        mean_std = float(np.sqrt(np.diag(self.Sigma)).mean())
        return f"InferenceState(K={K}, n_obs={self.n_observations}, mean_std={mean_std:.4f})"

    def __str__(self) -> str:
        stds = np.sqrt(np.diag(self.Sigma))
        max_name = max(len(n) for n in self.feature_names)
        lines = [f"Agent Vector ({self.n_observations} observations, {len(self.mu)} skills)"]
        lines.append(f"{'Skill':<{max_name}}  {'Mean':>8}  {'± Std':>8}")
        lines.append("-" * (max_name + 20))
        for i, name in enumerate(self.feature_names):
            lines.append(f"{name:<{max_name}}  {self.mu[i]:>8.4f}  {stds[i]:>8.4f}")
        return "\n".join(lines)
