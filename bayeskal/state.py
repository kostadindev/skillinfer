"""InferenceState: mutable posterior for one entity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayeskal._kalman import kalman_update, kalman_update_batch


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
