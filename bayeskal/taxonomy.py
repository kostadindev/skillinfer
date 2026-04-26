"""Taxonomy: a population of entities described by K features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayeskal._covariance import (
    ledoit_wolf_covariance,
    sample_covariance,
    correlation_matrix,
    pca_embedding,
)


class Taxonomy:
    """A population of entities described by K features, with learned covariance.

    The covariance structure encodes how features co-vary across the population.
    This enables Bayesian inference: observing one feature of a new entity
    updates beliefs about all other features via the off-diagonal covariance.

    Attributes
    ----------
    matrix : pd.DataFrame
        (N_entities, K_features) raw data.
    feature_names : list[str]
        Column names.
    entity_names : list[str]
        Row index values.
    covariance : np.ndarray
        (K, K) covariance matrix.
    correlation : np.ndarray
        (K, K) Pearson correlation matrix.
    population_mean : np.ndarray
        (K,) mean across all entities.
    shrinkage : float | None
        Ledoit-Wolf shrinkage coefficient (None if sample covariance).
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        covariance: np.ndarray,
        shrinkage: float | None = None,
    ):
        self.matrix = matrix
        self.feature_names = list(matrix.columns)
        self.entity_names = list(matrix.index)
        self.covariance = covariance
        self.correlation = correlation_matrix(covariance)
        self.population_mean = matrix.values.mean(axis=0)
        self.shrinkage = shrinkage
        self._feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        normalize: bool = True,
        covariance: str = "ledoit-wolf",
    ) -> Taxonomy:
        """Construct a Taxonomy from a pandas DataFrame.

        Parameters
        ----------
        df : DataFrame with rows=entities, columns=features.
        normalize : if True, scale each column to [0, 1].
        covariance : "ledoit-wolf" (default, recommended) or "sample".
        """
        df = df.copy()
        if normalize:
            col_min = df.min()
            col_range = df.max() - col_min
            col_range = col_range.replace(0, 1)
            df = (df - col_min) / col_range

        R = df.values
        if covariance == "ledoit-wolf":
            Sigma, alpha = ledoit_wolf_covariance(R)
        elif covariance == "sample":
            Sigma = sample_covariance(R)
            alpha = None
        else:
            raise ValueError(f"Unknown covariance method: {covariance}")

        return cls(matrix=df, covariance=Sigma, shrinkage=alpha)

    @classmethod
    def from_csv(
        cls,
        path: str,
        index_col: int | str = 0,
        normalize: bool = True,
        covariance: str = "ledoit-wolf",
    ) -> Taxonomy:
        """Construct from a CSV file."""
        df = pd.read_csv(path, index_col=index_col)
        return cls.from_dataframe(df, normalize=normalize, covariance=covariance)

    def entity(self, name: str) -> np.ndarray:
        """Get the feature vector for a named entity."""
        return self.matrix.loc[name].values.copy()

    def new_state(
        self,
        prior_entity: str | None = None,
        prior_mean: np.ndarray | None = None,
        obs_noise: float = 0.05,
    ):
        """Create an InferenceState for a new individual.

        Parameters
        ----------
        prior_entity : if given, use this entity's vector as the prior mean.
        prior_mean : if given, use this directly as the prior mean.
        obs_noise : observation noise standard deviation.

        If neither prior_entity nor prior_mean is given, uses the population mean.
        """
        from bayeskal.state import InferenceState

        if prior_entity is not None:
            mu = self.entity(prior_entity)
        elif prior_mean is not None:
            mu = np.asarray(prior_mean, dtype=float).copy()
        else:
            mu = self.population_mean.copy()

        return InferenceState(
            mu=mu,
            Sigma=self.covariance.copy(),
            feature_names=self.feature_names,
            obs_noise=obs_noise,
        )

    def pca(self, n_components: int = 15) -> dict:
        """PCA summary of the entity-feature matrix.

        Returns dict with keys: components, explained_variance_ratio, cumulative.
        """
        return pca_embedding(self.matrix.values, n_components)

    def top_correlations(self, k: int = 20) -> pd.DataFrame:
        """Top-k strongest feature-feature correlations (absolute value).

        Returns DataFrame with columns [feature_a, feature_b, correlation].
        """
        K = len(self.feature_names)
        rows = []
        for i in range(K):
            for j in range(i + 1, K):
                rows.append((
                    self.feature_names[i],
                    self.feature_names[j],
                    self.correlation[i, j],
                ))
        df = pd.DataFrame(rows, columns=["feature_a", "feature_b", "correlation"])
        df["abs_corr"] = df["correlation"].abs()
        return (
            df.sort_values("abs_corr", ascending=False)
            .head(k)
            .drop(columns="abs_corr")
            .reset_index(drop=True)
        )

    def condition_number(self) -> float:
        """Condition number of the covariance matrix."""
        return float(np.linalg.cond(self.covariance))

    def __repr__(self) -> str:
        N, K = self.matrix.shape
        s = f"Taxonomy({N} entities x {K} features"
        if self.shrinkage is not None:
            s += f", shrinkage={self.shrinkage:.4f}"
        s += ")"
        return s
