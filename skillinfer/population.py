"""Population: a population of entities described by K features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skillinfer._covariance import (
    ledoit_wolf_covariance,
    sample_covariance,
    correlation_matrix,
    pca_embedding,
)
from skillinfer.types import Skill


class Population:
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
        self._skills: dict[str, Skill] = {
            name: Skill(name) for name in self.feature_names
        }

    @property
    def skills(self) -> list[Skill]:
        """The skill dimensions with their descriptions."""
        return [self._skills[name] for name in self.feature_names]

    def describe_skills(
        self, descriptions: dict[str, str] | list[Skill]
    ) -> None:
        """Attach descriptions to skill dimensions.

        Parameters
        ----------
        descriptions : dict mapping skill names to descriptions,
            or a list of Skill objects.

        Examples
        --------
        >>> tax.describe_skills({
        ...     "BBH": "Big-Bench Hard: diverse challenging tasks",
        ...     "MMLU-PRO": "Professional-level multitask understanding",
        ... })
        >>> tax.describe_skills([Skill("BBH", description="Big-Bench Hard")])
        """
        if isinstance(descriptions, list):
            for skill in descriptions:
                if skill.name in self._skills:
                    self._skills[skill.name] = skill
        else:
            for name, desc in descriptions.items():
                if name in self._skills:
                    self._skills[name] = Skill(name, description=desc)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        normalize: bool = True,
        covariance: str = "ledoit-wolf",
    ) -> Population:
        """Construct a Population from a pandas DataFrame.

        Parameters
        ----------
        df : DataFrame with rows=entities, columns=features.
        normalize : if True, scale each column to [0, 1].
        covariance : "ledoit-wolf" (default, recommended) or "sample".
        """
        df = df.copy()
        if df.isna().any().any():
            raise ValueError(
                "DataFrame contains NaN values. Drop or impute them before "
                "building a Population: df.dropna() or df.fillna(method)."
            )
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
    def from_covariance(
        cls,
        covariance: np.ndarray,
        feature_names: list[str],
        population_mean: np.ndarray,
    ) -> Population:
        """Construct a Population from a pre-computed covariance matrix.

        Use this when you have a domain-expert covariance (e.g., from
        a previous study) rather than estimating it from data.

        Parameters
        ----------
        covariance : (K, K) covariance matrix.
        feature_names : list of K feature names.
        population_mean : (K,) mean vector.
        """
        K = len(feature_names)
        covariance = np.asarray(covariance, dtype=float)
        population_mean = np.asarray(population_mean, dtype=float)
        if covariance.shape != (K, K):
            raise ValueError(
                f"Covariance shape {covariance.shape} doesn't match "
                f"{K} features."
            )
        if population_mean.shape != (K,):
            raise ValueError(
                f"Mean shape {population_mean.shape} doesn't match "
                f"{K} features."
            )
        df = pd.DataFrame(
            population_mean.reshape(1, -1),
            columns=feature_names,
            index=["_population_mean"],
        )
        tax = cls(matrix=df, covariance=covariance, shrinkage=None)
        tax.population_mean = population_mean
        return tax

    @classmethod
    def from_csv(
        cls,
        path: str,
        index_col: int | str = 0,
        normalize: bool = True,
        covariance: str = "ledoit-wolf",
    ) -> Population:
        """Construct from a CSV file."""
        df = pd.read_csv(path, index_col=index_col)
        return cls.from_dataframe(df, normalize=normalize, covariance=covariance)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        normalize: bool = True,
        covariance: str = "ledoit-wolf",
    ) -> Population:
        """Construct from a Parquet file.

        Parameters
        ----------
        path : path to the Parquet file.
        normalize : if True, scale each column to [0, 1].
        covariance : "ledoit-wolf" or "sample".
        """
        df = pd.read_parquet(path)
        return cls.from_dataframe(df, normalize=normalize, covariance=covariance)

    def entity(self, name: str) -> np.ndarray:
        """Get the feature vector for a named entity."""
        return self.matrix.loc[name].values.copy()

    def skill_vector(self, name: str) -> pd.Series:
        """Get a named entity's skill vector as a labeled Series."""
        return self.matrix.loc[name].copy()

    def profile(
        self,
        prior_entity: str | None = None,
        prior_mean: np.ndarray | None = None,
        noise: float | None = None,
        method: str = "kalman",
        rank: int | None = None,
        blocks: list[list[str | int]] | dict[str, str | int] | None = None,
    ):
        """Create a Profile for a new individual.

        Parameters
        ----------
        prior_entity : if given, use this entity's vector as the prior mean.
        prior_mean : if given, use this directly as the prior mean.
        noise : observation noise (std dev). Models measurement error,
            e.g. a benchmark score of 55 might really be 53-57. Higher
            noise = gentler updates, more residual uncertainty. Default
            is 5% of the average feature spread in the population.
        method : inference method (selects which prior covariance the
            profile uses for Gaussian conditioning):

            - ``"kalman"`` (default): full Ledoit--Wolf covariance.
            - ``"diagonal"``: zero off-diagonal entries; no cross-feature
              transfer (the no-transfer ablation).
            - ``"block-diagonal"``: keep covariance only inside each
              block; zero across blocks. Requires ``blocks=``.
            - ``"pmf"``: rank-``rank`` eigentruncation of the covariance
              (PMF / probabilistic PCA prior). Requires ``rank=``.

            All methods use the same Gaussian conditioning machinery; only
            the prior covariance differs.
        rank : top-r eigencomponents to retain when ``method="pmf"``.
        blocks : block specification for ``method="block-diagonal"``.
            Either a list of feature-name (or index) lists --- one per
            block --- or a dict mapping feature name to block label.

        If neither prior_entity nor prior_mean is given, uses the population mean.
        """
        from skillinfer.state import Profile
        from skillinfer._kalman import (
            block_diagonal_covariance,
            diagonal_covariance,
            low_rank_covariance,
        )

        if prior_entity is not None:
            mu = self.entity(prior_entity)
        elif prior_mean is not None:
            mu = np.asarray(prior_mean, dtype=float).copy()
        else:
            mu = self.population_mean.copy()

        if method == "kalman":
            cov = self.covariance
        elif method == "diagonal":
            cov = diagonal_covariance(self.covariance)
        elif method == "block-diagonal":
            if blocks is None:
                raise ValueError(
                    "method='block-diagonal' requires blocks= "
                    "(list of feature lists, or {feature: block} dict)."
                )
            cov = block_diagonal_covariance(
                self.covariance, self._resolve_blocks(blocks),
            )
        elif method == "pmf":
            if rank is None:
                raise ValueError("method='pmf' requires rank=.")
            cov = low_rank_covariance(self.covariance, int(rank))
        else:
            raise ValueError(
                f"Unknown method: {method!r}. Choose from "
                "'kalman', 'diagonal', 'block-diagonal', 'pmf'."
            )

        if noise is None:
            noise = float(np.sqrt(np.diag(self.covariance)).mean() * 0.05)
            noise = max(noise, 1e-8)

        profile = Profile(
            prior_mean=mu,
            pop_cov=cov,
            feature_names=self.feature_names,
            noise=noise,
        )
        profile._skills = dict(self._skills)
        return profile

    def _resolve_blocks(
        self,
        blocks: list[list[str | int]] | dict[str, str | int],
    ) -> list[list[int]]:
        """Resolve a block spec to a list of integer-index lists."""
        if isinstance(blocks, dict):
            grouped: dict[object, list[int]] = {}
            for feat, label in blocks.items():
                if feat not in self._feature_to_idx:
                    raise KeyError(f"Unknown feature in blocks: {feat!r}")
                grouped.setdefault(label, []).append(self._feature_to_idx[feat])
            return list(grouped.values())
        resolved: list[list[int]] = []
        for block in blocks:
            idxs: list[int] = []
            for feat in block:
                if isinstance(feat, str):
                    if feat not in self._feature_to_idx:
                        raise KeyError(f"Unknown feature in blocks: {feat!r}")
                    idxs.append(self._feature_to_idx[feat])
                else:
                    idxs.append(int(feat))
            resolved.append(idxs)
        return resolved

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

    @property
    def covariance_df(self) -> pd.DataFrame:
        """Covariance matrix as a labeled DataFrame."""
        return pd.DataFrame(
            self.covariance,
            index=self.feature_names,
            columns=self.feature_names,
        )

    @property
    def correlation_df(self) -> pd.DataFrame:
        """Correlation matrix as a labeled DataFrame."""
        return pd.DataFrame(
            self.correlation,
            index=self.feature_names,
            columns=self.feature_names,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """The entity-feature matrix as a DataFrame (copy)."""
        return self.matrix.copy()

    def to_csv(self, path: str) -> None:
        """Export the entity-feature matrix to CSV.

        Parameters
        ----------
        path : file path to write.
        """
        self.matrix.to_csv(path)

    def to_parquet(self, path: str) -> None:
        """Export the entity-feature matrix to Parquet.

        Parameters
        ----------
        path : file path to write.
        """
        self.matrix.to_parquet(path)

    def summary(self) -> dict:
        """Summary statistics for this population.

        Returns a dict with:
        - n_entities, n_features: dimensions
        - shrinkage: Ledoit-Wolf coefficient (None if sample covariance)
        - condition_number: covariance matrix condition number
        - effective_dimensions: components needed for 90% variance
        - mean_correlation: average absolute off-diagonal correlation
        - sparsity: fraction of off-diagonal |correlations| below 0.1
        - top_correlations: top 5 feature pairs by |correlation|
        """
        N, K = self.matrix.shape
        pca = pca_embedding(self.matrix.values, min(K, 15))
        cum = pca["cumulative"]
        eff_dims = int(next((i + 1 for i, c in enumerate(cum) if c > 0.9), len(cum)))

        # Off-diagonal correlation stats
        corr = self.correlation
        mask = ~np.eye(K, dtype=bool)
        abs_corr = np.abs(corr[mask])
        mean_abs_corr = float(abs_corr.mean())
        sparsity = float((abs_corr < 0.1).mean())

        top = self.top_correlations(k=5)
        top_list = [
            {
                "feature_a": row["feature_a"],
                "feature_b": row["feature_b"],
                "correlation": float(row["correlation"]),
            }
            for _, row in top.iterrows()
        ]

        return {
            "n_entities": N,
            "n_features": K,
            "shrinkage": self.shrinkage,
            "condition_number": self.condition_number(),
            "effective_dimensions": eff_dims,
            "mean_correlation": mean_abs_corr,
            "sparsity": sparsity,
            "top_correlations": top_list,
        }

    def __repr__(self) -> str:
        N, K = self.matrix.shape
        s = f"Population({N} entities x {K} skills"
        if self.shrinkage is not None:
            s += f", shrinkage={self.shrinkage:.4f}"
        s += ")"
        return s

    def __str__(self) -> str:
        N, K = self.matrix.shape
        lines = [repr(self)]
        lines.append(f"  Condition number: {self.condition_number():.1f}")

        pca = pca_embedding(self.matrix.values, min(5, K))
        cum = pca["cumulative"]
        dims = next((i + 1 for i, c in enumerate(cum) if c > 0.9), len(cum))
        lines.append(f"  Effective dimensions: ~{dims} (90% variance)")

        lines.append("")
        lines.append("  Top skill correlations:")
        top = self.top_correlations(k=min(5, K * (K - 1) // 2))
        max_a = max(len(r["feature_a"]) for _, r in top.iterrows())
        max_b = max(len(r["feature_b"]) for _, r in top.iterrows())
        for _, row in top.iterrows():
            lines.append(
                f"    {row['feature_a']:>{max_a}} <-> {row['feature_b']:<{max_b}}"
                f"  r = {row['correlation']:+.3f}"
            )
        return "\n".join(lines)
