"""Covariance estimation internals."""

import numpy as np
from sklearn.covariance import LedoitWolf


def ledoit_wolf_covariance(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance estimate.

    Parameters
    ----------
    R : (N, K) entity-feature matrix

    Returns
    -------
    Sigma : (K, K) shrinkage covariance matrix
    alpha : shrinkage coefficient
    """
    lw = LedoitWolf().fit(R)
    return lw.covariance_, lw.shrinkage_


def sample_covariance(R: np.ndarray) -> np.ndarray:
    """Unregularized sample covariance with ridge for stability.

    Parameters
    ----------
    R : (N, K) entity-feature matrix

    Returns
    -------
    Sigma : (K, K) covariance matrix
    """
    Sigma = np.cov(R, rowvar=False)
    Sigma += np.eye(Sigma.shape[0]) * 1e-6
    return Sigma


def correlation_matrix(Sigma: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to Pearson correlation matrix.

    Parameters
    ----------
    Sigma : (K, K) covariance matrix

    Returns
    -------
    Corr : (K, K) correlation matrix with unit diagonal
    """
    d = np.sqrt(np.diag(Sigma))
    d = np.where(d > 0, d, 1.0)
    return Sigma / np.outer(d, d)


def pca_embedding(
    R: np.ndarray,
    n_components: int = 15,
) -> dict:
    """PCA decomposition of the entity-feature matrix.

    Parameters
    ----------
    R : (N, K) entity-feature matrix
    n_components : number of principal components

    Returns
    -------
    dict with keys:
        components : (n_components, K) principal component vectors
        explained_variance_ratio : (n_components,) fraction of variance per component
        cumulative : (n_components,) cumulative variance explained
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, R.shape[0], R.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(R)

    return {
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    }
