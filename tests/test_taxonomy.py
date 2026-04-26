"""Tests for Taxonomy class."""

import numpy as np
import pandas as pd
import pytest

from bayeskal import Taxonomy


@pytest.fixture
def sample_df():
    """5 entities x 4 features with known structure."""
    rng = np.random.default_rng(42)
    data = rng.random((5, 4))
    return pd.DataFrame(
        data,
        index=["a", "b", "c", "d", "e"],
        columns=["f1", "f2", "f3", "f4"],
    )


def test_from_dataframe(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    assert tax.matrix.shape == (5, 4)
    assert len(tax.feature_names) == 4
    assert len(tax.entity_names) == 5
    assert tax.covariance.shape == (4, 4)
    assert tax.shrinkage is not None


def test_from_dataframe_normalize(sample_df):
    tax = Taxonomy.from_dataframe(sample_df, normalize=True)
    assert tax.matrix.values.min() >= -1e-10
    assert tax.matrix.values.max() <= 1.0 + 1e-10


def test_from_dataframe_no_normalize(sample_df):
    tax = Taxonomy.from_dataframe(sample_df, normalize=False)
    np.testing.assert_array_equal(tax.matrix.values, sample_df.values)


def test_entity(sample_df):
    tax = Taxonomy.from_dataframe(sample_df, normalize=False)
    vec = tax.entity("a")
    np.testing.assert_array_almost_equal(vec, sample_df.loc["a"].values)


def test_entity_not_found(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    with pytest.raises(KeyError):
        tax.entity("nonexistent")


def test_new_state_population_mean(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    state = tax.new_state()
    np.testing.assert_array_almost_equal(state.mu, tax.population_mean)


def test_new_state_prior_entity(sample_df):
    tax = Taxonomy.from_dataframe(sample_df, normalize=False)
    state = tax.new_state(prior_entity="b")
    np.testing.assert_array_almost_equal(state.mu, sample_df.loc["b"].values)


def test_new_state_prior_mean(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    custom = np.array([0.1, 0.2, 0.3, 0.4])
    state = tax.new_state(prior_mean=custom)
    np.testing.assert_array_almost_equal(state.mu, custom)


def test_pca(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    pca = tax.pca(n_components=3)
    assert "explained_variance_ratio" in pca
    assert "cumulative" in pca
    assert len(pca["explained_variance_ratio"]) == 3
    assert pca["cumulative"][-1] <= 1.0 + 1e-10


def test_top_correlations(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    top = tax.top_correlations(k=3)
    assert len(top) == 3
    assert "feature_a" in top.columns
    assert "feature_b" in top.columns
    assert "correlation" in top.columns


def test_condition_number(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    cond = tax.condition_number()
    assert cond >= 1.0


def test_repr(sample_df):
    tax = Taxonomy.from_dataframe(sample_df)
    r = repr(tax)
    assert "5 entities" in r
    assert "4 features" in r
    assert "shrinkage" in r


def test_sample_covariance(sample_df):
    tax = Taxonomy.from_dataframe(sample_df, covariance="sample")
    assert tax.shrinkage is None
    assert tax.covariance.shape == (4, 4)
