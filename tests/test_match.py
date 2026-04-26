"""Tests for match_score() and rank_agents()."""

import numpy as np
import pandas as pd
import pytest

from skillinfer import Population, Profile, MatchResult, rank_agents


@pytest.fixture
def taxonomy():
    rng = np.random.default_rng(42)
    data = rng.random((20, 6))
    df = pd.DataFrame(
        data,
        columns=["math", "physics", "writing", "art", "strength", "speed"],
    )
    return Population.from_dataframe(df)


# --- match_score tests ---

def test_match_score_dict(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    result = state.match_score({"math": 1.0, "physics": 0.5})
    assert isinstance(result, MatchResult)
    assert result.score > 0
    assert result.std > 0
    assert result.ci_lower < result.score < result.ci_upper
    assert result.p_above_threshold is None


def test_match_score_array(taxonomy):
    state = taxonomy.profile()
    w = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    result = state.match_score(w)
    assert isinstance(result, MatchResult)
    assert result.score > 0


def test_match_score_with_threshold(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    result = state.match_score({"math": 1.0}, threshold=0.5)
    assert result.p_above_threshold is not None
    assert 0.0 <= result.p_above_threshold <= 1.0


def test_match_score_unknown_feature(taxonomy):
    state = taxonomy.profile()
    with pytest.raises(KeyError, match="Unknown feature"):
        state.match_score({"nonexistent": 1.0})


def test_match_score_wrong_shape(taxonomy):
    state = taxonomy.profile()
    with pytest.raises(ValueError, match="shape"):
        state.match_score(np.array([1.0, 2.0]))


def test_match_score_named_tuple_access(taxonomy):
    state = taxonomy.profile()
    result = state.match_score({"math": 1.0})
    assert result.score == result[0]
    assert result.std == result[1]


# --- rank_agents tests ---

def test_rank_agents_basic(taxonomy):
    states = {}
    for i, name in enumerate(taxonomy.entity_names[:5]):
        s = taxonomy.profile()
        s.observe("math", taxonomy.entity(name)[0])
        states[name] = s

    df = rank_agents({"math": 1.0, "physics": 0.5}, states)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ["agent", "expected_score", "std", "p_above_threshold"]
    assert df["expected_score"].is_monotonic_decreasing


def test_rank_agents_with_threshold(taxonomy):
    states = {"a": taxonomy.profile(), "b": taxonomy.profile()}
    states["a"].observe("math", 0.9)
    states["b"].observe("math", 0.1)
    df = rank_agents({"math": 1.0}, states, threshold=0.5)
    assert df.iloc[0]["p_above_threshold"] > df.iloc[1]["p_above_threshold"]


def test_rank_agents_empty(taxonomy):
    df = rank_agents({"math": 1.0}, {})
    assert len(df) == 0


# --- Input validation tests ---


def test_index_bounds_checking(taxonomy):
    state = taxonomy.profile()
    with pytest.raises(IndexError, match="out of range"):
        state.observe(999, 0.5)
    with pytest.raises(IndexError, match="out of range"):
        state.observe(-1, 0.5)


def test_nan_in_dataframe():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    with pytest.raises(ValueError, match="NaN"):
        Population.from_dataframe(df)


# --- copy() bug fix ---

def test_copy_preserves_n_observations(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    state.observe("physics", 0.8)
    copy = state.copy()
    assert copy.n_observations == 2


# --- from_covariance tests ---

def test_from_covariance():
    K = 3
    cov = np.eye(K) * 0.1
    names = ["a", "b", "c"]
    mean = np.array([1.0, 2.0, 3.0])
    tax = Population.from_covariance(cov, names, mean)
    assert tax.feature_names == names
    np.testing.assert_array_equal(tax.population_mean, mean)
    np.testing.assert_array_equal(tax.covariance, cov)
    state = tax.profile()
    np.testing.assert_array_equal(state.mu, mean)


def test_from_covariance_shape_mismatch():
    with pytest.raises(ValueError, match="Covariance shape"):
        Population.from_covariance(np.eye(2), ["a", "b", "c"], np.zeros(3))
    with pytest.raises(ValueError, match="Mean shape"):
        Population.from_covariance(np.eye(3), ["a", "b", "c"], np.zeros(2))
