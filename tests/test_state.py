"""Tests for Profile class."""

import numpy as np
import pandas as pd
import pytest

from skillinfer import Population, Profile


@pytest.fixture
def taxonomy():
    rng = np.random.default_rng(42)
    data = rng.random((20, 6))
    df = pd.DataFrame(
        data,
        columns=["math", "physics", "writing", "art", "strength", "speed"],
    )
    return Population.from_dataframe(df)


def test_observe_updates_mean(taxonomy):
    state = taxonomy.profile()
    old_mean = state.mean("math")
    state.observe("math", 0.95)
    new_mean = state.mean("math")
    assert new_mean != old_mean
    assert abs(new_mean - 0.95) < abs(old_mean - 0.95)


def test_observe_returns_self(taxonomy):
    state = taxonomy.profile()
    result = state.observe("math", 0.5)
    assert result is state


def test_observe_many(taxonomy):
    state1 = taxonomy.profile()
    state2 = taxonomy.profile()

    # Sequential
    state1.observe("math", 0.9).observe("physics", 0.8)

    # Batch
    state2.observe_many({"math": 0.9, "physics": 0.8})

    np.testing.assert_allclose(state1.mu, state2.mu, atol=1e-12)


def test_observe_increments_count(taxonomy):
    state = taxonomy.profile()
    assert state.n_observations == 0
    state.observe("math", 0.5)
    assert state.n_observations == 1
    state.observe_many({"physics": 0.3, "writing": 0.7})
    assert state.n_observations == 3


def test_observe_unknown_feature(taxonomy):
    state = taxonomy.profile()
    with pytest.raises(KeyError):
        state.observe("nonexistent", 0.5)


def test_observe_by_index(taxonomy):
    state = taxonomy.profile()
    state.observe(0, 0.9)  # integer index
    assert state.n_observations == 1


def test_std_decreases_after_observation(taxonomy):
    state = taxonomy.profile()
    old_std = state.std("math")
    state.observe("math", 0.5)
    new_std = state.std("math")
    assert new_std < old_std


def test_confidence_interval(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.7)
    lo, hi = state.confidence_interval("math", level=0.95)
    mu = state.mean("math")
    assert lo < mu < hi
    assert hi - lo > 0


def test_most_uncertain(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.5)
    df = state.most_uncertain(k=3)
    assert len(df) == 3
    assert "feature" in df.columns
    assert "std" in df.columns
    # "math" should NOT be the most uncertain anymore
    assert df.iloc[0]["feature"] != "math"


def test_to_dataframe(taxonomy):
    state = taxonomy.profile()
    df = state.to_dataframe()
    assert len(df) == 6
    assert {"feature", "mean", "std"} <= set(df.columns)
    assert "source" not in df.columns  # detail=False by default
    # detail=True adds confidence and source
    df2 = state.to_dataframe(detail=True)
    assert "source" in df2.columns
    assert "confidence" in df2.columns


def test_similarity(taxonomy):
    state = taxonomy.profile(prior_entity=taxonomy.entity_names[0])
    vec = taxonomy.entity(taxonomy.entity_names[0])
    sim = state.similarity(vec)
    assert sim > 0.99  # same vector should be ~1.0


def test_copy_independence(taxonomy):
    state = taxonomy.profile()
    copy = state.copy()
    copy.observe("math", 0.9)
    assert copy.n_observations == 1
    assert state.n_observations == 0
    assert not np.allclose(copy.mu, state.mu)


def test_predict_single(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    pred = state.predict("physics")
    assert isinstance(pred, dict)
    assert "mean" in pred
    assert "std" in pred
    assert "ci_lower" in pred
    assert "ci_upper" in pred
    assert pred["ci_lower"] < pred["mean"] < pred["ci_upper"]


def test_predict_all(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    df = state.predict()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert {"feature", "mean", "std", "ci_lower", "ci_upper"} <= set(df.columns)
    assert "source" not in df.columns  # detail=False by default
    assert (df["ci_lower"] < df["mean"]).all()
    assert (df["mean"] < df["ci_upper"]).all()

    # detail=True adds confidence and source
    df2 = state.predict(detail=True)
    math_row = df2[df2["feature"] == "math"].iloc[0]
    assert math_row["source"] == "observed"
    assert math_row["confidence"] > 0.9
    phys_row = df2[df2["feature"] == "physics"].iloc[0]
    assert phys_row["source"] == "predicted"


def test_agent_vector(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    av = state.agent_vector
    assert isinstance(av, pd.Series)
    assert av.name == "agent_vector"
    assert list(av.index) == taxonomy.feature_names
    np.testing.assert_array_almost_equal(av.values, state.mu)


def test_covariance_matrix(taxonomy):
    state = taxonomy.profile()
    cov = state.covariance_matrix
    assert isinstance(cov, pd.DataFrame)
    assert list(cov.columns) == taxonomy.feature_names
    assert list(cov.index) == taxonomy.feature_names


def test_str(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    s = str(state)
    assert "Agent Vector" in s
    assert "1 observations" in s
    assert "math" in s


def test_repr(taxonomy):
    state = taxonomy.profile()
    r = repr(state)
    assert "K=6" in r
    assert "n_obs=0" in r
