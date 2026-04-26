"""Tests for InferenceState class."""

import numpy as np
import pandas as pd
import pytest

from bayeskal import Taxonomy, InferenceState


@pytest.fixture
def taxonomy():
    rng = np.random.default_rng(42)
    data = rng.random((20, 6))
    df = pd.DataFrame(
        data,
        columns=["math", "physics", "writing", "art", "strength", "speed"],
    )
    return Taxonomy.from_dataframe(df)


def test_observe_updates_mean(taxonomy):
    state = taxonomy.new_state()
    old_mean = state.mean("math")
    state.observe("math", 0.95)
    new_mean = state.mean("math")
    assert new_mean != old_mean
    assert abs(new_mean - 0.95) < abs(old_mean - 0.95)


def test_observe_returns_self(taxonomy):
    state = taxonomy.new_state()
    result = state.observe("math", 0.5)
    assert result is state


def test_observe_many(taxonomy):
    state1 = taxonomy.new_state()
    state2 = taxonomy.new_state()

    # Sequential
    state1.observe("math", 0.9).observe("physics", 0.8)

    # Batch
    state2.observe_many({"math": 0.9, "physics": 0.8})

    np.testing.assert_allclose(state1.mu, state2.mu, atol=1e-12)


def test_observe_increments_count(taxonomy):
    state = taxonomy.new_state()
    assert state.n_observations == 0
    state.observe("math", 0.5)
    assert state.n_observations == 1
    state.observe_many({"physics": 0.3, "writing": 0.7})
    assert state.n_observations == 3


def test_observe_unknown_feature(taxonomy):
    state = taxonomy.new_state()
    with pytest.raises(KeyError):
        state.observe("nonexistent", 0.5)


def test_observe_by_index(taxonomy):
    state = taxonomy.new_state()
    state.observe(0, 0.9)  # integer index
    assert state.n_observations == 1


def test_std_decreases_after_observation(taxonomy):
    state = taxonomy.new_state()
    old_std = state.std("math")
    state.observe("math", 0.5)
    new_std = state.std("math")
    assert new_std < old_std


def test_confidence_interval(taxonomy):
    state = taxonomy.new_state()
    state.observe("math", 0.7)
    lo, hi = state.confidence_interval("math", level=0.95)
    mu = state.mean("math")
    assert lo < mu < hi
    assert hi - lo > 0


def test_most_uncertain(taxonomy):
    state = taxonomy.new_state()
    state.observe("math", 0.5)
    df = state.most_uncertain(k=3)
    assert len(df) == 3
    assert "feature" in df.columns
    assert "std" in df.columns
    # "math" should NOT be the most uncertain anymore
    assert df.iloc[0]["feature"] != "math"


def test_to_dataframe(taxonomy):
    state = taxonomy.new_state()
    df = state.to_dataframe()
    assert len(df) == 6
    assert set(df.columns) == {"feature", "mean", "std"}


def test_similarity(taxonomy):
    state = taxonomy.new_state(prior_entity=taxonomy.entity_names[0])
    vec = taxonomy.entity(taxonomy.entity_names[0])
    sim = state.similarity(vec)
    assert sim > 0.99  # same vector should be ~1.0


def test_copy_independence(taxonomy):
    state = taxonomy.new_state()
    copy = state.copy()
    copy.observe("math", 0.9)
    assert copy.n_observations == 1
    assert state.n_observations == 0
    assert not np.allclose(copy.mu, state.mu)


def test_repr(taxonomy):
    state = taxonomy.new_state()
    r = repr(state)
    assert "K=6" in r
    assert "n_obs=0" in r
