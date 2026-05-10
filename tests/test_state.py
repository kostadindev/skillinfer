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


def test_summary_basic(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    state.observe("physics", 0.8)
    s = state.summary()
    assert s["n_features"] == 6
    assert s["n_observed"] == 2
    assert s["n_predicted"] == 4
    assert 0.0 < s["mean_std"]
    assert 0.0 < s["uncertainty_reduction"] <= 1.0
    assert len(s["top_predicted"]) == 3
    assert len(s["most_uncertain"]) == 3
    # No accuracy metrics without true_vector
    assert "mae" not in s


def test_summary_with_ground_truth(taxonomy):
    state = taxonomy.profile()
    state.observe("math", 0.9)
    true_vec = taxonomy.entity(taxonomy.entity_names[0])
    s = state.summary(true_vector=true_vec)
    assert "mae" in s
    assert "rmse" in s
    assert "max_error" in s
    assert "cosine_similarity" in s
    assert "coverage_95" in s
    assert s["mae"] >= 0
    assert s["rmse"] >= 0
    assert s["max_error"] >= s["mae"]
    assert -1.0 <= s["cosine_similarity"] <= 1.0
    assert 0.0 <= s["coverage_95"] <= 1.0


# ---------------------------------------------------------------------------
# metrics_by_category
# ---------------------------------------------------------------------------

@pytest.fixture
def prefixed_taxonomy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.random((30, 6)),
        columns=[
            "Skill:reading", "Skill:math",
            "Knowledge:biology", "Knowledge:law",
            "Ability:strength", "Ability:dexterity",
        ],
    )
    return Population.from_dataframe(df)


def test_metrics_by_category_default_separator(prefixed_taxonomy):
    state = prefixed_taxonomy.profile()
    state.observe("Skill:math", 0.9)
    true_vec = prefixed_taxonomy.entity(prefixed_taxonomy.entity_names[0])
    df = state.metrics_by_category(true_vec)
    assert set(df["category"]) == {"Skill", "Knowledge", "Ability"}
    assert set(df.columns) == {
        "category", "n_features", "n_observed", "mae", "rmse", "recovery",
    }
    skill_row = df[df["category"] == "Skill"].iloc[0]
    assert skill_row["n_features"] == 2
    assert skill_row["n_observed"] == 1
    knowledge_row = df[df["category"] == "Knowledge"].iloc[0]
    assert knowledge_row["n_observed"] == 0


def test_metrics_by_category_custom_mapping(taxonomy):
    state = taxonomy.profile()
    true_vec = taxonomy.entity(taxonomy.entity_names[0])
    cats = {
        "math": "cognitive", "physics": "cognitive", "writing": "cognitive",
        "art": "creative",
        "strength": "physical", "speed": "physical",
    }
    df = state.metrics_by_category(true_vec, categories=cats)
    assert set(df["category"]) == {"cognitive", "creative", "physical"}
    assert df.loc[df["category"] == "cognitive", "n_features"].iloc[0] == 3


def test_metrics_by_category_uncategorised_bucket(taxonomy):
    state = taxonomy.profile()
    true_vec = taxonomy.entity(taxonomy.entity_names[0])
    df = state.metrics_by_category(true_vec)
    assert list(df["category"]) == ["uncategorised"]
    assert df["n_features"].iloc[0] == 6


def test_metrics_by_category_shape_mismatch(taxonomy):
    state = taxonomy.profile()
    with pytest.raises(ValueError, match="shape"):
        state.metrics_by_category(np.zeros(state.mu.size + 1))


def test_profile_method_diagonal_no_transfer(taxonomy):
    """method='diagonal' must not propagate observations to other features."""
    state = taxonomy.profile(method="diagonal")
    prior = state.mean()
    state.observe("math", 0.95)
    new = state.mean()
    # Observed dim moved.
    assert new[state._resolve_index("math")] != prior[state._resolve_index("math")]
    # All other dims unchanged.
    for f in ["physics", "writing", "art", "strength", "speed"]:
        j = state._resolve_index(f)
        assert new[j] == prior[j]


def test_profile_method_pmf_propagates_transfer():
    """method='pmf' retains transfer within the top-r eigenspace."""
    rng = np.random.default_rng(7)
    # Build correlated data: features 0-2 share a latent factor.
    z = rng.normal(size=(80, 1))
    block_a = z + 0.1 * rng.normal(size=(80, 3))
    block_b = rng.normal(size=(80, 3))
    data = np.hstack([block_a, block_b])
    df = pd.DataFrame(
        data,
        columns=["math", "physics", "writing", "art", "strength", "speed"],
    )
    pop = Population.from_dataframe(df)

    state = pop.profile(method="pmf", rank=2)
    prior = state.mean().copy()
    state.observe("math", 0.95)
    new = state.mean()
    # Within-block features should move with math under rank-2 PMF.
    moved = sum(
        not np.isclose(new[state._resolve_index(f)], prior[state._resolve_index(f)])
        for f in ["physics", "writing", "art", "strength", "speed"]
    )
    assert moved >= 1


def test_profile_method_pmf_requires_rank(taxonomy):
    with pytest.raises(ValueError, match="rank"):
        taxonomy.profile(method="pmf")


def test_profile_method_block_diagonal_blocks_transfer(taxonomy):
    """Across-block features must not move under block-diagonal."""
    blocks = [["math", "physics", "writing"], ["art", "strength", "speed"]]
    state = taxonomy.profile(method="block-diagonal", blocks=blocks)
    prior = state.mean().copy()
    state.observe("math", 0.95)
    new = state.mean()
    # At least one within-block feature may move.
    # Across-block features (art / strength / speed) must not.
    for f in ["art", "strength", "speed"]:
        j = state._resolve_index(f)
        assert np.isclose(new[j], prior[j], atol=1e-12)


def test_profile_method_block_diagonal_dict_form(taxonomy):
    """Dict block spec is equivalent to list-of-lists."""
    blocks_dict = {
        "math": "A", "physics": "A", "writing": "A",
        "art": "B", "strength": "B", "speed": "B",
    }
    state = taxonomy.profile(method="block-diagonal", blocks=blocks_dict)
    state.observe("math", 0.95)
    j_art = state._resolve_index("art")
    # Across-block: untouched.
    assert state.mean()[j_art] == taxonomy.population_mean[j_art]


def test_profile_method_block_diagonal_requires_blocks(taxonomy):
    with pytest.raises(ValueError, match="blocks"):
        taxonomy.profile(method="block-diagonal")


def test_profile_method_unknown_raises(taxonomy):
    with pytest.raises(ValueError, match="Unknown method"):
        taxonomy.profile(method="bogus")


def test_profile_method_kalman_default_matches_explicit(taxonomy):
    """Default profile() must match method='kalman' exactly."""
    a = taxonomy.profile()
    b = taxonomy.profile(method="kalman")
    a.observe("math", 0.9)
    b.observe("math", 0.9)
    np.testing.assert_allclose(a.mean(), b.mean(), atol=1e-12)

