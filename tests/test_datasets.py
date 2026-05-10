"""Tests for bundled datasets."""

import numpy as np
import pytest

import skillinfer
from skillinfer import Population, Profile


def test_onet_loads():
    pop = skillinfer.datasets.onet()
    assert isinstance(pop, Population)
    n, k = pop.matrix.shape
    assert n == 894 and k == 120


def test_esco_loads():
    pop = skillinfer.datasets.esco()
    assert isinstance(pop, Population)
    n, k = pop.matrix.shape
    assert n == 2999 and k == 134


# ---------------------------------------------------------------------------
# PIAAC prior — covariance-only, 9 dimensions
# ---------------------------------------------------------------------------

EXPECTED_PIAAC_FEATURES = [
    "literacy", "numeracy", "problem_solving",
    "readwork", "writwork", "numwork",
    "ictwork", "influence", "taskdisc",
]


@pytest.fixture(scope="module")
def piaac():
    return skillinfer.datasets.piaac_prior()


def test_piaac_prior_shape_and_features(piaac):
    assert piaac.feature_names == EXPECTED_PIAAC_FEATURES
    assert piaac.population_mean.shape == (9,)
    assert piaac.covariance.shape == (9, 9)


def test_piaac_prior_is_covariance_only(piaac):
    # Single placeholder row in the matrix — this is a from_covariance
    # population, not a row-bearing one.
    assert piaac.matrix.shape == (1, 9)


def test_piaac_prior_mean_inside_unit_box(piaac):
    # The bundled data was min-max scaled to [0, 1] before fitting, so
    # the mean should lie strictly inside the unit cube.
    assert (piaac.population_mean > 0.0).all()
    assert (piaac.population_mean < 1.0).all()


def test_piaac_prior_covariance_psd(piaac):
    eigenvalues = np.linalg.eigvalsh(piaac.covariance)
    assert (eigenvalues > -1e-9).all()


def test_piaac_prior_recovers_assessed_block_correlation(piaac):
    # The chapter's headline finding: the three assessed cognitive
    # scores are tightly correlated. Cross-check on the bundled prior.
    corr = piaac.correlation
    names = piaac.feature_names
    i, j, k = names.index("literacy"), names.index("numeracy"), names.index("problem_solving")
    assert corr[i, j] > 0.85
    assert corr[i, k] > 0.85
    assert corr[j, k] > 0.85


def test_piaac_prior_supports_profile_pipeline(piaac):
    profile = piaac.profile()
    assert isinstance(profile, Profile)
    profile.observe("literacy", 0.85)
    df = profile.predict()
    assert len(df) == 9
    # Observing literacy should move numeracy and problem_solving up
    # (they are the two strongest correlates).
    num = df[df["feature"] == "numeracy"]["mean"].iloc[0]
    ps = df[df["feature"] == "problem_solving"]["mean"].iloc[0]
    base_num = piaac.population_mean[piaac.feature_names.index("numeracy")]
    base_ps = piaac.population_mean[piaac.feature_names.index("problem_solving")]
    assert num > base_num
    assert ps > base_ps
