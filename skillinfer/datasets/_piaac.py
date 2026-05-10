"""PIAAC Cycle 2 US prior loader.

Ships the within-individual prior (mean + covariance) derived from the
OECD PIAAC Cycle 2 US public-use file. Only the summary statistics are
bundled — no individual records. The 9 dimensions are:

    literacy, numeracy, problem_solving      (mean of 10 IRT plausible values)
    readwork, writwork, numwork,             (skill-use-at-work WLEs)
    ictwork, influence, taskdisc

Each dimension was min-max scaled to [0, 1] within the PIAAC sample
before computing covariance, so the prior is consumable by the standard
[0, 1] scoring contract used elsewhere in the package. Min-max scaling
preserves the correlation structure.

Source: PIAAC Cycle 2 US PUF (NCES product 2024-XX, OECD).
        https://nces.ed.gov/surveys/piaac/datafiles.asp

Use as a within-individual prior — i.e. when you want to profile a
specific person and your population covariance is "how individuals'
skills covary" (not "how occupations differ"). For occupation-level
priors use ``onet()`` or ``esco()`` instead.

Examples
--------
>>> import skillinfer
>>> pop = skillinfer.datasets.piaac_prior()
>>> profile = pop.profile()
>>> profile.observe("literacy", 0.72)
>>> profile.predict()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_DATA = Path(__file__).parent / "piaac_prior.npz"


def piaac_prior() -> "Population":
    """Within-individual prior from the PIAAC Cycle 2 US sample (n=2,548 adults).

    Returns a Population with no entity rows: only the population mean
    and covariance over 9 skill / skill-use dimensions, scaled to [0, 1].
    Use ``pop.profile()`` to start a posterior, then ``observe()``
    individual-level scores (also scaled to [0, 1]).

    Returns
    -------
    Population
        9-feature population built via ``Population.from_covariance``.
        ``pop.matrix`` contains a single placeholder row (the mean);
        ``pop.covariance`` is the bundled prior.
    """
    from skillinfer.population import Population

    data = np.load(_DATA, allow_pickle=False)
    feature_names = [str(n) for n in data["feature_names"]]
    return Population.from_covariance(
        covariance=data["cov"],
        feature_names=feature_names,
        population_mean=data["mean"],
    )
