"""ESCO v1.2.1 dataset loader.

2,999 occupations x 134 skill groups (binary: 1 if the occupation requires
at least one essential skill in that group, 0 otherwise). Skill groups are
Level-2 categories from the ESCO skill hierarchy.

Source: ESCO v1.2.1, European Commission.
License: https://esco.ec.europa.eu/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_PARQUET = Path(__file__).parent / "esco.parquet"


def esco(normalize: bool = False) -> "Population":
    """Load the ESCO v1.2.1 population (2,999 occupations x 134 skill groups).

    The data is binary (0/1), so ``normalize`` defaults to False.

    Returns
    -------
    Population
        Ready for ``pop.profile()`` / ``pop.describe_skills()`` etc.

    Examples
    --------
    >>> import skillinfer
    >>> pop = skillinfer.datasets.esco()
    >>> profile = pop.profile()
    >>> profile.observe("education", 1.0)
    >>> print(profile.predict())
    """
    from skillinfer.population import Population

    df = pd.read_parquet(_PARQUET)
    return Population.from_dataframe(df, normalize=normalize)
