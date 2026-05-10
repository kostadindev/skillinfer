"""O*NET 30.2 dataset loader.

894 occupations x 120 features (35 skills, 33 knowledge areas, 52 abilities).
Each value is an importance rating normalised to [0, 1].

Source: O*NET 30.2 Database, U.S. Department of Labor / ETA.
License: CC BY 4.0 — https://www.onetcenter.org/database.html
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_DATA = Path(__file__).parent / "onet.csv.gz"


def onet(normalize: bool = False) -> "Population":
    """Load the O*NET 30.2 population (894 occupations x 120 features).

    The data is already normalised to [0, 1], so ``normalize`` defaults
    to False. Pass ``normalize=True`` to re-normalise if you want.

    Returns
    -------
    Population
        Ready for ``pop.profile()`` / ``pop.describe_skills()`` etc.

    Examples
    --------
    >>> import skillinfer
    >>> pop = skillinfer.datasets.onet()
    >>> profile = pop.profile()
    >>> profile.observe("Skill:Programming", 0.92)
    >>> print(profile.predict())
    """
    from skillinfer.population import Population

    df = pd.read_csv(_DATA, index_col=0, compression="gzip")
    return Population.from_dataframe(df, normalize=normalize)
