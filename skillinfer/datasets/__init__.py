"""Built-in datasets for skillinfer.

Each loader returns a Population ready for profiling.

Bundled data
------------
onet()         — O*NET 30.2: 894 occupations x 120 features (skills,
                 knowledge, abilities). U.S. Department of Labor / ETA,
                 CC BY 4.0.
esco()         — ESCO v1.2.1: 2,999 occupations x 134 skill groups
                 (binary). European Commission.
piaac_prior()  — PIAAC Cycle 2 US within-individual prior over 9
                 dimensions (n=2,548 adults). Summary statistics only —
                 no individual records bundled.
"""

from skillinfer.datasets._onet import onet
from skillinfer.datasets._esco import esco
from skillinfer.datasets._piaac import piaac_prior

__all__ = ["onet", "esco", "piaac_prior"]
