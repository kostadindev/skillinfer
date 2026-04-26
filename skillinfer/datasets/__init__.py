"""Built-in datasets for skillinfer.

Each loader returns a Population ready for profiling.

Bundled data
------------
onet()  — O*NET 30.2: 894 occupations x 120 features (skills, knowledge, abilities).
           U.S. Department of Labor / ETA, CC BY 4.0.
esco()  — ESCO v1.2.1: 2,999 occupations x 134 skill groups (binary).
           European Commission.
"""

from skillinfer.datasets._onet import onet
from skillinfer.datasets._esco import esco

__all__ = ["onet", "esco"]
