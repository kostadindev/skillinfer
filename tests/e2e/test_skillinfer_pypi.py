"""E2E smoke test for `skillinfer` installed from PyPI.

Run inside an env where `pip install skillinfer` has succeeded.
Exits non-zero on any assertion failure.
"""
from __future__ import annotations

import sys

import skillinfer
from skillinfer import Population, Profile


def main() -> None:
    assert skillinfer.__version__ == "0.1.1", skillinfer.__version__

    pop = skillinfer.datasets.onet()
    N, K = pop.matrix.shape
    assert N > 100 and K > 50, (N, K)

    profile = pop.profile()
    assert isinstance(profile, Profile)
    assert profile.n_observations == 0

    # Pick two real feature names from the population.
    f0, f1 = pop.feature_names[0], pop.feature_names[1]
    profile.observe(f0, 0.8)
    assert profile.n_observations == 1

    profile.observe_many({f1: 0.6})
    assert profile.n_observations == 2

    df = profile.predict(detail=True)
    assert len(df) == K
    assert {"mean", "std"}.issubset(df.columns), df.columns.tolist()

    unc = profile.most_uncertain(k=3)
    assert len(unc) == 3

    # rank_agents end-to-end.
    p2 = pop.profile()
    p2.observe(f0, 0.2)
    ranked = skillinfer.rank_agents({f0: 1.0, f1: 0.5}, {"a": profile, "b": p2})
    assert list(ranked["agent"])[0] in {"a", "b"}
    assert ranked["expected_score"].iloc[0] >= ranked["expected_score"].iloc[-1]

    print(f"OK  skillinfer {skillinfer.__version__}: {N}x{K} pop, predict + rank_agents pass")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL  {e}", file=sys.stderr)
        sys.exit(1)
