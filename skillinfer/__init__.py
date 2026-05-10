"""
skillinfer: Predict all features from a few observations.

Bayesian feature inference via Kalman filtering with learned covariance.
Given a population of entities described by K features, skillinfer learns
how features co-vary and uses that structure to infer unobserved features
from partial observations.

Quick start::

    import pandas as pd
    import skillinfer

    df = pd.read_parquet("hf://datasets/open-llm-leaderboard/...")
    pop = skillinfer.Population.from_dataframe(df)
    profile = pop.profile()
    profile.observe("BBH", 55.0)
    print(profile.predict())

Classes:
    Population — Learned covariance structure from a population
    Profile    — Skill profile for one entity (gets sharper with observations)

Modules:
    visualization — Plotting (correlation heatmap, scree plot, etc.)
    validation — Held-out evaluation (does transfer help?)
"""

from skillinfer.types import Skill, Task
from skillinfer.population import Population
from skillinfer.state import Profile, GMMProfile, MatchResult
from skillinfer import visualization
from skillinfer import datasets
from skillinfer import validation

import numpy as np
import pandas as pd


def rank_agents(
    task_vector: dict[str, float] | np.ndarray | Task,
    profiles: dict[str, Profile],
    threshold: float | None = None,
) -> pd.DataFrame:
    """Rank agents by expected performance on a task.

    Parameters
    ----------
    task_vector : dict mapping feature names to importance weights,
        or a (K,) numpy array.
    profiles : dict mapping agent name/ID to their Profile.
    threshold : if given, include P(score > threshold) in results.

    Returns
    -------
    DataFrame sorted by expected_score (descending) with columns:
        [agent, expected_score, std, p_above_threshold]
    """
    if not profiles:
        return pd.DataFrame(
            columns=["agent", "expected_score", "std", "p_above_threshold"]
        )
    rows = []
    for name, profile in profiles.items():
        result = profile.match_score(task_vector, threshold=threshold)
        rows.append({
            "agent": name,
            "expected_score": result.score,
            "std": result.std,
            "p_above_threshold": result.p_above_threshold,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("expected_score", ascending=False).reset_index(drop=True)


__version__ = "0.1.1"
__all__ = [
    "Skill",
    "Task",
    "Population",
    "Profile",
    "GMMProfile",
    "MatchResult",
    "rank_agents",
    "visualization",
    "validation",
]
