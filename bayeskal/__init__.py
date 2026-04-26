"""
bayeskal: Predict all features from a few observations.

Bayesian feature inference via Kalman filtering with learned covariance.
Given a population of entities described by K features, bayeskal learns
how features co-vary and uses that structure to infer unobserved features
from partial observations.

Quick start::

    import bayeskal

    tax = bayeskal.Taxonomy.from_dataframe(df)   # learn covariance
    state = tax.new_state(obs_noise=0.05)         # new entity
    state.observe("math", 0.9)                    # one observation
    print(state.mean("physics"))                  # predict another
    print(state.most_uncertain(k=5))              # what to observe next?

Classes:
    Taxonomy       — Population model (entity-feature matrix + covariance)
    InferenceState — Posterior belief about one entity (mu + Sigma)

Modules:
    analysis       — Visualization (correlation heatmap, scree plot, etc.)
    validation     — Held-out evaluation (does transfer help?)
"""

from bayeskal.taxonomy import Taxonomy
from bayeskal.state import InferenceState
from bayeskal import analysis
from bayeskal import validation

__version__ = "0.1.0"
__all__ = ["Taxonomy", "InferenceState", "analysis", "validation"]
