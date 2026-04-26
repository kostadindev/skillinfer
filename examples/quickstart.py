"""
Quickstart: predict a student's full grade profile from one exam score.

Generates 200 synthetic students with correlated subject scores,
builds a taxonomy, then shows how observing a single subject
predicts all the others.

Run:
    python examples/quickstart.py
"""

import numpy as np
import pandas as pd
import skillinfer


# ── Step 1: Generate synthetic students with correlated scores ────────

rng = np.random.default_rng(0)
N = 200
subjects = ["Math", "Physics", "Chemistry", "English", "History", "Art"]

# Latent factor: general academic ability
g = rng.normal(70, 12, N)

# STEM subjects correlate with each other, humanities with each other
scores = pd.DataFrame({
    "Math":      g + rng.normal(0, 6, N),
    "Physics":   g * 0.9 + rng.normal(5, 7, N),
    "Chemistry": g * 0.85 + rng.normal(8, 8, N),
    "English":   -0.3 * g + rng.normal(90, 10, N),
    "History":   -0.2 * g + rng.normal(80, 9, N),
    "Art":       rng.normal(65, 15, N),  # nearly independent
}).clip(0, 100)

scores.index = [f"student_{i}" for i in range(N)]

print("Generated data:")
print(f"  {N} students x {len(subjects)} subjects")
print(f"  Score ranges: {scores.min().min():.0f} – {scores.max().max():.0f}")
print()

# ── Step 2: Hold out one student, build taxonomy from the rest ─────────

target = "student_42"
true_scores = scores.loc[target].values.copy()

# Build taxonomy from everyone EXCEPT the target (no data leakage)
pop = skillinfer.Population.from_dataframe(scores.drop(target), normalize=False)
print(f"Population: {pop}")
print(f"Condition number: {pop.condition_number():.1f}")
print()

# Show the covariance structure
print("Top correlations:")
for _, row in pop.top_correlations(k=5).iterrows():
    print(f"  {row['feature_a']:>10} <-> {row['feature_b']:<10}  r = {row['correlation']:+.3f}")
print()

print(f"{'=' * 55}")
print(f"  Target: {target}")
print(f"  True scores: {dict(zip(subjects, [f'{v:.1f}' for v in true_scores]))}")
print(f"{'=' * 55}")
print()

# Observe only Math
state = pop.profile()
state.observe("Math", true_scores[subjects.index("Math")])

print(f"After observing Math = {true_scores[subjects.index('Math')]:.1f}:")
print()
print(f"  {'Subject':<12} {'True':>6} {'Predicted':>10} {'± Std':>8} {'Error':>7}")
print(f"  {'-' * 47}")

errors = []
for i, subj in enumerate(subjects):
    pred = state.mean(subj)
    std = state.std(subj)
    err = abs(true_scores[i] - pred)
    tag = "  <-- observed" if subj == "Math" else ""
    print(f"  {subj:<12} {true_scores[i]:>6.1f} {pred:>10.1f} {std:>8.1f} {err:>7.1f}{tag}")
    if subj != "Math":
        errors.append(err)

cosine = state.similarity(true_scores)
print()
print(f"  MAE on unobserved: {np.mean(errors):.1f}")
print(f"  Cosine similarity: {cosine:.4f}")

# ── Step 4: What to observe next? ─────────────────────────────────────

print()
print("Most uncertain (observe next):")
for _, row in state.most_uncertain(k=3).iterrows():
    print(f"  {row['feature']}: std = {row['std']:.1f}")

# ── Step 5: Observe a second subject and see improvement ──────────────

second = state.most_uncertain(k=1).iloc[0]["feature"]
state.observe(second, true_scores[subjects.index(second)])

remaining = [
    abs(true_scores[i] - state.mean(subj))
    for i, subj in enumerate(subjects)
    if subj not in ("Math", second)
]
cosine2 = state.similarity(true_scores)

print()
print(f"After also observing {second} = {true_scores[subjects.index(second)]:.1f}:")
print(f"  MAE on remaining {len(remaining)} subjects: {np.mean(remaining):.1f}")
print(f"  Cosine similarity: {cosine2:.4f}  (+{cosine2 - cosine:.4f})")

# ── Step 6: Validate that transfer helps ──────────────────────────────

print()
print("Held-out validation (does covariance transfer help?):")
results = skillinfer.validation.held_out_evaluation(
    pop, frac_observed=[0.2, 0.5], n_splits=5, obs_noise=2.0,
)
summary = results.groupby(["frac_observed", "method"])["cosine_similarity"].mean()
print(summary.to_string())
