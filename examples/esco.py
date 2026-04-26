"""
ESCO: Cross-taxonomy validation with the European Skills taxonomy.

Uses ESCO v1.2.1 (European Commission) — an independently curated taxonomy
with binary skill assignments across ~3,000 occupations. Individual skills
(~13K) are aggregated to Level-2 skill groups via the ESCO hierarchy,
yielding a binary occupation x skill-group matrix.

ESCO differs from O*NET in three ways:
  - Curated by EU expert panels (not U.S. DoL surveys)
  - Binary assignments (not continuous ratings)
  - Independently defined skill taxonomy (134 groups vs 120 features)

These differences make ESCO a strong cross-taxonomy validation: if the
Kalman filter works on both, the method generalises beyond any single
taxonomy's design choices.

Data: ESCO v1.2.1, European Commission.
      https://esco.ec.europa.eu/en/use-esco/download

Run:
    python examples/esco.py

Requires ESCO CSV files. See download instructions below.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skillinfer


# ── Step 1: Locate ESCO data ─────────────────────────────────────────

# ESCO CSVs are not available via a single stable download URL.
# Download from: https://esco.ec.europa.eu/en/use-esco/download
# Select "ESCO dataset" → CSV format → English.
#
# Place these files in examples/.data/esco/:
#   - occupationSkillRelations_en.csv
#   - skillsHierarchy_en.csv
#   - skillGroups_en.csv
#   - broaderRelationsSkillPillar_en.csv

ESCO_DIR = Path(__file__).parent / ".data" / "esco"

# Also check the Tailor thesis repo as a fallback
FALLBACK_DIR = Path.home() / "repos" / "Tailor" / "papers" / "oring" / "onet" / "data" / "esco"

REQUIRED_FILES = [
    "occupationSkillRelations_en.csv",
    "skillsHierarchy_en.csv",
    "skillGroups_en.csv",
    "broaderRelationsSkillPillar_en.csv",
]


def find_esco_dir() -> Path:
    """Find ESCO data directory, checking multiple locations."""
    for d in [ESCO_DIR, FALLBACK_DIR]:
        if d.exists() and all((d / f).exists() for f in REQUIRED_FILES):
            return d

    print("ESCO data not found. Please download from:")
    print("  https://esco.ec.europa.eu/en/use-esco/download")
    print()
    print("Select 'ESCO dataset' → CSV → English, then place these files in:")
    print(f"  {ESCO_DIR}/")
    print()
    for f in REQUIRED_FILES:
        print(f"    {f}")
    sys.exit(1)


def build_esco_matrix() -> pd.DataFrame:
    """
    Build a binary occupation x skill-group matrix from ESCO.

    Processing:
      1. Load occupation-skill relations (essential only)
      2. Walk each skill up the hierarchy to its Level-2 skill group
      3. Binary encode: 1 if occupation has >= 1 essential skill in that group
      4. Filter to occupations with >= 5 skill groups
    """
    esco_dir = find_esco_dir()
    print(f"Loading ESCO data from {esco_dir}")

    rel = pd.read_csv(esco_dir / "occupationSkillRelations_en.csv")
    sg = pd.read_csv(esco_dir / "skillGroups_en.csv")
    br = pd.read_csv(esco_dir / "broaderRelationsSkillPillar_en.csv")
    sh = pd.read_csv(esco_dir / "skillsHierarchy_en.csv")

    # Map all hierarchy levels to their Level-2 ancestor
    group_uri_to_l2 = {}
    for _, row in sh.iterrows():
        l2_uri = row.get("Level 2 URI")
        if pd.isna(l2_uri):
            l2_uri = row.get("Level 1 URI")
        if pd.isna(l2_uri):
            continue
        for col in ["Level 0 URI", "Level 1 URI", "Level 2 URI", "Level 3 URI"]:
            uri = row.get(col)
            if pd.notna(uri):
                group_uri_to_l2[uri] = l2_uri

    # Walk each skill up to its nearest skill-group ancestor, then map to L2
    group_uris = set(sg["conceptUri"])
    broader_map = dict(zip(br["conceptUri"], br["broaderUri"]))

    skill_to_l2 = {}
    for skill_uri in rel["skillUri"].unique():
        current = skill_uri
        visited = set()
        while current and current not in group_uris and current not in visited:
            visited.add(current)
            current = broader_map.get(current)
        if current and current in group_uris:
            l2 = group_uri_to_l2.get(current, current)
            skill_to_l2[skill_uri] = l2

    # Build binary matrix from essential relations
    ess = rel[rel["relationType"] == "essential"].copy()
    ess["l2_group"] = ess["skillUri"].map(skill_to_l2)
    ess = ess.dropna(subset=["l2_group"])

    R = ess.groupby(["occupationUri", "l2_group"]).size().unstack(fill_value=0)
    R = (R > 0).astype(float)

    # Use readable group labels
    l2_labels = {}
    for _, row in sh.iterrows():
        for level in ["Level 2", "Level 1"]:
            uri = row.get(f"{level} URI")
            name = row.get(f"{level} preferred term")
            if pd.notna(uri) and pd.notna(name) and uri not in l2_labels:
                l2_labels[uri] = name
    R.columns = [l2_labels.get(c, c) for c in R.columns]

    # Filter: occupations with >= 5 skill groups
    R = R[R.sum(axis=1) >= 5]

    # Use occupation URIs as index (could resolve to titles if needed)
    return R


# ── Step 2: Build taxonomy ────────────────────────────────────────────

print("Building ESCO feature matrix...")
R = build_esco_matrix()
N, K = R.shape
density = R.values.mean()
print(f"  {N} occupations x {K} skill groups (binary, density={density:.2%})")

# Hold out a target occupation
rng = np.random.default_rng(42)
target_idx = rng.integers(N)
target_name = R.index[target_idx]
true_vec = R.iloc[target_idx].values.copy()

pop = skillinfer.Population.from_dataframe(R.drop(target_name), normalize=False)
print(f"  {pop}")
print(f"  Condition number: {pop.condition_number():.1f}")

# ── Step 3: Covariance structure ──────────────────────────────────────

print(f"\nTop skill-group correlations:")
for _, row in pop.top_correlations(k=8).iterrows():
    a = row['feature_a'][:40]
    b = row['feature_b'][:40]
    print(f"  {a:<40} <-> {b:<40}  r = {row['correlation']:+.3f}")

pca = pop.pca(n_components=5)
print(f"\nPCA: {pca['cumulative'][-1]:.1%} variance in 5 components")

# ── Step 4: Observe a few skill groups, predict the rest ──────────────

# Find skill groups this occupation actually has (value = 1)
active_groups = [R.columns[i] for i in range(K) if true_vec[i] > 0.5]
inactive_groups = [R.columns[i] for i in range(K) if true_vec[i] < 0.5]

print(f"\n{'=' * 65}")
print(f"  Target occupation: {target_name}")
print(f"  Has {len(active_groups)}/{K} skill groups")
print(f"{'=' * 65}")

# Observe 3 skill groups
state = pop.profile()
to_observe = active_groups[:3] if len(active_groups) >= 3 else active_groups
for feat in to_observe:
    state.observe(feat, 1.0)

print(f"\nObserved {len(to_observe)} skill groups (all = 1.0):")
for feat in to_observe:
    print(f"  {feat}")

# Show predictions
print(f"\nPredictions for unobserved skill groups (top 10 by predicted value):")
print(f"  {'Skill Group':<45} {'True':>5} {'Pred':>6} {'± Std':>6}")
print(f"  {'-' * 64}")

unobserved = [(f, i) for i, f in enumerate(R.columns) if f not in to_observe]
unobserved.sort(key=lambda x: state.mean(x[0]), reverse=True)

for feat, idx in unobserved[:10]:
    pred = state.mean(feat)
    std = state.std(feat)
    true = true_vec[idx]
    marker = " *" if true > 0.5 else ""
    print(f"  {feat:<45} {true:>5.0f} {pred:>6.3f} {std:>6.3f}{marker}")

print(f"\n  (* = occupation actually has this skill group)")

cosine = state.similarity(true_vec)
print(f"\n  Cosine similarity to true profile: {cosine:.4f}")

# ── Step 5: Held-out validation ───────────────────────────────────────

print(f"\nHeld-out validation (10 splits):")
results = skillinfer.validation.held_out_evaluation(
    pop, frac_observed=[0.1, 0.3, 0.5], n_splits=10, obs_noise=0.1,
)
summary = results.groupby(["frac_observed", "method"])[["cosine_similarity", "mse"]].mean()
print(summary.to_string())

# ── Attribution ───────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print("  ESCO v1.2.1, European Commission.")
print("  https://esco.ec.europa.eu/")
print(f"{'=' * 65}")
