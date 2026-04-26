"""
O*NET: Infer a worker's full skill profile from a few task observations.

Uses the U.S. Department of Labor's O*NET 30.2 taxonomy (894 occupations x
120 features: 35 skills, 33 knowledge areas, 52 abilities). Each feature has
a continuous importance rating on a 1-5 scale, normalised to [0, 1].

The covariance structure reveals that cognitive/verbal skills form a dense
positive block, physical/manual skills form another, and the two blocks are
anti-correlated. This enables cross-skill transfer: observing high Writing
performance simultaneously updates Written Expression (r=0.95) and decreases
Static Strength (r=-0.55).

Data: O*NET 30.2, CC BY 4.0, U.S. Department of Labor / ETA.
      https://www.onetcenter.org/database.html

Run:
    python examples/onet.py
"""

import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import skillinfer


# ── Step 1: Download and parse O*NET data ─────────────────────────────

ONET_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text.zip"
CACHE_DIR = Path(__file__).parent / ".data"
ZIP_PATH = CACHE_DIR / "db_30_2_text.zip"
EXTRACT_DIR = CACHE_DIR / "db_30_2_text"


def download_onet():
    """Download and extract O*NET 30.2 database files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    needed = ["Skills.txt", "Knowledge.txt", "Abilities.txt", "Occupation Data.txt"]
    if EXTRACT_DIR.exists() and all((EXTRACT_DIR / f).exists() for f in needed):
        # Verify the file looks right (has expected header)
        header = (EXTRACT_DIR / "Skills.txt").read_text()[:50]
        if "Scale ID" in header:
            return

    if not ZIP_PATH.exists():
        print(f"Downloading O*NET 30.2 (~8 MB)...")
        urllib.request.urlretrieve(ONET_URL, ZIP_PATH)
        print(f"  Saved to {ZIP_PATH}")

    print("Extracting...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for f in needed:
            match = [m for m in zf.namelist() if m.endswith("/" + f)]
            if match:
                (EXTRACT_DIR / f).write_bytes(zf.read(match[0]))


def build_onet_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the 894 occupations x 120 features matrix from O*NET.

    Processing:
      1. Load Skills.txt, Knowledge.txt, Abilities.txt
      2. Filter to Importance scale (IM), 1-5 range
      3. Pivot to occupation x feature matrix
      4. Normalise each column to [0, 1]
    """
    download_onet()

    dfs = []
    for filename, category in [
        ("Skills.txt", "Skill"),
        ("Knowledge.txt", "Knowledge"),
        ("Abilities.txt", "Ability"),
    ]:
        df = pd.read_csv(EXTRACT_DIR / filename, sep="\t")
        df = df[df["Scale ID"] == "IM"]
        if "Recommend Suppress" in df.columns:
            df = df[df["Recommend Suppress"] != "Y"]
        pivot = df.pivot_table(
            index="O*NET-SOC Code",
            columns="Element Name",
            values="Data Value",
            aggfunc="mean",
        )
        pivot.columns = [f"{category}:{name}" for name in pivot.columns]
        dfs.append(pivot)

    R = dfs[0].join(dfs[1], how="outer").join(dfs[2], how="outer")
    R = R.dropna(thresh=int(R.shape[1] * 0.5))
    R = R.fillna(0)
    R = (R - R.min()) / (R.max() - R.min() + 1e-10)

    # Load occupation titles for display
    occ = pd.read_csv(EXTRACT_DIR / "Occupation Data.txt", sep="\t")
    occ = occ[["O*NET-SOC Code", "Title"]].drop_duplicates()
    titles = dict(zip(occ["O*NET-SOC Code"], occ["Title"]))

    R.index = [f"{titles.get(code, code)} ({code})" for code in R.index]

    return R, titles


# ── Step 2: Build taxonomy ────────────────────────────────────────────

print("Building O*NET feature matrix...")
R, titles = build_onet_matrix()
N, K = R.shape
n_skill = sum(1 for c in R.columns if c.startswith("Skill:"))
n_know = sum(1 for c in R.columns if c.startswith("Knowledge:"))
n_abil = sum(1 for c in R.columns if c.startswith("Ability:"))
print(f"  {N} occupations x {K} features ({n_skill} skills, {n_know} knowledge, {n_abil} abilities)")

# Hold out a target occupation before building taxonomy
target_name = [n for n in R.index if "Software Developer" in n][0]
true_vec = R.loc[target_name].values.copy()

tax = skillinfer.Taxonomy.from_dataframe(R.drop(target_name), normalize=False)
print(f"  {tax}")
print(f"  Condition number: {tax.condition_number():.1f}")

# ── Step 3: Covariance structure ──────────────────────────────────────

print(f"\nTop feature correlations:")
for _, row in tax.top_correlations(k=8).iterrows():
    print(f"  {row['feature_a']:>35} <-> {row['feature_b']:<35}  r = {row['correlation']:+.3f}")

pca = tax.pca(n_components=5)
print(f"\nPCA: {pca['cumulative'][-1]:.1%} variance in 5 components")
for i, v in enumerate(pca["explained_variance_ratio"]):
    print(f"  PC{i+1}: {v:.1%}")

# ── Step 4: Observe a few skills, predict the rest ────────────────────

print(f"\n{'=' * 65}")
print(f"  Target: {target_name}")
print(f"{'=' * 65}")

state = tax.new_state(obs_noise=0.05)

# Observe 3 skills (a software developer evaluated on these)
observed = {
    "Skill:Programming": true_vec[list(R.columns).index("Skill:Programming")],
    "Skill:Mathematics": true_vec[list(R.columns).index("Skill:Mathematics")],
    "Knowledge:Computers and Electronics": true_vec[list(R.columns).index("Knowledge:Computers and Electronics")],
}

for feat, val in observed.items():
    state.observe(feat, val)

print(f"\nObserved {len(observed)} features:")
for feat, val in observed.items():
    print(f"  {feat}: {val:.3f}")

# Show predictions for some interesting features
check_features = [
    "Skill:Complex Problem Solving",
    "Skill:Critical Thinking",
    "Knowledge:Mathematics",
    "Knowledge:Engineering and Technology",
    "Ability:Deductive Reasoning",
    "Ability:Mathematical Reasoning",
    "Ability:Written Comprehension",
    "Ability:Static Strength",
    "Ability:Manual Dexterity",
    "Ability:Stamina",
]
check_features = [f for f in check_features if f in R.columns]

print(f"\nPredictions for unobserved features:")
print(f"  {'Feature':<42} {'True':>6} {'Pred':>6} {'± Std':>6} {'Error':>6}")
print(f"  {'-' * 70}")

for feat in check_features:
    idx = list(R.columns).index(feat)
    pred = state.mean(feat)
    std = state.std(feat)
    true = true_vec[idx]
    err = abs(true - pred)
    print(f"  {feat:<42} {true:>6.3f} {pred:>6.3f} {std:>6.3f} {err:>6.3f}")

cosine = state.similarity(true_vec)
print(f"\n  Cosine similarity to true profile: {cosine:.4f}")

# ── Step 5: Most uncertain features ──────────────────────────────────

print(f"\nMost uncertain (observe next):")
for _, row in state.most_uncertain(k=5).iterrows():
    print(f"  {row['feature']}: std = {row['std']:.3f}")

# ── Step 6: Held-out validation ───────────────────────────────────────

print(f"\nHeld-out validation (10 splits):")
results = skillinfer.validation.held_out_evaluation(
    tax, frac_observed=[0.1, 0.3, 0.5], n_splits=10, obs_noise=0.05,
)
summary = results.groupby(["frac_observed", "method"])[["cosine_similarity", "mse"]].mean()
print(summary.to_string())

# ── Attribution ───────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print("  O*NET 30.2 Database, U.S. Department of Labor / ETA.")
print("  Licensed under CC BY 4.0.")
print(f"{'=' * 65}")
