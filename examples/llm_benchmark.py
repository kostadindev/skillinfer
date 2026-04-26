"""
End-to-end example: Predict LLM benchmark performance from one observation.

1. Download the Open LLM Leaderboard v2 results (4,500+ models, 6 benchmarks)
2. Build a Population (covariance structure across benchmarks)
3. A "new model" enters — observe it on 1 benchmark
4. Predict its performance on all other benchmarks
5. Compare predictions to ground truth
"""

import pandas as pd
import numpy as np
import skillinfer


# ── Step 1: Fetch benchmark data from HuggingFace ──────────────────────

print("Fetching Open LLM Leaderboard v2 data...")
url = (
    "https://huggingface.co/datasets/open-llm-leaderboard/contents"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
df_raw = pd.read_parquet(url)

bench_cols = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
df = df_raw[["fullname"] + bench_cols].dropna().set_index("fullname")
print(f"  {df.shape[0]} models x {df.shape[1]} benchmarks")

# ── Step 2: Hold out a model, build taxonomy from the rest ─────────────

# Find a Llama model to hold out
all_names = list(df.index)
candidates = [n for n in all_names if "Llama-3" in n and "70B" in n]
model_name = candidates[0] if candidates else all_names[100]

true_scores = df.loc[model_name].values.copy()
features = list(df.columns)

# Build taxonomy from everyone EXCEPT the target (no data leakage)
print(f"\nHolding out: {model_name}")
pop = skillinfer.Population.from_dataframe(df.drop(model_name), normalize=False)
print(f"  {pop}")
print(f"  Condition number: {pop.condition_number():.1f}")

# Covariance structure
top = pop.top_correlations(k=5)
print(f"\n  Strongest benchmark correlations:")
for _, row in top.iterrows():
    print(f"    {row['feature_a']:>12} <-> {row['feature_b']:<12}  r = {row['correlation']:.3f}")

pca = pop.pca(n_components=3)
print(f"\n  PCA: {pca['cumulative'][-1]:.1%} variance in 3 components")

print(f"\n{'='*65}")
print(f"  New model: {model_name}")
print(f"{'='*65}")

# Observe just ONE benchmark
observed_feature = "BBH"
observed_idx = features.index(observed_feature)
observed_value = true_scores[observed_idx]

print(f"\n  Observed: {observed_feature} = {observed_value:.2f}")
print(f"  Predicting {len(features) - 1} other benchmarks...\n")

state = pop.profile()
state.observe(observed_feature, observed_value)

# ── Step 4: Compare predictions to ground truth ────────────────────────

print(f"  {'Benchmark':<15} {'True':>8} {'Predicted':>10} {'±Std':>8} {'Error':>8}")
print(f"  {'-'*53}")

errors = []
for i, feat in enumerate(features):
    pred = state.mean(feat)
    std = state.std(feat)
    err = abs(true_scores[i] - pred)
    marker = "  <-- observed" if feat == observed_feature else ""
    print(f"  {feat:<15} {true_scores[i]:>8.2f} {pred:>10.2f} {std:>8.2f} {err:>8.2f}{marker}")
    if feat != observed_feature:
        errors.append(err)

cosine = state.similarity(true_scores)

print(f"\n  {'='*53}")
print(f"  From 1 observation ({observed_feature} = {observed_value:.2f}):")
print(f"    MAE on unobserved benchmarks: {np.mean(errors):.2f}")
print(f"    Cosine similarity to true:    {cosine:.4f}")

# ── Step 5: What to observe next? ──────────────────────────────────────

print(f"\n  Most uncertain (observe next):")
for _, row in state.most_uncertain(k=3).iterrows():
    print(f"    {row['feature']}: std = {row['std']:.2f}")

# ── Step 6: Observe a second benchmark and see improvement ─────────────

second_feature = state.most_uncertain(k=1).iloc[0]["feature"]
second_value = true_scores[features.index(second_feature)]
state.observe(second_feature, second_value)

errors2 = []
for i, feat in enumerate(features):
    if feat not in (observed_feature, second_feature):
        errors2.append(abs(true_scores[i] - state.mean(feat)))

cosine2 = state.similarity(true_scores)

print(f"\n  After observing {second_feature} = {second_value:.2f}:")
print(f"    MAE on remaining {len(errors2)} benchmarks: {np.mean(errors2):.2f}")
print(f"    Cosine similarity to true:       {cosine2:.4f}")
print(f"    Improvement: +{cosine2 - cosine:.4f} cosine")
