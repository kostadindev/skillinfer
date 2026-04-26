"""Task-agent matching: rank models by expected task performance.

Demonstrates how skillinfer can rank a pool of partially-observed LLMs
by expected performance on a specific task, using only a fraction of
the full evaluation budget.

Usage:
    python examples/task_matching.py
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import skillinfer


def main():
    # --- 1. Fetch LLM benchmark data ---
    print("Fetching Open LLM Leaderboard data...")
    df = pd.read_parquet(
        "https://huggingface.co/datasets/open-llm-leaderboard/contents"
        "/resolve/main/data/train-00000-of-00001.parquet"
    )
    benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    df = df[["fullname"] + benchmarks].dropna().set_index("fullname")

    # Take the top 50 models by average score for a cleaner demo
    df["avg"] = df.mean(axis=1)
    df = df.nlargest(50, "avg").drop(columns="avg")
    print(f"Using {len(df)} models x {len(benchmarks)} benchmarks\n")

    # --- 2. Build taxonomy ---
    pop = skillinfer.Population.from_dataframe(df, normalize=False)

    # --- 3. Define a math-heavy task ---
    task = {"MATH Lvl 5": 1.0, "GPQA": 0.5}
    print(f"Task vector: {task}")
    print("(Higher weight = more important for this task)\n")

    # --- 4. Simulate partial observation ---
    # For each model, observe only 1 random benchmark (out of 6)
    rng = np.random.default_rng(42)
    states = {}
    for model in df.index:
        state = pop.profile()
        obs_idx = rng.integers(0, len(benchmarks))
        obs_bench = benchmarks[obs_idx]
        state.observe(obs_bench, df.loc[model, obs_bench])
        states[model] = state

    budget_used = len(df)  # 50 observations
    budget_full = len(df) * len(benchmarks)  # 300 observations
    budget_pct = budget_used / budget_full * 100

    # --- 5. Rank by expected task performance ---
    ranking = skillinfer.rank_agents(task, states, threshold=50.0)

    # --- 6. Compare to ground truth ---
    # True ranking: computed from full benchmark data
    true_scores = {}
    w = np.zeros(len(benchmarks))
    for feat, weight in task.items():
        w[benchmarks.index(feat)] = weight
    for model in df.index:
        true_scores[model] = float(w @ df.loc[model].values)

    true_ranking = sorted(true_scores.keys(), key=lambda m: true_scores[m], reverse=True)
    pred_ranking = list(ranking["agent"])

    # Rank correlation
    true_order = {m: i for i, m in enumerate(true_ranking)}
    pred_ranks = [true_order[m] for m in pred_ranking]
    true_ranks = list(range(len(pred_ranking)))
    tau, _ = kendalltau(true_ranks, pred_ranks)

    # Top-k accuracy
    true_top3 = set(true_ranking[:3])
    pred_top3 = set(pred_ranking[:3])
    top3_overlap = len(true_top3 & pred_top3)

    true_top5 = set(true_ranking[:5])
    pred_top5 = set(pred_ranking[:5])
    top5_overlap = len(true_top5 & pred_top5)

    # --- 7. Print results ---
    print("=" * 70)
    print("RESULTS: Task-Agent Matching from Partial Observations")
    print("=" * 70)
    print(f"  Budget used:      {budget_used}/{budget_full} evaluations "
          f"({budget_pct:.0f}%)")
    print(f"  Rank correlation: τ = {tau:.3f} (Kendall's tau)")
    print(f"  Top-3 accuracy:   {top3_overlap}/3 correct")
    print(f"  Top-5 accuracy:   {top5_overlap}/5 correct")
    print()

    print("Top 10 predicted ranking:")
    print(f"  {'Rank':<5} {'Model':<50} {'Score':>7} {'± Std':>7} {'P>50':>6}")
    print("  " + "-" * 75)
    for i, row in ranking.head(10).iterrows():
        name = row["agent"]
        if len(name) > 48:
            name = name[:45] + "..."
        marker = " ✓" if name in true_top3 or row["agent"] in true_top3 else ""
        print(f"  {i+1:<5} {name:<50} {row['expected_score']:>7.1f} "
              f"{row['std']:>7.1f} {row['p_above_threshold']:>5.1%}{marker}")

    print()
    print(f"True top 3:      {', '.join(m[:40] for m in true_ranking[:3])}")
    print(f"Predicted top 3: {', '.join(m[:40] for m in pred_ranking[:3])}")


if __name__ == "__main__":
    main()
