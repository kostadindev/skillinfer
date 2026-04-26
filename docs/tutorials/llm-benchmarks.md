# Tutorial: LLM Benchmark Prediction

Predict a model's performance across all benchmarks from a single evaluation.

This tutorial uses the [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — 4,500+ models evaluated on 6 benchmarks — to show how `skillinfer` transfers information between correlated benchmarks.

!!! info "What you'll learn"
    - Building a taxonomy from real-world benchmark data
    - Predicting 5 benchmarks from 1 observation
    - Using `most_uncertain()` for active evaluation
    - Measuring prediction quality

## Setup

```python
import pandas as pd
import numpy as np
import skillinfer
```

## Step 1: Fetch benchmark data

Download the Open LLM Leaderboard v2 results from HuggingFace:

```python
url = (
    "https://huggingface.co/datasets/open-llm-leaderboard/contents"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
df_raw = pd.read_parquet(url)

benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
df = df_raw[["fullname"] + benchmarks].dropna().set_index("fullname")
print(f"{df.shape[0]} models x {df.shape[1]} benchmarks")
```

```text
4576 models x 6 benchmarks
```

## Step 2: Hold out a model and build the taxonomy

To evaluate prediction quality, we hold out one model and build the taxonomy from the rest:

```python
model_name = "meta-llama/Llama-3-70B"
true_scores = df.loc[model_name].values.copy()

tax = skillinfer.Taxonomy.from_dataframe(df.drop(model_name), normalize=False)
print(tax)
```

```text
Taxonomy(4575 entities x 6 features, shrinkage=0.0006)
```

### Explore the covariance structure

```python
print(tax.top_correlations(k=5))
```

```text
      feature_a     feature_b  correlation
0          GPQA      MMLU-PRO        0.879
1           BBH      MMLU-PRO        0.871
2          GPQA           BBH        0.863
3    MATH Lvl 5      MMLU-PRO        0.811
4    MATH Lvl 5          GPQA        0.788
```

All benchmarks are strongly positively correlated — knowing one tells you a lot about the others.

```python
pca = tax.pca(n_components=3)
print(f"3 components explain {pca['cumulative'][-1]:.1%} of variance")
```

```text
3 components explain 92.3% of variance
```

## Step 3: Observe one benchmark, predict the rest

```python
state = tax.new_state(obs_noise=1.0)
state.observe("BBH", true_scores[benchmarks.index("BBH")])
```

Now query predictions for all benchmarks:

```python
for feat in benchmarks:
    pred = state.mean(feat)
    std = state.std(feat)
    true = true_scores[benchmarks.index(feat)]
    err = abs(true - pred)
    marker = "  ← observed" if feat == "BBH" else ""
    print(f"  {feat:<15} true={true:6.2f}  pred={pred:6.2f} ± {std:.2f}  error={err:.2f}{marker}")
```

```text
  IFEval          true= 69.15  pred= 47.92 ± 9.72  error=21.23
  BBH             true= 63.79  pred= 63.79 ± 1.00  error= 0.00  ← observed
  MATH Lvl 5      true= 27.49  pred= 25.48 ± 5.02  error= 2.01
  GPQA            true= 14.09  pred= 16.32 ± 3.12  error= 2.23
  MUSR            true= 17.72  pred= 14.60 ± 4.83  error= 3.12
  MMLU-PRO        true= 42.68  pred= 37.05 ± 4.06  error= 5.63
```

From a single benchmark observation, most predictions are within a few points of the true value.

## Step 4: Active evaluation — what to observe next?

```python
state.most_uncertain(k=3)
```

```text
     feature    mean    std
0     IFEval  47.92  9.72  ← highest uncertainty
1       MUSR  14.60  4.83
2  MATH Lvl 5 25.48  5.02
```

IFEval has the highest uncertainty — it's the least predictable from BBH alone. Evaluating on IFEval next would give the most information gain.

## Step 5: Observe a second benchmark

```python
state.observe("IFEval", true_scores[benchmarks.index("IFEval")])

cosine = state.similarity(true_scores)
print(f"Cosine similarity to true profile: {cosine:.4f}")
```

With just 2 out of 6 benchmarks observed, the predicted profile is highly accurate.

## Full example

See [`examples/llm_benchmark.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/llm_benchmark.py) for the complete runnable script.

## Key takeaway

LLM benchmarks are highly correlated. `skillinfer` exploits this: **evaluating a model on 1-2 benchmarks is often enough to predict the rest**, saving significant compute and evaluation time. The `most_uncertain()` method tells you exactly which benchmark to run next for maximum information gain.
