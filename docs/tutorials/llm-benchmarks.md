# Tutorial: LLM Benchmark Prediction

Predict a model's performance across all benchmarks from a single evaluation.

This tutorial uses the [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — 4,500+ models evaluated on 6 benchmarks — to show how `skillinfer` transfers information between correlated benchmarks.

!!! info "What you'll learn"
    - Building a population from real-world benchmark data
    - Predicting 5 benchmarks from 1 observation
    - Using `most_uncertain()` for active evaluation
    - Measuring prediction quality with MAE, RMSE, and cosine similarity

## Setup

```python
import pandas as pd
import numpy as np
import skillinfer
```

## Step 1: Fetch benchmark data

Download the Open LLM Leaderboard v2 results from HuggingFace:

```python
df = pd.read_parquet(
    "hf://datasets/open-llm-leaderboard/contents/data/train-00000-of-00001.parquet"
)
benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
df = df[["fullname"] + benchmarks].dropna().set_index("fullname")
print(f"{df.shape[0]} models x {df.shape[1]} benchmarks")
```

```text
4576 models x 6 benchmarks
```

## Step 2: Hold out a model and build the population

To evaluate prediction quality, we hold out one model and build the population from the rest:

```python
model_name = "meta-llama/Llama-3.1-70B-Instruct"
true_scores = df.loc[model_name].values.copy()

pop = skillinfer.Population.from_dataframe(df.drop(model_name), normalize=False)
print(pop)
```

### Explore the covariance structure

```python
print(pop.top_correlations(k=5))
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
pca = pop.pca(n_components=3)
print(f"3 components explain {pca['cumulative'][-1]:.1%} of variance")
```

## Step 3: Observe one benchmark, predict the rest

```python
profile = pop.profile()
profile.observe("BBH", true_scores[benchmarks.index("BBH")])

for feat in benchmarks:
    pred = profile.mean(feat)
    std = profile.std(feat)
    true = true_scores[benchmarks.index(feat)]
    err = abs(true - pred)
    marker = "  ← observed" if feat == "BBH" else ""
    print(f"  {feat:<15} true={true:6.1f}  pred={pred:6.1f} ± {std:.1f}  err={err:.1f}{marker}")

print(f"\n  MAE:  {profile.mae(true_scores):.1f}")
print(f"  RMSE: {profile.rmse(true_scores):.1f}")
print(f"  Cosine similarity: {profile.similarity(true_scores):.3f}")
```

```text
  IFEval          true=  86.7  pred= 69.3 ± 16.0  err=17.4
  BBH             true=  55.9  pred= 55.9 ±  0.6  err= 0.0  ← observed
  MATH Lvl 5      true=  38.1  pred= 35.1 ± 10.2  err= 3.0
  GPQA            true=  14.2  pred= 14.6 ±  2.8  err= 0.4
  MUSR            true=  17.7  pred= 17.8 ±  4.1  err= 0.1
  MMLU-PRO        true=  47.9  pred= 50.7 ±  4.2  err= 2.9

  MAE:  4.0
  RMSE: 7.3
  Cosine similarity: 0.993
```

From a single benchmark observation, most predictions are within a few points of the true value. The cosine similarity of 0.993 means the predicted profile almost perfectly preserves the shape of the true profile.

## Step 4: Active evaluation — what to observe next?

```python
print(profile.most_uncertain(k=3))
```

```text
     feature    mean    std
0     IFEval  69.3   16.0  ← highest uncertainty
1  MATH Lv 5  35.1   10.2
2       MUSR  17.8    4.1
```

IFEval has the highest uncertainty — it's the least predictable from BBH alone (and indeed had the largest error above). Evaluating it next would give the most information gain.

## Step 5: Observe a second benchmark and measure improvement

```python
second = profile.most_uncertain(k=1).iloc[0]["feature"]
profile.observe(second, true_scores[benchmarks.index(second)])

print(f"After 2 observations (BBH + {second}):")
for feat in benchmarks:
    pred = profile.mean(feat)
    true = true_scores[benchmarks.index(feat)]
    err = abs(true - pred)
    obs = "  ← observed" if feat in ("BBH", second) else ""
    print(f"  {feat:<15} true={true:6.1f}  pred={pred:6.1f}  err={err:.1f}{obs}")

print(f"\n  MAE:  {profile.mae(true_scores):.1f}")
print(f"  RMSE: {profile.rmse(true_scores):.1f}")
print(f"  Cosine similarity: {profile.similarity(true_scores):.3f}")
```

```text
After 2 observations (BBH + IFEval):
  IFEval          true=  86.7  pred= 86.7  err= 0.0  ← observed
  BBH             true=  55.9  pred= 55.9  err= 0.0  ← observed
  MATH Lvl 5      true=  38.1  pred= 37.8  err= 0.3
  GPQA            true=  14.2  pred= 14.0  err= 0.2
  MUSR            true=  17.7  pred= 16.9  err= 0.8
  MMLU-PRO        true=  47.9  pred= 51.7  err= 3.8

  MAE:  0.9
  RMSE: 1.6
  Cosine similarity: 1.000
```

| Metric | 1 observation | 2 observations | Improvement |
|--------|--------------|----------------|-------------|
| MAE | 4.0 | 0.9 | 4.4x lower |
| RMSE | 7.3 | 1.6 | 4.6x lower |
| Cosine similarity | 0.993 | 1.000 | → perfect |

With just 2 out of 6 benchmarks observed (33% eval budget), the predicted profile is almost indistinguishable from the true one.

## Full example

See [`examples/llm_benchmark.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/llm_benchmark.py) for the complete runnable script.

## Key takeaway

LLM benchmarks are highly correlated. `skillinfer` exploits this: **evaluating a model on 1-2 benchmarks is often enough to predict the rest**, saving significant compute and evaluation time. The `most_uncertain()` method tells you exactly which benchmark to run next for maximum information gain.
