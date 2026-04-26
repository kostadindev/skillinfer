# bayeskal

**Predict all features from a few observations.** Bayesian inference with learned covariance structure.

Given a population of entities described by K features, `bayeskal` learns how features co-vary and uses that structure to infer unobserved features from partial observations. Observe one feature, update beliefs about all others.

```python
import bayeskal

tax = bayeskal.Taxonomy.from_dataframe(df)   # learn covariance structure
state = tax.new_state()                       # new entity, unknown profile
state.observe("BBH", 32.7)                    # observe one benchmark
print(state.mean("MMLU-PRO"))                 # predict another → 29.9
print(state.most_uncertain(k=3))              # what to observe next?
```

## Why?

Features are rarely independent. Math scores predict physics scores. Writing ability correlates with reading comprehension. A model good at logical deduction is likely good at causal reasoning. `bayeskal` exploits this structure: **one observation updates all K features simultaneously** via the population covariance.

The core is a multivariate Kalman filter — the exact Bayesian update for linear-Gaussian observations. No neural networks, no training loop, no GPU. Just matrix algebra that runs in milliseconds.

## Install

```bash
pip install bayeskal
```

For visualization:
```bash
pip install bayeskal[viz]
```

## Quick Start

### 1. Build a taxonomy from any entity-feature matrix

```python
import pandas as pd
import bayeskal

# Rows = entities (people, models, students, ...)
# Columns = features (skills, benchmarks, subjects, ...)
df = pd.read_csv("scores.csv", index_col=0)
tax = bayeskal.Taxonomy.from_dataframe(df)
# Taxonomy(4576 entities x 6 features, shrinkage=0.0006)
```

### 2. Observe and infer

```python
# A new entity arrives — start from the population mean
state = tax.new_state(obs_noise=1.0)

# Observe one feature
state.observe("BBH", 32.7)

# ALL features are now updated via covariance transfer
print(state.mean("MMLU-PRO"))   # 29.9 (predicted, never observed)
print(state.std("MMLU-PRO"))    # 4.2  (uncertainty)

# Chain observations
state.observe("IFEval", 47.1).observe("MATH Lvl 5", 18.0)
```

### 3. Query the posterior

```python
# Full profile
state.to_dataframe()
#        feature   mean    std
# 0       IFEval  47.10   1.00
# 1          BBH  32.70   1.00
# 2   MATH Lvl 5  18.00   1.00
# 3         GPQA   8.13   2.78  ← inferred
# 4         MUSR  11.37   4.09  ← inferred
# 5     MMLU-PRO  29.93   4.21  ← inferred

# Similarity to a known entity
state.similarity(tax.entity("meta-llama/Llama-3-70B"))  # 0.987

# What should we evaluate next?
state.most_uncertain(k=3)
#      feature    std
# 0     IFEval  16.0   ← highest uncertainty
# 1  MATH Lvl 5 10.2
# 2   MMLU-PRO   4.2

# Confidence intervals
state.confidence_interval("GPQA", level=0.95)  # (2.7, 13.6)
```

### 4. Use a specific entity as prior

```python
# "This new model is based on Llama-3-70B"
state = tax.new_state(prior_entity="meta-llama/Llama-3-70B", obs_noise=1.0)
state.observe("MATH Lvl 5", 25.0)  # fine-tuned for math
# Now the posterior reflects Llama-3's profile updated with the math observation
```

## End-to-End Example: LLM Benchmarks

Predict a model's performance across all benchmarks from a single evaluation:

```python
import pandas as pd
import bayeskal

# Fetch 4,500+ models from the Open LLM Leaderboard
df = pd.read_parquet(
    "https://huggingface.co/datasets/open-llm-leaderboard/contents"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
df = df[["fullname"] + benchmarks].dropna().set_index("fullname")

# Build taxonomy — learns how benchmarks co-vary across 4,500+ models
tax = bayeskal.Taxonomy.from_dataframe(df, normalize=False)

# A new model arrives — we only know its BBH score
state = tax.new_state(obs_noise=1.0)
state.observe("BBH", 55.0)

# Predict all other benchmarks
for feat in benchmarks:
    print(f"  {feat:>12}: {state.mean(feat):6.1f} ± {state.std(feat):.1f}")
```

See [`examples/llm_benchmark.py`](examples/llm_benchmark.py) for the full working example.

## How It Works

### The Kalman Update

When you observe feature *j* with value *y*:

```
innovation = y - mu[j]
K_gain     = Sigma[:, j] / (Sigma[j,j] + noise²)
mu        += K_gain * innovation
Sigma     -= outer(K_gain, Sigma[j,:])
```

- `K_gain` is a K-dimensional vector — the **Kalman gain**
- Feature *i* gets updated proportionally to `Sigma[i,j]`: how much it co-varies with the observed feature
- Positively correlated features move in the same direction; anti-correlated features move opposite
- The covariance shrinks with each observation (uncertainty decreases)

### Covariance Estimation

The covariance matrix is estimated from the population via [Ledoit-Wolf shrinkage](https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance), which regularizes the sample covariance for numerical stability. This is critical when the number of features approaches the number of entities.

### Computational Cost

- **Building the taxonomy**: O(N * K²) — one-time cost, typically < 1 second
- **Each observation**: O(K²) — one matrix-vector product + rank-1 update
- **Memory**: O(K²) for the covariance matrix

For K=120 features, each observation takes ~0.1ms. The method scales comfortably to K=1000+.

## Analyze Your Taxonomy

```python
# How correlated are the features?
tax.top_correlations(k=10)

# How many independent dimensions?
pca = tax.pca(n_components=10)
print(pca["cumulative"])  # [0.71, 0.82, 0.89, ...]

# Condition number (lower = more stable)
tax.condition_number()
```

## Validate That Transfer Helps

Does the covariance structure actually improve predictions? Run a held-out evaluation:

```python
results = bayeskal.validation.held_out_evaluation(
    tax,
    frac_observed=[0.1, 0.2, 0.3, 0.5],
    n_splits=10,
    obs_noise=0.02,
)
print(results.groupby(["frac_observed", "method"])["cosine_similarity"].mean())
# frac_observed  method
# 0.1            kalman    0.947  ← with transfer
#                diagonal  0.921  ← without transfer
#                prior     0.889  ← no observations
```

The `kalman` method (full covariance) consistently outperforms `diagonal` (no transfer), especially when few features are observed.

## Visualize

```python
import bayeskal

# Feature correlation structure (hierarchically clustered)
bayeskal.analysis.correlation_heatmap(tax)

# PCA scree plot (how many dimensions matter?)
bayeskal.analysis.scree_plot(tax)

# Posterior profile with error bars
bayeskal.analysis.posterior_profile(state, reference=tax.population_mean)
```

## API Reference

### `Taxonomy`

The population model. Wraps an entity-feature matrix and its covariance structure.

| Method | Description |
|--------|-------------|
| `Taxonomy.from_dataframe(df, normalize=True, covariance="ledoit-wolf")` | Build from a DataFrame |
| `Taxonomy.from_csv(path, ...)` | Build from a CSV file |
| `.entity(name)` | Get a named entity's feature vector |
| `.new_state(prior_entity=None, prior_mean=None, obs_noise=0.05)` | Create an InferenceState |
| `.pca(n_components=15)` | PCA decomposition |
| `.top_correlations(k=20)` | Strongest feature-feature correlations |
| `.condition_number()` | Covariance matrix condition number |

### `InferenceState`

The posterior belief about one entity. Created via `Taxonomy.new_state()`.

| Method | Description |
|--------|-------------|
| `.observe(feature, value)` | Kalman update on one feature (returns self) |
| `.observe_many({feature: value, ...})` | Batch update |
| `.mean(feature=None)` | Posterior mean (scalar or full vector) |
| `.std(feature=None)` | Posterior standard deviation |
| `.confidence_interval(feature, level=0.95)` | Gaussian CI |
| `.most_uncertain(k=10)` | Highest-variance features |
| `.to_dataframe()` | Full posterior as DataFrame |
| `.similarity(other_vector)` | Cosine similarity to a target |
| `.copy()` | Deep copy |

### `analysis`

Visualization utilities (requires `pip install bayeskal[viz]`).

| Function | Description |
|----------|-------------|
| `correlation_heatmap(taxonomy)` | Clustered correlation matrix |
| `scree_plot(taxonomy)` | PCA variance explained |
| `posterior_profile(state, reference=None)` | Bar chart with error bars |

### `validation`

| Function | Description |
|----------|-------------|
| `held_out_evaluation(taxonomy, frac_observed, n_splits, obs_noise)` | Compare transfer vs no-transfer |

## Use Cases

- **AI model capabilities**: Predict benchmark performance from partial evaluations. Observe a model on 2 benchmarks, predict the other 37. ([example](examples/llm_benchmark.py))
- **Human skills**: Infer a worker's full skill profile from a few task observations. A strong writer likely has strong reading comprehension. (O\*NET: 894 occupations x 120 skills)
- **Student assessment**: Estimate mastery across subjects from a few test scores. A student good at algebra is likely good at geometry.
- **Recommender systems**: Predict user preferences from partial ratings. Users who like sci-fi tend to like fantasy.
- **Sensor networks**: Infer missing sensor readings from nearby observations. Temperature and humidity co-vary.

The only requirement: a matrix of entities x features where features co-vary across the population.

## Citation

If you use this package in research, please cite:

```bibtex
@mastersthesis{devedzhiev2026orchestration,
  author = {Kostadin Devedzhiev},
  title  = {Human-AI Orchestration},
  school = {University of Cambridge},
  year   = {2026},
}
```

## License

MIT
