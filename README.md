# skillinfer

**Infer a full skill profile from a few observations.** Few-shot capability estimation for AI agents and humans.

Evaluate an AI model on one benchmark and predict its performance on 37 others. Observe a worker on one task and infer their full capability profile across 120 skills. `skillinfer` learns how skills co-vary across a population and uses that structure to fill in the gaps — so you can understand what an agent or person can do without testing everything.

The algorithm is a multivariate Kalman filter over the population covariance — the exact Bayesian update for linear-Gaussian observations. Developed as part of a [Cambridge master's thesis on Human-AI Orchestration](https://github.com/kostadindev/skillinfer), where it powers few-shot capability estimation for both human workers (via O\*NET) and AI agents (via benchmark leaderboards).

```python
import skillinfer

tax = skillinfer.Taxonomy.from_dataframe(df)   # learn how skills co-vary
state = tax.new_state()                        # new agent, unknown profile
state.observe("BBH", 55.0)                     # evaluate one benchmark
print(state.mean("MMLU-PRO"))                  # predict another → 37.1
print(state.most_uncertain(k=3))               # what to evaluate next?
```

## Why?

Every agent — AI or human — has a **skill vector**: a profile of capabilities across many dimensions. An LLM's skill vector is its benchmark scores. A worker's skill vector is their competency ratings. You rarely get to observe the full vector, but skills are correlated — a model good at logical deduction is likely good at causal reasoning; a strong programmer likely has strong analytical reasoning.

`skillinfer` infers the full skill vector from a few observations. Each observation updates all dimensions simultaneously via the population covariance. The result is an **agent vector** — a probabilistic estimate of the full profile with uncertainty on every skill. Because agent vectors and **task vectors** (what a task requires) live in the same space, you can directly match agents to tasks.

No neural networks, no training loop, no GPU. Just matrix algebra that runs in milliseconds.

## Install

```bash
pip install skillinfer
```

For visualization:
```bash
pip install skillinfer[viz]
```

## Quick Start

### 1. Build a taxonomy from any entity-feature matrix

```python
import pandas as pd
import skillinfer

# Rows = entities (people, models, students, ...)
# Columns = features (skills, benchmarks, subjects, ...)
df = pd.read_csv("scores.csv", index_col=0)
tax = skillinfer.Taxonomy.from_dataframe(df)
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

### Choosing `obs_noise`

The `obs_noise` parameter controls how much the filter trusts each observation relative to the prior. It is the standard deviation of assumed observation noise.

- **Small values (0.01–0.1)**: observations are trusted heavily; a single observation will snap the posterior close to the observed value. Use when your data is normalised to [0, 1] and measurements are precise.
- **Large values (1.0–10.0)**: observations are treated as noisy; updates are gentler and the prior has more influence. Use with raw (unnormalised) data where feature scales are large, or when observations are inherently noisy (e.g., human ratings, single-run benchmark scores).

A good rule of thumb: set `obs_noise` to roughly the standard deviation of measurement error you'd expect for a single observation. When using `normalize=True` (the default), values around `0.05` work well. When using `normalize=False` with raw scores, `1.0` or higher is typical.

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
#      feature    mean   std
# 0   MMLU-PRO   29.93  4.21  ← never observed, highest uncertainty
# 1       MUSR   11.37  4.09  ← never observed
# 2       GPQA    8.13  2.78  ← never observed

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
import skillinfer

# Fetch 4,500+ models from the Open LLM Leaderboard
df = pd.read_parquet(
    "https://huggingface.co/datasets/open-llm-leaderboard/contents"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
df = df[["fullname"] + benchmarks].dropna().set_index("fullname")

# Build taxonomy — learns how benchmarks co-vary across 4,500+ models
tax = skillinfer.Taxonomy.from_dataframe(df, normalize=False)

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
results = skillinfer.validation.held_out_evaluation(
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
import skillinfer

# Feature correlation structure (hierarchically clustered)
skillinfer.analysis.correlation_heatmap(tax)

# PCA scree plot (how many dimensions matter?)
skillinfer.analysis.scree_plot(tax)

# Posterior profile with error bars
skillinfer.analysis.posterior_profile(state, reference=tax.population_mean)
```

## API Reference

### `Taxonomy`

The population model. Wraps an entity-feature matrix and its covariance structure.

| Method | Description |
|--------|-------------|
| `Taxonomy.from_dataframe(df, normalize=True, covariance="ledoit-wolf")` | Build from a DataFrame. `normalize=True` scales each column to [0, 1]. `covariance` can be `"ledoit-wolf"` (recommended) or `"sample"`. |
| `Taxonomy.from_csv(path, index_col=0, normalize=True, covariance="ledoit-wolf")` | Build from a CSV file. `index_col` specifies the entity name column. |
| `.entity(name)` | Get a named entity's feature vector (returns a copy). |
| `.new_state(prior_entity=None, prior_mean=None, obs_noise=0.05)` | Create an `InferenceState`. Uses the named entity, a custom vector, or the population mean as the prior. |
| `.pca(n_components=15)` | PCA decomposition. Returns dict with `components`, `explained_variance_ratio`, `cumulative`. |
| `.top_correlations(k=20)` | Top-k strongest feature-feature correlations (by absolute value). Returns DataFrame with columns `[feature_a, feature_b, correlation]`. |
| `.condition_number()` | Condition number of the covariance matrix (lower = more stable). |

**Attributes:** `matrix` (DataFrame), `feature_names`, `entity_names`, `covariance` (K×K ndarray), `correlation` (K×K ndarray), `population_mean` (K ndarray), `shrinkage` (float or None).

### `InferenceState`

The posterior belief about one entity. Created via `Taxonomy.new_state()`.

| Method | Description |
|--------|-------------|
| `.observe(feature, value)` | Kalman update on one feature. `feature` can be a name (str) or index (int). Returns self for chaining. |
| `.observe_many({feature: value, ...})` | Batch update — equivalent to calling `.observe()` sequentially for each entry. |
| `.mean(feature=None)` | Posterior mean. Returns scalar if `feature` given, full vector otherwise. |
| `.std(feature=None)` | Posterior standard deviation (sqrt of covariance diagonal). Returns scalar if `feature` given, full vector otherwise. |
| `.confidence_interval(feature, level=0.95)` | Gaussian confidence interval. Returns `(lower, upper)` tuple. |
| `.most_uncertain(k=10)` | Top-k features with highest posterior standard deviation. Returns DataFrame with columns `[feature, mean, std]`. |
| `.to_dataframe()` | Full posterior as DataFrame with columns `[feature, mean, std]`. |
| `.similarity(other_vector)` | Cosine similarity between the posterior mean and a target vector. |
| `.rmse(true_vector)` | Root mean squared error between posterior mean and a target vector. |
| `.mae(true_vector)` | Mean absolute error between posterior mean and a target vector. |
| `.uncertainty_ratio(Sigma_0)` | `tr(Sigma) / tr(Sigma_0)` — fraction of prior uncertainty remaining. 0.5 means uncertainty has halved. Pass `taxonomy.covariance` as `Sigma_0`. |
| `.copy()` | Deep copy of the state (independent mu and Sigma). |

**Attributes:** `mu` (K ndarray), `Sigma` (K×K ndarray), `obs_noise` (float), `feature_names`, `n_observations` (int).

### `analysis`

Visualization utilities (requires `pip install skillinfer[viz]`). All functions return a matplotlib `Figure` and accept an optional `ax` parameter to draw on an existing Axes.

| Function | Description |
|----------|-------------|
| `correlation_heatmap(taxonomy, cluster=True, figsize=(12,10))` | Clustered correlation matrix. Set `cluster=False` to preserve original feature order. |
| `scree_plot(taxonomy, max_components=30)` | PCA variance explained (bar + cumulative line). |
| `posterior_profile(state, reference=None, top_k=20)` | Horizontal bar chart of posterior mean with error bars. Pass `reference` (e.g., `tax.population_mean`) to overlay comparison points. |

### `validation`

| Function | Description |
|----------|-------------|
| `held_out_evaluation(taxonomy, frac_observed=0.3, n_splits=10, obs_noise=0.02, seed=42)` | Hold-out evaluation comparing `kalman`, `diagonal`, and `prior`. Returns DataFrame with columns: `cosine_similarity`, `rmse`, `mae`, `mse`, `r_squared`, `calibration_coverage` (Kalman only — fraction of true values inside 90% CI), `mean_log_likelihood` (Kalman only — proper scoring rule rewarding both accuracy and calibrated uncertainty). |
| `uncertainty_shrinkage(state_or_Sigma, Sigma_0)` | `tr(Sigma_k) / tr(Sigma_0)` — fraction of prior uncertainty remaining. Used in the thesis to trigger transitions between oversight levels. |
| `transfer_delta(results, metric="cosine_similarity")` | Computes Kalman advantage over diagonal baseline from `held_out_evaluation()` output. Returns DataFrame with `[frac_observed, kalman, diagonal, delta]`. |

## Assumptions and Limitations

- **Linear-Gaussian model.** The Kalman update is exact when observations are linear functions of the latent state with Gaussian noise. For binary or ordinal data, the update is approximate.
- **Stationary features.** The model assumes features don't change between observations (`c_{t+1} = c_t`, no process noise). This is appropriate when the observation window is short relative to the timescale of change. Over longer horizons (e.g., skill development over months), the posterior may become overconfident.
- **Point-estimate covariance.** The population covariance is estimated once via Ledoit-Wolf shrinkage and treated as known. A fully Bayesian treatment (inverse-Wishart prior on Sigma) would propagate uncertainty about the covariance itself, but is not implemented.
- **Requires correlated features.** If features are truly independent, the Kalman filter reduces to independent per-feature updates (the diagonal baseline). The method's value comes from off-diagonal covariance structure.

## Use Cases

- **AI agent capabilities**: Evaluate a model on 1-2 benchmarks, predict its performance on all others. Skip redundant evaluations, focus compute where it matters. ([example](examples/llm_benchmark.py))
- **Human skill profiling**: Observe a worker on a few tasks, infer their full capability profile across 120+ skills. A strong programmer likely has strong analytical reasoning and systems thinking. ([O\*NET example](examples/onet.py))
- **Worker-task matching**: Know a worker's strengths in a few areas, predict their fit for new roles and tasks before assigning them.
- **Model selection**: Partially evaluate candidate models, predict which one will perform best on your target task — without running every benchmark on every model.
- **Student assessment**: One exam score predicts performance across subjects. Test less, know more.

The only requirement: a population of agents or people described by skills that co-vary.

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
