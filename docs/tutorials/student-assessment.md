# Tutorial: Student Assessment

Predict a student's full grade profile from one exam score using synthetic data with known structure.

!!! info "What you'll learn"
    - Generating correlated data with known block structure
    - How covariance transfer works in a controlled setting
    - The effect of observing additional features
    - Validating that transfer beats the diagonal baseline

This tutorial uses synthetic data so you can see exactly how the covariance structure drives predictions without needing any external datasets.

## Setup

```python
import numpy as np
import pandas as pd
import skillinfer
```

## Step 1: Generate students with correlated scores

Create 200 students with a latent "general academic ability" factor that drives correlations:

```python
rng = np.random.default_rng(0)
N = 200
subjects = ["Math", "Physics", "Chemistry", "English", "History", "Art"]

# Latent factor: general academic ability
g = rng.normal(70, 12, N)

scores = pd.DataFrame({
    "Math":      g + rng.normal(0, 6, N),
    "Physics":   g * 0.9 + rng.normal(5, 7, N),
    "Chemistry": g * 0.85 + rng.normal(8, 8, N),
    "English":   -0.3 * g + rng.normal(90, 10, N),    # anti-correlated with STEM
    "History":   -0.2 * g + rng.normal(80, 9, N),      # weakly anti-correlated
    "Art":       rng.normal(65, 15, N),                 # nearly independent
}).clip(0, 100)

scores.index = [f"student_{i}" for i in range(N)]
```

The data has known structure:

- **STEM block**: Math, Physics, Chemistry are positively correlated (driven by `g`)
- **Humanities block**: English, History are anti-correlated with STEM
- **Art**: nearly independent of everything

## Step 2: Build the taxonomy

Hold out one student to test predictions:

```python
target = "student_42"
true_scores = scores.loc[target].values.copy()

tax = skillinfer.Taxonomy.from_dataframe(scores.drop(target), normalize=False)
print(tax)
```

```text
Taxonomy(199 entities x 6 features, shrinkage=0.0218)
```

Verify the covariance structure matches our design:

```python
for _, row in tax.top_correlations(k=5).iterrows():
    print(f"  {row['feature_a']:>10} ↔ {row['feature_b']:<10}  r = {row['correlation']:+.3f}")
```

```text
       Math ↔ Physics     r = +0.832
       Math ↔ Chemistry   r = +0.762
    Physics ↔ Chemistry   r = +0.724
       Math ↔ English     r = -0.514
    Physics ↔ English     r = -0.423
```

The learned correlations reflect the latent structure we designed.

## Step 3: Observe Math, predict everything

```python
state = tax.new_state(obs_noise=2.0)
state.observe("Math", true_scores[subjects.index("Math")])

for i, subj in enumerate(subjects):
    pred = state.mean(subj)
    std = state.std(subj)
    err = abs(true_scores[i] - pred)
    tag = "  ← observed" if subj == "Math" else ""
    print(f"  {subj:<12} true={true_scores[i]:6.1f}  pred={pred:6.1f} ± {std:.1f}  err={err:.1f}{tag}")
```

```text
  Math         true=  78.2  pred=  78.2 ± 2.0  err= 0.0  ← observed
  Physics      true=  73.5  pred=  72.8 ± 5.1  err= 0.7
  Chemistry    true=  69.3  pred=  68.1 ± 6.3  err= 1.2
  English      true=  61.4  pred=  63.9 ± 8.2  err= 2.5
  History      true=  67.2  pred=  70.1 ± 7.9  err= 2.9
  Art          true=  58.7  pred=  64.2 ± 14.1  err= 5.5
```

Notice:

- **Physics and Chemistry** are predicted accurately (strong positive correlation with Math)
- **English** prediction moved down from the population mean (anti-correlation with Math)
- **Art** barely moved and has high uncertainty (near-independent)

## Step 4: Observe next and improve

The most uncertain feature tells us where to look next:

```python
state.most_uncertain(k=3)
```

```text
     feature    mean   std
0        Art   64.2  14.1  ← most uncertain (independent)
1    English   63.9   8.2
2    History   70.1   7.9
```

Art has the highest uncertainty because it's nearly independent of Math — observing Math told us almost nothing about Art.

```python
# Observe Art (the most uncertain)
state.observe("Art", true_scores[subjects.index("Art")])
print(f"Cosine similarity: {state.similarity(true_scores):.4f}")
```

## Step 5: Validate that transfer helps

```python
results = skillinfer.validation.held_out_evaluation(
    tax, frac_observed=[0.2, 0.5], n_splits=5, obs_noise=2.0
)
summary = results.groupby(["frac_observed", "method"])["cosine_similarity"].mean()
print(summary)
```

```text
frac_observed  method
0.2            diagonal    0.966
               kalman      0.981  ← transfer helps
               prior       0.946
0.5            diagonal    0.987
               kalman      0.994
               prior       0.946
```

## Full example

See [`examples/quickstart.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/quickstart.py) for the complete runnable script.

## Key takeaway

Even with synthetic data, the pattern is clear: **correlated features transfer information**. Observing one STEM subject predicts the others well, while independent features (Art) require their own observation. The `most_uncertain()` method correctly identifies Art as the highest-value next observation.
