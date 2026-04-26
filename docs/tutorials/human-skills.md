# Tutorial: Human Skills (O*NET)

Infer a worker's full skill profile from a few task observations using the U.S. Department of Labor's O\*NET taxonomy.

!!! info "What you'll learn"
    - Working with large feature spaces (894 occupations x 120 features)
    - Block structure in covariance (cognitive vs. physical skills)
    - Cross-domain transfer: observing one skill type predicts others
    - Anti-correlation: cognitive skills *negatively* predict physical skills

## About O\*NET

[O\*NET 30.2](https://www.onetcenter.org/database.html) describes 894 occupations across 120 features:

- **35 skills** (Programming, Writing, Mathematics, ...)
- **33 knowledge areas** (Computers & Electronics, Engineering, ...)
- **52 abilities** (Deductive Reasoning, Static Strength, ...)

Each feature has a continuous importance rating on a 1–5 scale, normalized to [0, 1].

## Setup

```python
import numpy as np
import pandas as pd
import skillinfer
```

The example script handles downloading and parsing O\*NET data automatically. See [`examples/onet.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/onet.py) for the full data pipeline.

## Step 1: Build the taxonomy

After parsing, you get a matrix of 894 occupations x 120 features:

```python
tax = skillinfer.Taxonomy.from_dataframe(R, normalize=False)
print(tax)
```

```text
Taxonomy(893 entities x 120 features, shrinkage=0.0241)
```

!!! note "Higher shrinkage"
    With 120 features and 894 entities, the Ledoit-Wolf estimator applies more shrinkage (0.024) than in the LLM example (0.0006 with only 6 features). This regularization is critical for numerical stability when K approaches N.

## Step 2: Explore the covariance structure

```python
tax.top_correlations(k=8)
```

The top correlations reveal **block structure**:

- **Cognitive block**: Writing ↔ Written Expression (r=0.95), Reading Comprehension ↔ Written Comprehension (r=0.94)
- **Physical block**: Static Strength ↔ Trunk Strength (r=0.97), Stamina ↔ Dynamic Strength (r=0.95)
- **Cross-block anti-correlation**: Writing ↔ Static Strength (r=-0.55)

This means observing high Writing skill simultaneously:

- Increases predictions for Written Expression, Critical Thinking
- Decreases predictions for Static Strength, Manual Dexterity

## Step 3: Observe and predict

Hold out a Software Developer and observe 3 features:

```python
state = tax.new_state(obs_noise=0.05)
state.observe("Skill:Programming", 0.87)
state.observe("Skill:Mathematics", 0.72)
state.observe("Knowledge:Computers and Electronics", 0.91)
```

Predictions for unobserved features:

```text
  Feature                                     True   Pred  ± Std  Error
  Skill:Complex Problem Solving              0.781  0.769  0.036  0.012
  Skill:Critical Thinking                    0.714  0.701  0.042  0.013
  Knowledge:Mathematics                      0.626  0.645  0.051  0.019
  Knowledge:Engineering and Technology       0.715  0.688  0.055  0.027
  Ability:Deductive Reasoning                0.714  0.698  0.039  0.016
  Ability:Mathematical Reasoning             0.680  0.663  0.048  0.017
  Ability:Written Comprehension              0.627  0.614  0.044  0.013
  Ability:Static Strength                    0.143  0.178  0.062  0.035  ← correctly low
  Ability:Manual Dexterity                   0.286  0.312  0.058  0.026  ← correctly low
  Ability:Stamina                            0.143  0.189  0.065  0.046  ← correctly low
```

From just 3 observations, `skillinfer` correctly predicts that a software developer has:

- High cognitive skills (Complex Problem Solving, Critical Thinking, Deductive Reasoning)
- Low physical skills (Static Strength, Manual Dexterity, Stamina)

## Step 4: Validate transfer helps

```python
results = skillinfer.validation.held_out_evaluation(
    tax, frac_observed=[0.1, 0.3, 0.5], n_splits=10, obs_noise=0.05
)
summary = results.groupby(["frac_observed", "method"])["cosine_similarity"].mean()
print(summary)
```

```text
frac_observed  method
0.1            diagonal    0.921
               kalman      0.953  ← +3.2% with transfer
               prior       0.889
0.3            diagonal    0.964
               kalman      0.981  ← +1.7% with transfer
               prior       0.889
0.5            diagonal    0.983
               kalman      0.992
               prior       0.889
```

The Kalman filter (full covariance) consistently outperforms the diagonal baseline (no cross-feature transfer), especially when few features are observed.

## Full example

See [`examples/onet.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/onet.py) for the complete script including data download, parsing, and validation.

## Key takeaway

With 120 features, observing just 12 (10%) gives a cosine similarity of 0.953 to the true profile. The block structure in human skills — cognitive vs. physical — means that **a few observations from one domain predict the entire profile**, including anti-correlated features in other domains.
