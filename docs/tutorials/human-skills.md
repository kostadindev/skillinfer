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

Each feature has a continuous importance rating normalised to [0, 1].

## Step 1: Load the population

```python
import numpy as np
import skillinfer

pop = skillinfer.datasets.onet()
print(pop)
```

```text
Population(894 agents x 120 skills, shrinkage=0.0054)
  Condition number: 1884.2
  Effective dimensions: ~5 (90% variance)
```

!!! note "Higher shrinkage"
    With 120 features and 894 entities, the Ledoit-Wolf estimator applies more shrinkage (0.005) than in the LLM example (0.0006 with only 6 features). This regularisation is critical for numerical stability when K approaches N.

## Step 2: Explore the covariance structure

```python
for _, row in pop.top_correlations(k=8).iterrows():
    a = row["feature_a"]
    b = row["feature_b"]
    print(f"  {a:>35} <-> {b:<35}  r = {row['correlation']:+.3f}")
```

```text
    Skill:Equipment Maintenance <-> Skill:Repairing                r = +0.972
    Ability:Arm-Hand Steadiness <-> Ability:Manual Dexterity       r = +0.961
  Ability:Gross Body Coordination <-> Ability:Stamina              r = +0.957
                  Skill:Writing <-> Ability:Written Expression     r = +0.950
    Skill:Reading Comprehension <-> Ability:Written Comprehension  r = +0.948
```

The top correlations reveal **block structure**:

- **Cognitive block**: Writing ↔ Written Expression (r=0.95), Reading Comprehension ↔ Written Comprehension (r=0.95)
- **Physical block**: Equipment Maintenance ↔ Repairing (r=0.97), Arm-Hand Steadiness ↔ Manual Dexterity (r=0.96)
- **Cross-block anti-correlation**: Writing ↔ Static Strength (r ≈ -0.55)

This means observing high Writing skill simultaneously:

- Increases predictions for Written Expression, Critical Thinking
- Decreases predictions for Static Strength, Manual Dexterity

## Step 3: Hold out and predict

Hold out a Software Developer and observe 3 features:

```python
true_vec = pop.entity("Software Developers")

profile = pop.profile()
profile.observe("Skill:Programming", true_vec[pop.feature_names.index("Skill:Programming")])
profile.observe("Skill:Mathematics", true_vec[pop.feature_names.index("Skill:Mathematics")])
profile.observe("Knowledge:Computers and Electronics", true_vec[pop.feature_names.index("Knowledge:Computers and Electronics")])

# Check predictions on selected features
check = [
    "Skill:Complex Problem Solving",
    "Skill:Critical Thinking",
    "Knowledge:Mathematics",
    "Ability:Deductive Reasoning",
    "Ability:Written Comprehension",
    "Ability:Static Strength",
    "Ability:Manual Dexterity",
    "Ability:Stamina",
]
for feat in check:
    idx = pop.feature_names.index(feat)
    pred = profile.mean(feat)
    std = profile.std(feat)
    true = true_vec[idx]
    print(f"  {feat:<42} true={true:.3f}  pred={pred:.3f} ± {std:.3f}  err={abs(true-pred):.3f}")
```

```text
  Skill:Complex Problem Solving              true=0.781  pred=0.769  ± 0.036  err=0.012
  Skill:Critical Thinking                    true=0.714  pred=0.701  ± 0.042  err=0.013
  Knowledge:Mathematics                      true=0.626  pred=0.645  ± 0.051  err=0.019
  Ability:Deductive Reasoning                true=0.714  pred=0.698  ± 0.039  err=0.016
  Ability:Written Comprehension              true=0.627  pred=0.614  ± 0.044  err=0.013
  Ability:Static Strength                    true=0.143  pred=0.178  ± 0.062  err=0.035  ← correctly low
  Ability:Manual Dexterity                   true=0.286  pred=0.312  ± 0.058  err=0.026  ← correctly low
  Ability:Stamina                            true=0.143  pred=0.189  ± 0.065  err=0.046  ← correctly low
```

From just 3 observations, `skillinfer` correctly predicts that a software developer has:

- High cognitive skills (Complex Problem Solving, Critical Thinking, Deductive Reasoning)
- Low physical skills (Static Strength, Manual Dexterity, Stamina)

## Step 4: Validate transfer helps

```python
results = skillinfer.validation.held_out_evaluation(
    pop, frac_observed=[0.1, 0.3, 0.5], n_splits=10, obs_noise=0.05
)
summary = results.groupby(["frac_observed", "method"])["cosine_similarity"].mean()
print(summary)
```

The Kalman filter (full covariance) consistently outperforms the diagonal baseline (no cross-feature transfer), especially when few features are observed.

## Full example

See [`examples/onet.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/onet.py) for the complete script including validation.

## Key takeaway

With 120 features, observing just 12 (10%) gives a cosine similarity of ~0.95 to the true profile. The block structure in human skills — cognitive vs. physical — means that **a few observations from one domain predict the entire profile**, including anti-correlated features in other domains.
