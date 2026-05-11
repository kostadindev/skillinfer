<div align="center">

# skillinfer

**Observe a few skills. Predict the rest. With calibrated uncertainty.**

Few-shot capability estimation for AI agents and humans — a closed-form Bayesian update, no training loop, no GPU.

[![PyPI](https://img.shields.io/pypi/v/skillinfer.svg?color=blue)](https://pypi.org/project/skillinfer/)
[![Python](https://img.shields.io/pypi/pyversions/skillinfer.svg)](https://pypi.org/project/skillinfer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://kostadindev.github.io/skillinfer)
[![Downloads](https://static.pepy.tech/badge/skillinfer/month)](https://pepy.tech/project/skillinfer)

</div>

---

## What it does

`skillinfer` learns how capabilities co-vary across a population, then uses that structure to infer a **full** skill profile from **partial** observations — with confidence intervals you can trust.

- **One observation → predictions for all skills.** Observe Programming, get a posterior over 119 other skills via learned covariance.
- **Closed-form, exact, fast.** Standard Kalman update. **<1 ms** per observation. Scales to **1000+ skills**.
- **Calibrated uncertainty.** Every prediction comes with a credible interval — observed skills shrink to ~zero variance, anti-correlated skills move opposite.
- **No training loop.** No GPU. No iteration. One matrix-vector product.

## Install

```bash
pip install skillinfer
```

## 30-second demo

```python
import skillinfer

pop = skillinfer.datasets.onet()          # 894 occupations x 120 skills
profile = pop.profile()                   # new entity, unknown
profile.observe("Skill:Programming", 0.92)
print(profile.predict())                  # predict all 120 skills
```

```
                           feature   mean    std  ci_lower  ci_upper
    Skill:Complex Problem Solving   0.81   0.17      0.47      1.00
          Skill:Critical Thinking   0.73   0.15      0.43      1.00
              Skill:Programming     0.92   0.01      0.90      0.93  ← observed
                Skill:Mathematics   0.67   0.12      0.43      0.91
         Ability:Static Strength    0.10   0.23      0.00      0.55  ← anti-correlated
...
[120 rows x 5 columns]
```

## Why skillinfer?

| | skillinfer | Manual scoring | Train a model |
|---|---|---|---|
| Time to first prediction | seconds | minutes | hours–days |
| Needs training data | ❌ ships with O\*NET + ESCO | ❌ | ✅ thousands of examples |
| Uncertainty estimates | ✅ exact, closed-form | ❌ | ⚠️ if you build it |
| GPU required | ❌ | ❌ | ✅ usually |
| Cost per update | <1 ms, O(K²) | human-time | inference call |
| Interpretable | ✅ covariance is inspectable | ✅ | ❌ usually black-box |

## How it works

<div align="center">
<img src="https://raw.githubusercontent.com/kostadindev/skillinfer/main/docs/assets/ch4_algorithm_pipeline.svg" alt="skillinfer algorithm pipeline" width="820">
</div>

Observe one skill → the Kalman update propagates the evidence to every other skill through the learned covariance:

- **Positive covariance** → skills move together (high Programming → high Analytical Reasoning)
- **Negative covariance** → skills move opposite (high Programming → low Static Strength)
- **Independent skills** → unaffected

That's it. The update is the standard closed-form Gaussian conditioning rule. Predictions are clipped to `[0, 1]` to match the population's natural scale. Each `observe()` call is O(K²) — one matrix-vector product. No iteration, no convergence.

## Core API

```python
import skillinfer
from skillinfer import Skill, Task

# Build a population from any entity-feature matrix
pop = skillinfer.Population.from_dataframe(df)

# Create a profile and observe
profile = pop.profile()
profile.observe("math", 0.95)
profile.observe_many({"code": 0.89, "writing": 0.70})

# Predict with uncertainty
profile.predict()                          # all skills, with CIs
profile.predict("reasoning")               # single skill
profile.most_uncertain(k=3)                # what to assess next

# Match agents to tasks
task = Task({"math": 1.0, "reasoning": 0.5})
result = profile.match_score(task, threshold=0.8)

# Rank a pool of agents
ranking = skillinfer.rank_agents(task, profiles, threshold=0.8)

# Summary statistics
profile.summary(true_vector=ground_truth)  # MAE, RMSE, coverage, etc.
pop.summary()                              # condition number, sparsity, etc.
```

## Built-in datasets

Two preprocessed datasets ship with the package (~440 KB total):

```python
# O*NET 30.2 — U.S. Department of Labor
# 894 occupations x 120 features (skills, knowledge, abilities)
pop = skillinfer.datasets.onet()

# ESCO v1.2.1 — European Commission
# 2,999 occupations x 134 skill groups (binary)
pop = skillinfer.datasets.esco()
```

| Dataset | Entities | Features | Scale | Source |
|---------|----------|----------|-------|--------|
| **O\*NET** | 894 occupations | 120 (35 skills, 33 knowledge, 52 abilities) | Continuous [0, 1] | [O\*NET 30.2](https://www.onetcenter.org/database.html), CC BY 4.0 |
| **ESCO** | 2,999 occupations | 134 Level-2 skill groups | Binary {0, 1} | [ESCO v1.2.1](https://esco.ec.europa.eu/) |

## Use cases

| Domain | Observe | Predict |
|--------|---------|---------|
| **AI model selection** | 1–2 benchmark scores | All benchmarks + best model for a task |
| **Human skill profiling** | A few task observations | Full occupational profile (120 skills) |
| **Human–AI orchestration** | Partial evals for both | Who handles which subtask |
| **Worker–task matching** | Known competencies | Fit for new roles and tasks |
| **Active assessment** | Adaptive testing | Pick the next most informative question |

## LLM orchestration

`skillinfer` profiles are structured context you feed to an LLM orchestrator alongside cost, latency, and business constraints. The LLM reasons about observed vs. inferred skills and applies natural language constraints that no scoring function could replicate:

```python
from openai import OpenAI

# Build profiles from partial evaluations
agents = {
    "gpt-4o":     {"reasoning": 0.92, "code": 0.89},
    "claude-3.5": {"reasoning": 0.90, "writing": 0.95},
    "gemini-pro": {"math": 0.88, "code": 0.82},
}
profiles = {
    name: pop.profile().observe_many(obs)
    for name, obs in agents.items()
}

# Format as context for the orchestrator
agent_context = ""
for name, profile in profiles.items():
    agent_context += f"\n{name}:\n"
    for skill in ["math", "reasoning", "code"]:
        pred = profile.predict(skill)
        source = "observed" if pred["std"] < 0.01 else "inferred"
        agent_context += f"  {skill}: {pred['mean']:.2f} ± {pred['std']:.2f} ({source})\n"

# The LLM decides — not a scoring function
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Pick an agent for this math task.\n{agent_context}"}],
)
```

## Visualization

Requires `pip install skillinfer[viz]`.

```python
import skillinfer

pop = skillinfer.datasets.onet()
profile = pop.profile()
profile.observe("Skill:Programming", 0.92)

# Population charts
skillinfer.visualization.correlation_heatmap(pop)     # clustered correlation matrix
skillinfer.visualization.scree_plot(pop)               # PCA variance explained
skillinfer.visualization.feature_distributions(pop)    # box plots by variance
skillinfer.visualization.skill_embedding(pop)          # 2D PCA feature map
skillinfer.visualization.convergence_curve(pop)        # MAE vs. observations

# Profile charts
skillinfer.visualization.posterior_profile(profile)    # predicted skills + uncertainty
skillinfer.visualization.prediction_scatter(profile, true_vec)  # predicted vs. true
skillinfer.visualization.uncertainty_waterfall(pop, observations)  # uncertainty per observation
skillinfer.visualization.compare_profiles({"dev": dev, "nurse": nurse})  # side-by-side
```

## Export / import

```python
# Population
pop.to_csv("population.csv")
pop.to_parquet("population.parquet")
pop = skillinfer.Population.from_csv("population.csv")
pop = skillinfer.Population.from_parquet("population.parquet")

# Profile
profile.to_json("profile.json")
restored = skillinfer.Profile.from_json("profile.json")

d = profile.to_dict()   # plain dict, JSON-serialisable
restored = skillinfer.Profile.from_dict(d)
```

## Documentation

Full documentation at **[kostadindev.github.io/skillinfer](https://kostadindev.github.io/skillinfer)**:

- 📘 [Quickstart](https://kostadindev.github.io/skillinfer/getting-started/quickstart/)
- 🎓 [Tutorials](https://kostadindev.github.io/skillinfer/tutorials/llm-benchmarks/) — LLM benchmarks, human skills, ESCO, agent orchestration
- 🧠 [How it works](https://kostadindev.github.io/skillinfer/how-it-works/kalman-update/) — Kalman update, covariance estimation, computational cost
- 📚 [API reference](https://kostadindev.github.io/skillinfer/api/population/) — Population, Profile, Datasets, Visualization

## Contributing

Issues, discussions, and PRs welcome. If `skillinfer` helps your work, **starring the repo** ⭐ is the simplest way to support it — and it helps others find the project.

## License

MIT
