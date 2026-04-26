# skillinfer

**Infer a full skill profile from a few observations.**

One benchmark predicts 37 others. One task observation reveals 120 skills. `skillinfer` learns how capabilities co-vary across a population and fills in the gaps — so you can understand what any agent, person, or team can do without testing everything.

No neural networks, no training loop, no GPU. A Kalman filter over population covariance: the exact Bayesian posterior in one matrix operation. Under 1ms per update, scales to 1000+ skills.

## Install

```bash
pip install skillinfer
```

## Quick start

```python
import skillinfer

pop = skillinfer.datasets.onet()          # 894 occupations x 120 skills
profile = pop.profile()                   # new entity, unknown
profile.observe("Skill:Programming", 0.92)
print(profile.predict())                  # predict all 120 skills
```

```
                           feature   mean    std  ci_lower  ci_upper
    Skill:Complex Problem Solving   1.12   0.17      0.78      1.45
          Skill:Critical Thinking   0.96   0.15      0.66      1.25
              Skill:Programming     0.92   0.01      0.90      0.93  ← observed
                Skill:Mathematics   0.85   0.12      0.61      1.09
         Ability:Static Strength  -0.37   0.23     -0.82      0.07  ← anti-correlated
...
[120 rows x 5 columns]
```

## How it works

When you observe one skill, the Kalman update propagates to every other skill via the learned covariance:

- Skills with **positive covariance** move in the same direction (observe high Programming → predict high Analytical Reasoning)
- Skills with **negative covariance** move opposite (observe high Programming → predict low Static Strength)
- **Independent skills** are unaffected

Each `observe()` call is O(K²) — one matrix-vector product. No iteration, no convergence, no approximation.

## Core API

```python
import skillinfer
from skillinfer import Task

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
| **AI model selection** | 1-2 benchmark scores | All benchmarks + best model for a task |
| **Human skill profiling** | A few task observations | Full occupational profile (120 skills) |
| **Human-AI orchestration** | Partial evals for both | Who handles which subtask |
| **Worker-task matching** | Known competencies | Fit for new roles and tasks |

## LLM orchestration

Skill profiles are structured context you feed to an LLM orchestrator — not a heuristic for matching. The LLM reasons about observed vs. inferred skills, balances capability against cost and latency, and applies natural language constraints:

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

## Documentation

Full documentation at [kostadindev.github.io/skillinfer](https://kostadindev.github.io/skillinfer):

- [Quickstart](https://kostadindev.github.io/skillinfer/getting-started/quickstart/)
- [Tutorials](https://kostadindev.github.io/skillinfer/tutorials/llm-benchmarks/) — LLM benchmarks, human skills, ESCO, agent orchestration
- [How It Works](https://kostadindev.github.io/skillinfer/how-it-works/kalman-update/) — Kalman update, covariance estimation, computational cost
- [API Reference](https://kostadindev.github.io/skillinfer/api/population/) — Population, Profile, Datasets, Visualization

## License

MIT
