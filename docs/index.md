---
hide:
  - navigation
---

<div class="hero" markdown>

# skillinfer

**Infer a full skill profile from a few observations.**
{ .hero-tagline }

One benchmark predicts 37 others. One task observation reveals 120 skills.
`skillinfer` learns how capabilities co-vary across a population and fills in the gaps, so you can understand what any agent, person, or team can do — without testing everything.

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[API Reference](api/population.md){ .md-button }

</div>

```python
import skillinfer

pop = skillinfer.datasets.onet()         # 894 occupations x 120 skills
profile = pop.profile()                  # new entity, unknown
profile.observe("Skill:Programming", 0.92)
print(profile.predict())                 # predict all 120 skills
```

```text title="Output — one observation, 120 predicted"
                           feature   mean    std  ci_lower  ci_upper
    Skill:Complex Problem Solving   0.81   0.17      0.47      1.00
          Skill:Critical Thinking   0.73   0.15      0.43      1.00
              Skill:Programming     0.92   0.01      0.90      0.93  ← observed
                Skill:Mathematics   0.67   0.12      0.43      0.91
         Ability:Static Strength   0.10   0.23      0.00      0.55  ← anti-correlated
...                                 ...    ...       ...       ...
[120 rows x 5 columns]
```

---

<div class="grid" markdown>

<div class="card" markdown>
### :robot: AI agents
Run one benchmark, predict the rest. Build a full capability profile for any model and hand it to an orchestrator — route tasks to the right model without exhaustive evals.
</div>

<div class="card" markdown>
### :bust_in_silhouette: Humans
Observe a few tasks, infer 120+ competencies. Surface skill gaps, match candidates to roles, and prioritize what to assess next — all from partial data.
</div>

<div class="card" markdown>
### :people_holding_hands: Human-AI teams
People and models profiled in the same skill space. Compare everyone on the same dimensions, then let an orchestrator assign each subtask to whoever is strongest.
</div>

<div class="card" markdown>
### :zap: Fast and scalable
No training loop, no GPU, no neural network. One matrix operation gives you the exact Bayesian posterior. Under 1ms per update, scales to 1000+ skills.
</div>

</div>

```bash
pip install skillinfer
```

---

## Core types

| Type | What it is | Example |
|------|-----------|---------|
| **`Population`** | Learned covariance from a population of entities | `Population.from_dataframe(df)` |
| **`Profile`** | One entity's skill profile — gets sharper with observations | `pop.profile()` → `.observe()` → `.predict()` |
| **`Skill`** | A label-score pair for a single skill measurement | `Skill("Programming", score=0.92)` |
| **`Task`** | A weighted mix of skills describing what a job requires | `Task({"math": 1.0, "code": 0.5})` |

---

## Use cases

| Domain | Observe | Predict | Scale |
|--------|---------|---------|-------|
| **AI model selection** | 1-2 benchmark scores | All benchmarks + best model for a task | 4,500+ models x 6-38 benchmarks |
| **Human skill profiling** | A few task observations | Full occupational skill profile | 894 occupations x 120 skills |
| **Human-AI orchestration** | Partial evals for both | Who handles which subtask | Mixed pools, same skill space |
| **Worker-task matching** | Known competencies | Fit for new roles and tasks | Skills, knowledge, abilities |

---

## Examples

`skillinfer` works anywhere entities have correlated capabilities. The same API profiles an LLM, a job candidate, or a hybrid team.

=== "AI Agents"

    ```python
    import pandas as pd
    import skillinfer

    # 4,500+ LLMs from the Open LLM Leaderboard
    df = pd.read_parquet("hf://datasets/open-llm-leaderboard/contents/data/train-00000-of-00001.parquet")
    benchmarks = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
    df = df[["fullname"] + benchmarks].dropna().set_index("fullname")

    # Hold out Llama-3.1-70B-Instruct, build population from the rest
    model = "meta-llama/Llama-3.1-70B-Instruct"
    true_scores = df.loc[model]
    pop = skillinfer.Population.from_dataframe(df.drop(model), normalize=False)

    # Observe ONE benchmark, predict the other five
    profile = pop.profile()
    profile.observe("BBH", true_scores["BBH"])

    # Compare predictions to ground truth
    for b in benchmarks:
        pred = profile.mean(b)
        err = abs(true_scores[b] - pred)
        tag = "  ← observed" if b == "BBH" else ""
        print(f"  {b:<15} true={true_scores[b]:5.1f}  pred={pred:5.1f}  err={err:4.1f}{tag}")
    ```

    ```text title="Output — one observation predicts five benchmarks"
      IFEval          true= 86.7  pred= 69.3  err=17.4
      BBH             true= 55.9  pred= 55.9  err= 0.0  ← observed
      MATH Lvl 5      true= 38.1  pred= 35.1  err= 3.0
      GPQA            true= 14.2  pred= 14.6  err= 0.4
      MUSR            true= 17.7  pred= 17.8  err= 0.1
      MMLU-PRO        true= 47.9  pred= 50.7  err= 2.9

      MAE on 5 unobserved benchmarks: 4.7
    ```

=== "Humans"

    ```python
    import skillinfer

    # O*NET 30.2: 894 occupations x 120 skills, knowledge, abilities
    pop = skillinfer.datasets.onet()

    profile = pop.profile()
    profile.observe("Skill:Programming", 0.92)
    profile.observe("Skill:Critical Thinking", 0.85)

    print(profile.predict())                   # predict all 120
    ```

    ```text title="Output — two observations, 120 predicted"
                              feature   mean    std  ci_lower  ci_upper
              Skill:Active Learning   0.94   0.12      0.71      1.17
             Skill:Active Listening   0.74   0.10      0.55      0.93
      Skill:Complex Problem Solving   1.02   0.09      0.83      1.20
            Skill:Critical Thinking   0.85   0.01      0.83      0.87  ← observed
                Skill:Programming     0.92   0.01      0.91      0.93  ← observed
                  Skill:Mathematics   0.81   0.11      0.59      1.03
    ...                               ...    ...       ...       ...
    [120 rows]
    ```

=== "Human-AI Teams"

    ```python
    import pandas as pd, skillinfer
    from skillinfer import Task

    # Humans and models scored on the same skills
    df = pd.DataFrame({
        "math":       [0.9, 0.7, 0.85, 0.95, 0.80],
        "code":       [0.8, 0.5, 0.90, 0.92, 0.70],
        "writing":    [0.6, 0.9, 0.40, 0.55, 0.85],
        "reasoning":  [0.8, 0.6, 0.80, 0.88, 0.75],
    }, index=["alice", "bob", "gpt-4o", "claude", "gemini"])

    pop = skillinfer.Population.from_dataframe(df, normalize=False)

    # Observe one skill each
    alice = pop.profile(); alice.observe("math", 0.95)
    gpt4o = pop.profile(); gpt4o.observe("math", 0.88)

    # Who handles a math-heavy task?
    task = Task({"math": 1.0, "reasoning": 0.5})
    ranking = skillinfer.rank_agents(task, {"alice": alice, "gpt-4o": gpt4o})
    print(ranking)
    ```

    ```text title="Output — ranked by expected task performance"
        agent  expected_score    std
    0   alice            0.91   0.03
    1  gpt-4o            0.85   0.03
    ```

---

## Skill profiles as context for LLM orchestration

Skill profiles are not a heuristic for matching — they are **structured context you feed to an LLM** alongside natural language instructions. The LLM makes the routing decision; `skillinfer` gives it the capability data.

```python
import pandas as pd
from openai import OpenAI
import skillinfer
from skillinfer import Task

# --- 1. Build skill profiles from partial evaluations ---

history = pd.DataFrame({
    "math": [0.88, 0.75, 0.82, 0.90, 0.72], "code": [0.85, 0.80, 0.78, 0.88, 0.70],
    "reasoning": [0.90, 0.82, 0.85, 0.92, 0.78], "writing": [0.80, 0.85, 0.75, 0.78, 0.92],
}, index=[f"model_{i}" for i in range(5)])
pop = skillinfer.Population.from_dataframe(history, normalize=False)

# Each agent evaluated on different skills — skillinfer infers the rest
agents = {
    "gpt-4o":      {"reasoning": 0.92, "code": 0.89},
    "claude-3.5":  {"reasoning": 0.90, "writing": 0.95},
    "gemini-pro":  {"math": 0.88, "code": 0.82},
}
profiles = {
    name: pop.profile().observe_many(obs)
    for name, obs in agents.items()
}

# --- 2. Format profiles as LLM context ---

task = Task({"math": 1.0, "reasoning": 0.8, "code": 0.3})

agent_context = ""
for name, profile in profiles.items():
    agent_context += f"\n{name}:\n"
    for skill in ["math", "reasoning", "code"]:
        pred = profile.predict(skill)
        source = "observed" if pred["std"] < 0.01 else "inferred"
        agent_context += f"  {skill}: {pred['mean']:.2f} ± {pred['std']:.2f} ({source})\n"

prompt = f"""Pick an agent for this task.

Task: Solve a competition math problem, write a Python proof.
Skill requirements: math (critical), reasoning (important), code (helpful).

Agent capabilities (from partial evaluations):
{agent_context}
"Observed" = directly measured. "Inferred" = predicted, higher uncertainty.

Constraints:
- Prefer agents whose critical skills (math) are observed, not inferred.
- If two agents are close, prefer the one with lower uncertainty.
- A slightly weaker agent with observed skills can be better than a
  stronger agent whose key skills are only inferred.

Pick one agent. One sentence of reasoning, then: ASSIGNED: <name>"""

# --- 3. Ask the LLM ---

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=100,
    messages=[{"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)
```

```text title="LLM response"
Gemini-pro has directly observed math (0.88) and code (0.82), while gpt-4o's
math is only inferred. Both are close, but gemini-pro's critical skill is
measured, not predicted — making it the safer choice.

ASSIGNED: gemini-pro
```

The LLM weighs observed vs. inferred confidence, balances skill scores against natural language constraints ("slightly weaker but observed can be better"), and arrives at a decision no scoring function could replicate. Change the constraints, and the same profiles route differently.

See the [Agent Orchestration tutorial](tutorials/orchestration.md) for the full walkthrough.
