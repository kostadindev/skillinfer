# Tutorial: Agent Orchestration

Skill profiles are not a heuristic for matching — they are **structured context you feed to an LLM orchestrator** alongside cost, latency, and business constraints. The LLM makes the final routing decision; `skillinfer` gives it the capability data it needs.

!!! info "What you'll learn"
    - Building profiles from partial evaluations across a pool of agents
    - Combining skill profiles with operational metadata (cost, latency)
    - Formatting everything into a prompt for an LLM orchestrator
    - Letting the LLM reason about observed vs. inferred skills

## The pattern

```
Partial evals  →  skillinfer profiles  →  LLM prompt  →  Routing decision
                  (structured context)    + cost/latency
                                          + constraints
```

`skillinfer` handles step 2: turning sparse, partial evaluations into full skill profiles with calibrated uncertainty. The orchestrating LLM handles the rest — weighing capability against cost, noting which skills are observed vs. inferred, and applying business rules you couldn't encode in a scoring function.

## Step 1: Build skill profiles

```python
import pandas as pd
import skillinfer
from skillinfer import Task

# Historical data: models you've fully evaluated
history = pd.DataFrame({
    "math":      [0.88, 0.75, 0.70, 0.82, 0.65, 0.90, 0.72, 0.68],
    "code":      [0.85, 0.80, 0.65, 0.78, 0.60, 0.88, 0.70, 0.62],
    "reasoning": [0.90, 0.82, 0.75, 0.85, 0.70, 0.92, 0.78, 0.72],
    "writing":   [0.80, 0.85, 0.90, 0.75, 0.88, 0.78, 0.82, 0.92],
    "creativity":[0.70, 0.78, 0.85, 0.68, 0.82, 0.65, 0.75, 0.88],
}, index=[f"model_{i}" for i in range(8)])

pop = skillinfer.Population.from_dataframe(history, normalize=False)
```

Each new agent has been evaluated on a **different subset** of skills. `skillinfer` infers the rest from covariance:

```python
agent_observations = {
    "gpt-4o":        {"reasoning": 0.92, "code": 0.89},
    "claude-3.5":    {"reasoning": 0.90, "writing": 0.95},
    "llama-3-70b":   {"code": 0.78},
    "gemini-pro":    {"reasoning": 0.85, "math": 0.88, "code": 0.82},
    "mistral-large": {"writing": 0.80},
}

profiles = {
    name: pop.profile().observe_many(obs)
    for name, obs in agent_observations.items()
}
```

Now every agent has a full 5-skill profile — even `llama-3-70b`, which was only evaluated on `code`.

## Step 2: Add operational metadata

Skill profiles are necessary but not sufficient. The orchestrator also needs cost, latency, and capacity data:

```python
agent_metadata = {
    "gpt-4o":        {"cost_per_1k": 0.005, "avg_latency_ms": 800,  "rate_limit_rpm": 500},
    "claude-3.5":    {"cost_per_1k": 0.003, "avg_latency_ms": 600,  "rate_limit_rpm": 1000},
    "llama-3-70b":   {"cost_per_1k": 0.001, "avg_latency_ms": 1200, "rate_limit_rpm": 200},
    "gemini-pro":    {"cost_per_1k": 0.004, "avg_latency_ms": 700,  "rate_limit_rpm": 600},
    "mistral-large": {"cost_per_1k": 0.002, "avg_latency_ms": 500,  "rate_limit_rpm": 800},
}
```

## Step 3: Build the orchestrator prompt

This is the key step — combining skill profiles, task requirements, metadata, and constraints into a single prompt:

```python
task = Task(
    {"math": 1.0, "reasoning": 0.8, "code": 0.3},
    "Solve a competition-level math problem that requires writing a Python proof",
)

# Build agent context blocks
agent_blocks = []
for name, profile in profiles.items():
    lines = [f"{name}:"]

    # Skill predictions — the LLM sees what's observed vs. inferred
    lines.append("  Skill profile:")
    for skill in sorted(task.weights, key=lambda s: -task.weights[s]):
        pred = profile.predict(skill)
        source = "observed" if pred["std"] < 0.01 else "inferred"
        lines.append(f"    {skill}: {pred['mean']:.2f} ± {pred['std']:.2f} ({source})")

    # Overall task fit
    result = profile.match_score(task, threshold=0.8)
    lines.append(f"  Task fit: {result.score:.2f} (P>0.8 = {result.p_above_threshold:.0%})")

    # Operational metadata
    meta = agent_metadata[name]
    lines.append(f"  Cost: ${meta['cost_per_1k']}/1k tokens, "
                 f"Latency: {meta['avg_latency_ms']}ms, "
                 f"Rate limit: {meta['rate_limit_rpm']} req/min")

    agent_blocks.append("\n".join(lines))
```

The prompt gives the LLM the full picture:

```python
prompt = f"""You are an AI orchestrator. A task has arrived and you need to
assign it to the best available agent.

## Task
{task.description}
Required skills (importance weights): {task.weights}

## Available agents
Each agent has a skill profile built from partial evaluations. "Observed" skills
were directly measured; "inferred" skills are predicted from correlated observations
(with higher uncertainty). Task fit is the expected performance score (0-1 scale).

{chr(10).join(agent_blocks)}

## Constraints
- The task is latency-sensitive: prefer agents under 1000ms
- Budget is moderate: avoid the most expensive option unless clearly the best fit
- If an agent's key skills are "inferred" rather than "observed", note the added risk

## Instructions
Pick one agent. Explain your reasoning in 2-3 sentences, referencing the skill
profiles, confidence levels, and operational constraints. Then state your choice
on a final line as: ASSIGNED: <agent name>"""
```

The resulting prompt looks like:

```text
gpt-4o:
  Skill profile:
    math: 0.89 ± 0.04 (inferred)
    reasoning: 0.92 ± 0.00 (observed)
    code: 0.89 ± 0.00 (observed)
  Task fit: 0.90 (P>0.8 = 100%)
  Cost: $0.005/1k tokens, Latency: 800ms, Rate limit: 500 req/min

gemini-pro:
  Skill profile:
    math: 0.88 ± 0.00 (observed)
    reasoning: 0.85 ± 0.00 (observed)
    code: 0.82 ± 0.00 (observed)
  Task fit: 0.86 (P>0.8 = 100%)
  Cost: $0.004/1k tokens, Latency: 700ms, Rate limit: 600 req/min

claude-3.5:
  Skill profile:
    math: 0.82 ± 0.04 (inferred)
    reasoning: 0.90 ± 0.00 (observed)
    code: 0.81 ± 0.04 (inferred)
  Task fit: 0.85 (P>0.8 = 99%)
  Cost: $0.003/1k tokens, Latency: 600ms, Rate limit: 1000 req/min
...
```

Notice what the LLM can reason about that a scoring function can't:

- **gpt-4o** has the best task fit but its math score is *inferred* — should the LLM trust it?
- **gemini-pro** has all three skills *observed* (100% confidence) at a lower cost
- **claude-3.5** is cheapest and fastest, but both math and code are inferred

## Step 4: Call the orchestrator

```python
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)
```

```text
Considering the task requires a balance of math, reasoning, and coding skills,
gpt-4o stands out with the highest task fit (0.90) and 100% probability of
exceeding the threshold. However, its math score is inferred rather than observed.
Gemini-pro offers all three skills directly observed with 100% confidence at a
lower cost ($0.004 vs $0.005/1k) and lower latency (700ms vs 800ms). Given the
moderate budget constraint and latency sensitivity, gemini-pro provides the best
balance of confidence and cost.

ASSIGNED: gemini-pro
```

The LLM chose gemini-pro over gpt-4o — not because its score was higher (it wasn't), but because all its relevant skills were *observed* rather than inferred, and it's cheaper. This is exactly the kind of nuanced decision that requires an LLM, not a `max()` call.

## Full example

See [`examples/orchestration.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/orchestration.py) for the complete runnable script. Without an API key, it prints the full prompt so you can inspect the context.

## Key takeaway

`skillinfer` is not the orchestrator — it's the **context layer**. It turns partial, heterogeneous evaluations into structured skill profiles that an LLM can reason about. The LLM then combines capability data with cost, latency, confidence levels, and business constraints to make a routing decision that no scoring function could replicate.
