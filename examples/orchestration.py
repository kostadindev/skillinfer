"""
Agent Orchestration: use skill profiles as context for LLM routing.

skillinfer profiles are not a heuristic for matching — they are structured
context that you feed to an LLM orchestrator alongside task details, cost
constraints, and other business logic. The LLM makes the final decision.

This example:
  1. Builds skill profiles for a pool of agents from partial evaluations
  2. Formats profiles + additional context into a prompt
  3. Calls an LLM to make the routing decision

Run:
    export OPENAI_API_KEY=sk-...
    python examples/orchestration.py

Without an API key, the script prints the full prompt so you can see
exactly what context the orchestrator receives.
"""

import os
import pandas as pd
import skillinfer
from skillinfer import Task


# ── Step 1: Build skill profiles from partial evaluations ────────────

# Historical data: models you've fully evaluated (training data)
history = pd.DataFrame({
    "math":      [0.88, 0.75, 0.70, 0.82, 0.65, 0.90, 0.72, 0.68],
    "code":      [0.85, 0.80, 0.65, 0.78, 0.60, 0.88, 0.70, 0.62],
    "reasoning": [0.90, 0.82, 0.75, 0.85, 0.70, 0.92, 0.78, 0.72],
    "writing":   [0.80, 0.85, 0.90, 0.75, 0.88, 0.78, 0.82, 0.92],
    "creativity":[0.70, 0.78, 0.85, 0.68, 0.82, 0.65, 0.75, 0.88],
}, index=[f"model_{i}" for i in range(8)])

pop = skillinfer.Population.from_dataframe(history, normalize=False)

# Each agent has been evaluated on a *different subset* of skills.
# skillinfer infers the rest from covariance.
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


# ── Step 2: Additional context (cost, latency, limits) ───────────────

agent_metadata = {
    "gpt-4o":        {"cost_per_1k": 0.005, "avg_latency_ms": 800,  "rate_limit_rpm": 500},
    "claude-3.5":    {"cost_per_1k": 0.003, "avg_latency_ms": 600,  "rate_limit_rpm": 1000},
    "llama-3-70b":   {"cost_per_1k": 0.001, "avg_latency_ms": 1200, "rate_limit_rpm": 200},
    "gemini-pro":    {"cost_per_1k": 0.004, "avg_latency_ms": 700,  "rate_limit_rpm": 600},
    "mistral-large": {"cost_per_1k": 0.002, "avg_latency_ms": 500,  "rate_limit_rpm": 800},
}


# ── Step 3: Format context for the orchestrating LLM ─────────────────

def build_orchestrator_prompt(task: Task, profiles: dict, metadata: dict) -> str:
    """Build a prompt that gives an LLM everything it needs to route a task."""

    # Skill profile context from skillinfer
    agent_blocks = []
    for name, profile in profiles.items():
        lines = [f"{name}:"]

        # Skill predictions (observed vs inferred)
        lines.append("  Skill profile:")
        for skill in sorted(task.weights, key=lambda s: -task.weights[s]):
            pred = profile.predict(skill)
            source = "observed" if pred["std"] < 0.01 else "inferred"
            lines.append(f"    {skill}: {pred['mean']:.2f} ± {pred['std']:.2f} ({source})")

        # Overall task fit
        result = profile.match_score(task, threshold=0.8)
        lines.append(f"  Task fit: {result.score:.2f} (P>0.8 = {result.p_above_threshold:.0%})")

        # Operational metadata
        meta = metadata[name]
        lines.append(f"  Cost: ${meta['cost_per_1k']}/1k tokens, "
                     f"Latency: {meta['avg_latency_ms']}ms, "
                     f"Rate limit: {meta['rate_limit_rpm']} req/min")

        agent_blocks.append("\n".join(lines))

    prompt = f"""You are an AI orchestrator. A task has arrived and you need to assign it to the best available agent.

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
- Budget is moderate: avoid the most expensive option unless it's clearly the best fit
- If an agent's key skills are "inferred" rather than "observed", note the added risk

## Instructions
Pick one agent for this task. Explain your reasoning in 2-3 sentences, referencing
the skill profiles, confidence levels, and operational constraints. Then state your
choice on a final line as: ASSIGNED: <agent name>"""

    return prompt


# ── Step 4: Route a task ──────────────────────────────────────────────

task = Task(
    {"math": 1.0, "reasoning": 0.8, "code": 0.3},
    "Solve a competition-level math problem that requires writing a Python proof",
)

prompt = build_orchestrator_prompt(task, profiles, agent_metadata)

print("=" * 70)
print("ORCHESTRATOR PROMPT")
print("=" * 70)
print(prompt)
print("=" * 70)

# Call the LLM if an API key is available
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\nORCHESTRATOR RESPONSE")
    print("=" * 70)
    print(response.choices[0].message.content)
else:
    print("\nSet OPENAI_API_KEY to see the LLM's routing decision.")
    print("The prompt above is exactly what the orchestrator receives.")
