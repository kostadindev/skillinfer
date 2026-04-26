---
hide:
  - navigation
---

<div class="hero" markdown>

# skillinfer

**Infer a full skill profile from a few observations.**
{ .hero-tagline }

Evaluate an AI model on one benchmark and predict its performance on 37 others. Observe a worker on one task and infer their full capability profile. `skillinfer` learns how skills co-vary across a population and uses that structure to fill in the gaps — so you can understand what an agent or person can do without testing everything.

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[API Reference](api/taxonomy.md){ .md-button }

</div>

```python
import skillinfer

tax = skillinfer.Taxonomy.from_dataframe(df)   # learn how skills co-vary
state = tax.new_state()                        # new agent, unknown profile
state.observe("BBH", 55.0)                     # evaluate one benchmark
print(state.mean("MMLU-PRO"))                  # predict another → 37.1
print(state.most_uncertain(k=3))               # what to evaluate next?
```

---

<div class="grid" markdown>

<div class="card" markdown>
### One test, full profile
Evaluate an AI model on a single benchmark or observe a worker on a single task — instantly predict their performance across every other skill. Skills are correlated: reasoning predicts math, writing predicts comprehension.
</div>

<div class="card" markdown>
### Milliseconds, not hours
No training loop, no GPU, no neural network. Just a Kalman filter over the population covariance — the exact Bayesian posterior in a single matrix operation. Build a profile in under 1ms.
</div>

<div class="card" markdown>
### Know what to test next
`most_uncertain()` tells you which skill to evaluate next for maximum information gain. Stop running redundant benchmarks. Stop giving redundant assessments. Test only what matters.
</div>

<div class="card" markdown>
### AI agents and humans
Built for capability estimation in Human-AI Orchestration. Works with LLM benchmark leaderboards (4,500+ models), O\*NET occupational skills (894 occupations x 120 skills), or any population where skills co-vary.
</div>

</div>

---

## The problem

Every agent — AI or human — has a **skill vector**: a profile of capabilities across many dimensions. An LLM's skill vector is its benchmark scores. A worker's skill vector is their competency ratings across skills, knowledge, and abilities.

You rarely get to observe the full vector. Evaluating an AI model on 38 benchmarks costs compute. Assessing a worker on 120 competencies is impractical. But skills are correlated — a model good at logical deduction is likely good at causal reasoning; a strong programmer likely has strong analytical reasoning.

**`skillinfer` infers the full skill vector from a few observations.** Each observation updates all dimensions simultaneously, weighted by how skills co-vary across the population. The result is an **agent vector** — a probabilistic estimate of the full profile, with uncertainty on every dimension.

Because skill vectors, task vectors, and agent vectors all live in the same space, you can directly match agents to tasks: which model is best for this benchmark? Which worker fits this role?

The algorithm is a multivariate Kalman filter — the exact Bayesian update for linear-Gaussian observations. Developed as part of a [Cambridge master's thesis on Human-AI Orchestration](https://github.com/kostadindev/skillinfer), where it powers few-shot capability estimation for both human workers (via O\*NET) and AI agents (via benchmark leaderboards). See [Core Concepts](getting-started/concepts.md) for the full framework.

---

## Use Cases

| Domain | What you observe | What you predict | Scale |
|--------|-----------------|------------------|-------|
| **AI agent capabilities** | 1-2 benchmark scores | Performance on all other benchmarks | 4,500+ models x 6-38 benchmarks |
| **Human skill profiling** | A few task observations | Full occupational skill profile | 894 occupations x 120 skills |
| **Worker-task matching** | Known competencies | Fit for new roles and tasks | Skills, knowledge, abilities |
| **Student assessment** | One exam score | Mastery across all subjects | Any student-subject matrix |
| **Model selection** | Partial evaluations | Which model fits a new task | Any model-benchmark matrix |

---

## Quick Install

```bash
pip install skillinfer
```

See the [Installation guide](getting-started/installation.md) for optional dependencies and development setup.
