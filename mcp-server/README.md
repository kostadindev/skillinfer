# skillinfer-mcp

MCP server for [skillinfer](https://github.com/kostadindev/bayeskal) — Bayesian skill inference for AI agents and humans.

Lets any MCP-compatible AI agent (Claude, Cursor, VS Code Copilot, etc.) build and query skill profiles via tool calls.

## Install

```bash
pip install skillinfer-mcp
```

Or run directly without installing:

```bash
uvx skillinfer-mcp
```

## Configure

Add to your Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "skillinfer": {
      "command": "uvx",
      "args": ["skillinfer-mcp"]
    }
  }
}
```

## Tools

### Populations

| Tool | Description |
|---|---|
| `load_dataset` | Load built-in dataset (O\*NET or ESCO) |
| `load_population_from_csv` | Load population from CSV file |
| `load_population_from_parquet` | Load population from Parquet file |
| `list_populations` | List loaded populations |
| `population_summary` | Summary statistics for a population |
| `list_features` | List feature names in a population |

### Profiles

| Tool | Description |
|---|---|
| `create_profile` | Create a new skill profile |
| `observe` | Observe a single skill score |
| `observe_many` | Observe multiple skills at once |
| `predict` | Predict all skills with uncertainty |
| `most_uncertain` | Find the most uncertain skills (for active learning) |
| `profile_summary` | Summary statistics for a profile |
| `list_profiles` | List active profiles |

### Task matching

| Tool | Description |
|---|---|
| `match_task` | Score a profile against a weighted task |
| `rank_agents` | Rank multiple profiles against a task |

### Persistence

| Tool | Description |
|---|---|
| `save_profile` | Save profile to JSON |
| `load_profile` | Load profile from JSON (requires population) |

## Example session

```
Agent: load_dataset(name="onet", dataset="onet")
→ Population 'onet' loaded: 894 entities x 120 features.

Agent: create_profile(name="alice", population="onet")
→ Profile 'alice' created (120 features, prior=population mean).

Agent: observe(profile="alice", skill="Skill:Programming", score=0.92)
→ Observed Skill:Programming=0.920 on 'alice'. Observations: 1.
  Top impacted features:
    Skill:Programming: +0.3891
    Knowledge:Computers and Electronics: +0.1842
    ...

Agent: most_uncertain(profile="alice", k=3)
→ [{"feature": "Skill:X", "mean": 0.51, "std": 0.12}, ...]

Agent: predict(profile="alice", skill="Knowledge:Mathematics")
→ {"mean": 0.68, "std": 0.09, "ci_lower": 0.50, "ci_upper": 0.86, ...}
```
