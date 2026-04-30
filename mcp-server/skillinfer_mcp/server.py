"""MCP server for skillinfer — Bayesian skill inference.

Exposes skillinfer's Population/Profile API as MCP tools so any
MCP-compatible agent can build and query skill profiles.

State (populations, profiles) lives in server memory. Profiles
can be saved/loaded via JSON for persistence across sessions.

Usage:
    skillinfer-mcp                    # stdio transport (default)
    python -m skillinfer_mcp.server   # same
"""

from __future__ import annotations

import json

import numpy as np
import skillinfer
from skillinfer import Population, Profile

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server + in-memory state
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "skillinfer",
    instructions=(
        "Bayesian skill inference server. Load a population (dataset or "
        "CSV/Parquet), create profiles for entities, observe skills, and "
        "predict full capability profiles with uncertainty.\n\n"
        "Typical workflow:\n"
        "1. load_dataset or load_population_from_csv\n"
        "2. create_profile\n"
        "3. observe / observe_many (repeat)\n"
        "4. predict or match_task\n\n"
        "Use most_uncertain to drive active learning — it tells you "
        "which skill to observe next for maximum information gain."
    ),
)

_populations: dict[str, Population] = {}
_profiles: dict[str, Profile] = {}
_profile_population: dict[str, str] = {}  # profile name -> population name

_ROUND = 4  # decimal places for numeric output


def _pop_or_error(name: str) -> Population:
    if name not in _populations:
        available = list(_populations.keys()) or ["(none loaded)"]
        raise ValueError(
            f"Population {name!r} not found. Available: {available}"
        )
    return _populations[name]


def _profile_or_error(name: str) -> Profile:
    if name not in _profiles:
        available = list(_profiles.keys()) or ["(none created)"]
        raise ValueError(
            f"Profile {name!r} not found. Available: {available}"
        )
    return _profiles[name]


def _round(x):
    """Round floats for compact output."""
    if isinstance(x, float):
        return round(x, _ROUND)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _df_to_records(df, round_digits=_ROUND) -> str:
    """DataFrame to compact JSON records with rounded numbers."""
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, (float, np.floating)):
                row[k] = round(float(v), round_digits)
    return json.dumps(records, indent=2)


# ---------------------------------------------------------------------------
# Population tools
# ---------------------------------------------------------------------------

@mcp.tool()
def load_dataset(name: str, dataset: str = "onet") -> str:
    """Load a built-in dataset as a population.

    Args:
        name: Name to assign to this population (used to reference it later).
        dataset: Built-in dataset. "onet" (894 occupations x 120 skills)
            or "esco" (2999 occupations x 134 binary skills).
    """
    if dataset == "onet":
        pop = skillinfer.datasets.onet()
    elif dataset == "esco":
        pop = skillinfer.datasets.esco()
    else:
        return f"Unknown dataset {dataset!r}. Choose 'onet' or 'esco'."
    _populations[name] = pop
    N, K = pop.matrix.shape
    return f"Population '{name}' loaded: {N} entities x {K} features."


@mcp.tool()
def load_population_from_csv(
    name: str,
    path: str,
    normalize: bool = True,
    index_col: int = 0,
) -> str:
    """Load a population from a CSV file (rows=entities, columns=features).

    Args:
        name: Name to assign to this population.
        path: Path to the CSV file.
        normalize: Scale each column to [0, 1] (recommended).
        index_col: Which column to use as the row index.
    """
    pop = Population.from_csv(path, index_col=index_col, normalize=normalize)
    _populations[name] = pop
    N, K = pop.matrix.shape
    return f"Population '{name}' loaded from CSV: {N} entities x {K} features."


@mcp.tool()
def load_population_from_parquet(
    name: str,
    path: str,
    normalize: bool = True,
) -> str:
    """Load a population from a Parquet file.

    Args:
        name: Name to assign to this population.
        path: Path to the Parquet file.
        normalize: Scale each column to [0, 1] (recommended).
    """
    pop = Population.from_parquet(path, normalize=normalize)
    _populations[name] = pop
    N, K = pop.matrix.shape
    return f"Population '{name}' loaded from Parquet: {N} entities x {K} features."


@mcp.tool()
def list_populations() -> str:
    """List all loaded populations with their dimensions."""
    if not _populations:
        return "No populations loaded. Use load_dataset or load_population_from_csv."
    lines = []
    for name, pop in _populations.items():
        N, K = pop.matrix.shape
        lines.append(f"  {name}: {N} entities x {K} features")
    return "Loaded populations:\n" + "\n".join(lines)


@mcp.tool()
def population_summary(population: str) -> str:
    """Get summary statistics for a loaded population.

    Returns entity/feature counts, covariance condition number,
    effective dimensions, and top feature correlations.

    Args:
        population: Name of a loaded population.
    """
    pop = _pop_or_error(population)
    return json.dumps(_round(pop.summary()), indent=2, default=str)


@mcp.tool()
def list_features(population: str, search: str = "") -> str:
    """List feature names in a population.

    Args:
        population: Name of a loaded population.
        search: Optional substring filter (case-insensitive).
            e.g. "math" returns only features containing "math".
    """
    pop = _pop_or_error(population)
    names = pop.feature_names
    if search:
        names = [n for n in names if search.lower() in n.lower()]
    return json.dumps({"total": len(pop.feature_names), "matched": len(names), "features": names})


# ---------------------------------------------------------------------------
# Profile tools
# ---------------------------------------------------------------------------

@mcp.tool()
def create_profile(
    name: str,
    population: str,
    prior_entity: str = "",
) -> str:
    """Create a new skill profile for an entity.

    Starts at the population mean (or a specific entity's values).
    Use observe() to refine it. Next step: observe skills or predict.

    Args:
        name: Name for this profile (used to reference it later).
        population: Name of the population to base this profile on.
        prior_entity: Optional entity name to use as the starting point.
    """
    pop = _pop_or_error(population)
    kwargs = {}
    if prior_entity:
        kwargs["prior_entity"] = prior_entity
    profile = pop.profile(**kwargs)
    _profiles[name] = profile
    _profile_population[name] = population
    K = len(profile.feature_names)
    base = f"prior='{prior_entity}'" if prior_entity else "prior=population mean"
    return f"Profile '{name}' created ({K} features, {base})."


@mcp.tool()
def observe(
    profile: str,
    skill: str,
    score: float,
) -> str:
    """Observe a single skill score for a profile.

    Updates predictions for ALL skills via learned covariance.
    Call most_uncertain() to find the best next skill to observe.

    Args:
        profile: Name of the profile to update.
        skill: Feature/skill name to observe.
        score: Observed score value (typically 0-1).
    """
    p = _profile_or_error(profile)
    p.observe(skill, score)

    # Include the top most uncertain feature as a hint
    unc = p.most_uncertain(k=1)
    next_hint = unc.iloc[0]

    return (
        f"Observed {skill}={score:.3f} on '{profile}'. "
        f"Total observations: {p.n_observations}. "
        f"Most uncertain remaining: {next_hint['feature']} "
        f"(std={next_hint['std']:.{_ROUND}f})."
    )


@mcp.tool()
def observe_many(
    profile: str,
    observations: dict[str, float],
) -> str:
    """Observe multiple skill scores at once.

    Args:
        profile: Name of the profile to update.
        observations: Mapping of skill names to scores,
            e.g. {"math": 0.9, "physics": 0.8}.
    """
    p = _profile_or_error(profile)
    p.observe_many(observations)
    return (
        f"Observed {len(observations)} skills on '{profile}'. "
        f"Total observations: {p.n_observations}."
    )


@mcp.tool()
def predict(
    profile: str,
    skill: str = "",
    top_k: int = 10,
) -> str:
    """Predict skill values with uncertainty.

    Returns predictions sorted by mean (descending). Each entry
    includes mean, std, confidence interval, confidence (0-1 how
    much uncertainty was reduced), and source (observed/predicted).

    Args:
        profile: Name of the profile.
        skill: Single skill to predict (returns one result).
        top_k: Number of top skills to return (default 10, 0 for all).
    """
    p = _profile_or_error(profile)

    if skill:
        result = p.predict(skill, detail=True)
        return json.dumps(_round(result), indent=2)

    df = p.predict(detail=True)
    if top_k > 0:
        df = df.nlargest(top_k, "mean")
    return _df_to_records(df)


@mcp.tool()
def most_uncertain(profile: str, k: int = 5) -> str:
    """Get the k skills with highest remaining uncertainty.

    Use this to decide what to observe next — observing the most
    uncertain skill maximally reduces overall prediction uncertainty.

    Args:
        profile: Name of the profile.
        k: Number of features to return.
    """
    p = _profile_or_error(profile)
    df = p.most_uncertain(k=k)
    return _df_to_records(df)


@mcp.tool()
def profile_summary(profile: str) -> str:
    """Get summary statistics for a profile.

    Returns observation count, uncertainty reduction (0-1),
    top predicted skills, and most uncertain skills.

    Args:
        profile: Name of the profile.
    """
    p = _profile_or_error(profile)
    return json.dumps(_round(p.summary()), indent=2)


@mcp.tool()
def list_profiles() -> str:
    """List all active profiles with their observation counts."""
    if not _profiles:
        return "No profiles created. Use create_profile first."
    lines = []
    for name, p in _profiles.items():
        pop_name = _profile_population.get(name, "?")
        lines.append(
            f"  {name}: {p.n_observations} observations, "
            f"{len(p.feature_names)} features (population: {pop_name})"
        )
    return "Active profiles:\n" + "\n".join(lines)


@mcp.tool()
def delete_profile(profile: str) -> str:
    """Delete a profile and free its memory.

    Args:
        profile: Name of the profile to delete.
    """
    _profile_or_error(profile)
    del _profiles[profile]
    _profile_population.pop(profile, None)
    return f"Profile '{profile}' deleted."


# ---------------------------------------------------------------------------
# Task matching tools
# ---------------------------------------------------------------------------

@mcp.tool()
def match_task(
    profile: str,
    task_weights: dict[str, float],
    threshold: float = -1.0,
) -> str:
    """Score a profile against a task.

    The task is a weighted combination of skills. Returns expected
    score, uncertainty, and P(score > threshold).

    Args:
        profile: Name of the profile.
        task_weights: Skill names to importance weights,
            e.g. {"math": 1.0, "reasoning": 0.5}.
        threshold: Compute P(score > threshold). Set to -1 to skip.
    """
    p = _profile_or_error(profile)
    th = threshold if threshold >= 0 else None
    result = p.match_score(task_weights, threshold=th)
    return json.dumps(_round({
        "score": result.score,
        "std": result.std,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "p_above_threshold": result.p_above_threshold,
    }), indent=2)


@mcp.tool()
def rank_agents(
    task_weights: dict[str, float],
    profiles: list[str] | None = None,
    threshold: float = -1.0,
) -> str:
    """Rank multiple profiles against a task.

    Returns profiles sorted by expected score (descending).

    Args:
        task_weights: Skill names to importance weights.
        profiles: Profile names to rank. If omitted, ranks all active profiles.
        threshold: Compute P(score > threshold). Set to -1 to skip.
    """
    names = profiles if profiles else list(_profiles.keys())

    if not names:
        return "No profiles to rank."

    profile_dict = {}
    for name in names:
        profile_dict[name] = _profile_or_error(name)

    th = threshold if threshold >= 0 else None
    df = skillinfer.rank_agents(task_weights, profile_dict, threshold=th)
    return _df_to_records(df)


# ---------------------------------------------------------------------------
# Persistence tools
# ---------------------------------------------------------------------------

@mcp.tool()
def save_profile(profile: str, path: str) -> str:
    """Save a profile to a JSON file.

    Includes observations so it can be fully restored with load_profile.

    Args:
        profile: Name of the profile to save.
        path: File path to write (e.g. "./alice.json").
    """
    p = _profile_or_error(profile)
    data = p.to_dict()
    pop_name = _profile_population.get(profile)
    if pop_name:
        data["population"] = pop_name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return f"Profile '{profile}' saved to {path} (population: {pop_name})."


@mcp.tool()
def load_profile(
    name: str,
    path: str,
    population: str,
) -> str:
    """Load a profile from a JSON file.

    Requires the same population used to create the original profile.

    Args:
        name: Name to assign to the loaded profile.
        path: Path to the JSON file.
        population: Name of the loaded population to reconstruct against.
    """
    pop = _pop_or_error(population)
    profile = Profile.from_json(path, population=pop)
    _profiles[name] = profile
    _profile_population[name] = population
    return (
        f"Profile '{name}' loaded from {path} "
        f"({profile.n_observations} observations, population: {population})."
    )


# ---------------------------------------------------------------------------
# Prompts — reusable workflows agents can discover and invoke
# ---------------------------------------------------------------------------

@mcp.prompt()
def assign_task(task_description: str, agent_names: str = "", population: str = "onet") -> str:
    """Assign a task to the best-fit agent or person.

    Given a task and candidates with known skills, infer full profiles
    and rank by expected fit. If agent_names is empty, ranks all
    currently loaded profiles.
    """
    candidates_block = (
        f"Candidates: {agent_names}\n" if agent_names
        else "Candidates: use list_profiles to see all loaded profiles.\n"
    )
    setup_block = (
        "3. For each candidate, call create_profile and observe their "
        "known skills. You only need the skills you know — the model "
        "infers the rest from covariance.\n"
        if agent_names else
        "3. Profiles are already loaded. Call list_profiles to see them.\n"
    )
    return (
        f"Assign a task to the best-fit agent using skillinfer.\n\n"
        f"Task: {task_description}\n"
        f"{candidates_block}\n"
        f"Steps:\n"
        f"1. Call load_dataset(name=\"{population}\", dataset=\"{population}\").\n"
        f"2. Call list_features(population=\"{population}\", search=\"...\") "
        f"to find skill names relevant to the task. Search for keywords "
        f"from the task description.\n"
        f"{setup_block}"
        f"4. Build task weights: map the task to a dict of skill names "
        f"and importance weights. Core skills = 1.0, supporting = 0.3-0.5.\n"
        f"5. Call rank_agents with the task weights and a threshold "
        f"(e.g. 0.7) to get P(score > threshold) for each candidate.\n"
        f"6. If the top candidate has P(above threshold) > 0.9, assign "
        f"to them confidently. If P < 0.7 for all, flag as \"no confident "
        f"match\".\n"
        f"7. Report: best candidate, expected score, confidence interval, "
        f"runner-up, and which unobserved skills drive the most uncertainty "
        f"(call most_uncertain on the top candidates)."
    )


@mcp.prompt()
def form_team(
    task_description: str,
    candidate_names: str,
    team_size: int = 3,
    population: str = "onet",
) -> str:
    """Form a team that covers all skill requirements for a task.

    Profiles candidates, identifies the skills the task needs,
    and selects a team that maximizes coverage with minimal gaps.
    """
    return (
        f"Form a team of {team_size} from candidates using skillinfer.\n\n"
        f"Task: {task_description}\n"
        f"Candidates: {candidate_names}\n\n"
        f"Steps:\n"
        f"1. Call load_dataset(name=\"{population}\", dataset=\"{population}\").\n"
        f"2. Identify task-relevant skills with list_features(search=\"...\").\n"
        f"3. For each candidate, create_profile and observe known skills.\n"
        f"4. Call predict for each candidate on the task-relevant skills.\n"
        f"5. Select the team greedily:\n"
        f"   a. Pick the candidate with the highest score on the most "
        f"critical skill.\n"
        f"   b. For each remaining slot, pick the candidate who best "
        f"covers the skills not already covered by the current team.\n"
        f"   c. \"Covered\" means someone on the team has predicted "
        f"mean > 0.7 for that skill.\n"
        f"6. Report: the team, which skills each member covers, and "
        f"any remaining skill gaps (no team member above 0.7).\n"
        f"7. For gaps, call most_uncertain on the team members — maybe "
        f"a gap is just high uncertainty, not low skill."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
