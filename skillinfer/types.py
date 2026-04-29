"""Core types: Skill and Task."""

from __future__ import annotations


class Skill:
    """A label-score pair for a single skill measurement.

    Anywhere the API accepts a feature name (str), it also accepts a Skill.
    When a score is set, ``profile.observe(skill)`` uses it directly.

    Parameters
    ----------
    name : the skill identifier (must match a column in the Population).
    score : observed score for this skill (optional).
    description : human-readable description of what this skill measures.

    Examples
    --------
    >>> Skill("Programming", score=0.92)
    >>> Skill("BBH", description="Big-Bench Hard: diverse challenging tasks")
    >>> Skill("Programming", score=0.92, description="General programming ability")
    >>> Skill("Writing")  # label only
    """

    __slots__ = ("name", "score", "description")

    def __init__(self, name: str, *, score: float | None = None, description: str = ""):
        self.name = name
        self.score = score
        self.description = description

    def __repr__(self) -> str:
        parts = [repr(self.name)]
        if self.score is not None:
            parts.append(repr(self.score))
        if self.description:
            parts.append(repr(self.description))
        return f"Skill({', '.join(parts)})"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Skill):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


class Task:
    """A task described as a weighted combination of skills.

    Anywhere the API accepts a task vector (dict), it also accepts a Task.

    Parameters
    ----------
    weights : dict mapping skill names (str or Skill) to importance weights.
    description : human-readable description of the task.

    Examples
    --------
    >>> Task({"MATH Lvl 5": 1.0, "GPQA": 0.5}, "Math-heavy reasoning")
    >>> Task({"code_generation": 1.0, "testing": 0.5})
    """

    __slots__ = ("weights", "description")

    def __init__(
        self,
        weights: dict[str | Skill, float],
        description: str = "",
    ):
        self.weights = {
            (k.name if isinstance(k, Skill) else str(k)): float(v)
            for k, v in weights.items()
        }
        self.description = description

    def __repr__(self) -> str:
        if self.description:
            return f"Task({self.weights}, {self.description!r})"
        return f"Task({self.weights})"

    def __str__(self) -> str:
        if self.description:
            return self.description
        skills = ", ".join(f"{k}={v:.1f}" for k, v in self.weights.items())
        return f"Task({skills})"
