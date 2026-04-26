# datasets

Built-in datasets that return a `Population` ready for profiling. No downloads, no preprocessing — just import and go.

```python
import skillinfer

pop = skillinfer.datasets.onet()      # or skillinfer.datasets.esco()
profile = pop.profile()
profile.observe("Skill:Programming", 0.92)
print(profile.predict())
```

---

## `onet()`

**O\*NET 30.2** — U.S. Department of Labor occupational taxonomy.

| | |
|---|---|
| **Entities** | 894 occupations |
| **Features** | 120 (35 skills, 33 knowledge areas, 52 abilities) |
| **Scale** | Continuous importance ratings, normalised to [0, 1] |
| **Source** | [O\*NET 30.2](https://www.onetcenter.org/database.html), U.S. Department of Labor / ETA |
| **License** | CC BY 4.0 |

Feature names are prefixed by category: `Skill:Programming`, `Knowledge:Mathematics`, `Ability:Deductive Reasoning`.

```python
pop = skillinfer.datasets.onet()

# Profile a software developer
profile = pop.profile()
profile.observe("Skill:Programming", 0.92)
profile.observe("Skill:Critical Thinking", 0.85)
print(profile.predict())
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `bool` | `False` | Re-normalise columns to [0, 1]. Data is already normalised, so this is rarely needed. |

---

## `esco()`

**ESCO v1.2.1** — European Commission skills and competences taxonomy.

| | |
|---|---|
| **Entities** | 2,999 occupations |
| **Features** | 134 Level-2 skill groups |
| **Scale** | Binary (1 = occupation requires at least one essential skill in that group) |
| **Source** | [ESCO v1.2.1](https://esco.ec.europa.eu/), European Commission |

ESCO differs from O\*NET in three ways: it is curated by EU expert panels (not U.S. survey data), uses binary assignments (not continuous ratings), and defines an independent skill taxonomy (134 groups vs 120 features). This makes it useful for cross-taxonomy validation.

```python
pop = skillinfer.datasets.esco()

# Profile a teacher
profile = pop.profile()
profile.observe("education", 1.0)
profile.observe("teaching and training", 1.0)
print(profile.predict())
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `bool` | `False` | Normalise columns to [0, 1]. Data is binary, so this is rarely needed. |
