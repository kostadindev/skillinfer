# Datasets

Built-in datasets that return a `Population` ready for profiling. No downloads, no preprocessing — just import and go.

```python
import skillinfer

pop = skillinfer.datasets.onet()      # or skillinfer.datasets.esco()
profile = pop.profile()
profile.observe("Skill:Programming", 0.92)
print(profile.predict())
```

Both datasets are shipped as compressed Parquet files inside the package (~440 KB total).

---

## `onet()`

**O\*NET 30.2** — the U.S. Department of Labor's occupational information network.

O\*NET is the most comprehensive public database of occupational skill requirements. It describes what workers in each occupation need to know and be able to do, based on surveys of incumbent workers and occupational analysts.

```python
pop = skillinfer.datasets.onet()
print(pop)
# Population(894 agents x 120 skills, shrinkage=0.0054)
```

### What's in it

| | |
|---|---|
| **Entities** | 894 occupations (e.g., "Software Developers", "Registered Nurses", "Chief Executives") |
| **Features** | 120 total: 35 skills, 33 knowledge areas, 52 abilities |
| **Scale** | Continuous importance ratings, normalised to [0, 1] |
| **Source** | [O\*NET 30.2](https://www.onetcenter.org/database.html), U.S. Department of Labor / ETA |
| **License** | CC BY 4.0 |

### Feature categories

Features are prefixed by category:

- **`Skill:`** — learned capabilities (e.g., `Skill:Programming`, `Skill:Critical Thinking`, `Skill:Writing`)
- **`Knowledge:`** — domain knowledge (e.g., `Knowledge:Mathematics`, `Knowledge:Computers and Electronics`)
- **`Ability:`** — enduring attributes (e.g., `Ability:Deductive Reasoning`, `Ability:Static Strength`, `Ability:Manual Dexterity`)

### How it was preprocessed

The raw [O\*NET 30.2 database](https://www.onetcenter.org/dl_files/database/db_30_2_text.zip) contains multiple scales per feature (importance, level, relevance). We extract:

1. **[Skills.txt](https://www.onetcenter.org/dictionary/30.2/text/Skills.html)**, **[Knowledge.txt](https://www.onetcenter.org/dictionary/30.2/text/Knowledge.html)**, **[Abilities.txt](https://www.onetcenter.org/dictionary/30.2/text/Abilities.html)** from the O\*NET 30.2 database
2. Filter to the **Importance** scale (`Scale ID = "IM"`), which rates each feature on a 1–5 scale
3. Drop rows marked as suppressed (`Recommend Suppress = "Y"`)
4. Pivot to an **occupation × feature** matrix (894 × 120)
5. Normalise each column to **[0, 1]** using min-max scaling
6. Replace O\*NET SOC codes with human-readable occupation titles

### Population statistics

| Statistic | Value |
|-----------|-------|
| Mean feature value | 0.385 |
| Feature std | 0.228 |
| Density (non-zero entries) | 92.2% |
| Ledoit-Wolf shrinkage | 0.0054 |
| Condition number | 1,884 |
| Effective dimensions (90% var) | ~15 |
| Mean \|correlation\| | 0.336 |
| Correlation sparsity (<0.1) | 14.8% |

The high mean correlation (0.336) and low sparsity (14.8%) mean most features are correlated — the population has rich transfer structure. Observing a few skills tells you a lot about the rest.

### Example

```python
pop = skillinfer.datasets.onet()

profile = pop.profile()
profile.observe("Skill:Programming", 0.92)
profile.observe("Skill:Critical Thinking", 0.85)
print(profile.predict())

# Use a specific occupation as prior
profile = pop.profile(prior_entity="Software Developers")
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `bool` | `False` | Re-normalise columns to [0, 1]. Data is already normalised, so this is rarely needed. |

---

## `esco()`

**ESCO v1.2.1** — the European Commission's taxonomy of Skills, Competences, Qualifications and Occupations.

ESCO is an independently curated taxonomy maintained by EU expert panels. Where O\*NET uses continuous importance ratings from U.S. surveys, ESCO uses binary skill assignments curated by European domain experts. This makes it a strong cross-validation target — if `skillinfer` works on both, the method generalises beyond any single taxonomy.

```python
pop = skillinfer.datasets.esco()
print(pop)
# Population(2999 agents x 134 skills, shrinkage=0.0211)
```

### What's in it

| | |
|---|---|
| **Entities** | 2,999 occupations (e.g., "technical director", "registered nurse", "software developer") |
| **Features** | 134 Level-2 skill groups |
| **Scale** | Binary (1 = occupation requires at least one essential skill in that group, 0 = does not) |
| **Source** | [ESCO v1.2.1](https://esco.ec.europa.eu/), European Commission |

### How it was preprocessed

The [ESCO classification](https://esco.ec.europa.eu/en/use-esco/download) assigns ~13,000 individual skills to occupations. We aggregate to a manageable matrix:

1. Load **[occupationSkillRelations_en.csv](https://esco.ec.europa.eu/en/use-esco/download)** — maps occupations to individual skills with relation types (essential/optional)
2. Filter to **essential** relations only
3. Walk each skill up the **[ESCO skill hierarchy](https://esco.ec.europa.eu/en/about-esco/what-does-esco-cover/skills-and-competences)** (via `broaderRelationsSkillPillar_en.csv` and `skillsHierarchy_en.csv`) to its **Level-2 skill group** ancestor
4. Build a binary **occupation × skill-group** matrix: 1 if the occupation has at least one essential skill in that group, 0 otherwise
5. Replace skill group URIs with human-readable **preferred labels** from the hierarchy
6. Drop occupations with fewer than 5 skill groups (too sparse to be informative)
7. Replace occupation URIs with human-readable titles from **occupations_en.csv**

### Population statistics

| Statistic | Value |
|-----------|-------|
| Density (fraction of 1s) | 10.1% |
| Ledoit-Wolf shrinkage | 0.0211 |
| Condition number | 468 |
| Effective dimensions (90% var) | ~15 |
| Mean \|correlation\| | 0.055 |
| Correlation sparsity (<0.1) | 84.6% |

ESCO is much sparser than O\*NET — only 10% of entries are 1, and 84.6% of feature correlations are near zero. The covariance structure is concentrated in a few meaningful clusters (e.g., healthcare skills co-occur, IT skills co-occur), with most skill groups being independent. Transfer is still valuable but more targeted.

### Key differences from O\*NET

| | O\*NET | ESCO |
|---|--------|------|
| **Source** | U.S. Department of Labor surveys | EU expert panel curation |
| **Scale** | Continuous [0, 1] | Binary {0, 1} |
| **Features** | 120 (skills + knowledge + abilities) | 134 (skill groups only) |
| **Entities** | 894 occupations | 2,999 occupations |
| **Correlation structure** | Dense (mean \|r\| = 0.34) | Sparse (mean \|r\| = 0.06) |
| **Best for** | Rich skill profiling, continuous predictions | Cross-validation, binary classification tasks |

### Example

```python
pop = skillinfer.datasets.esco()

profile = pop.profile()
profile.observe("education", 1.0)
profile.observe("teaching and training", 1.0)
print(profile.predict())
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | `bool` | `False` | Normalise columns to [0, 1]. Data is binary, so this is rarely needed. |
