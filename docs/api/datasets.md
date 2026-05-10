# Datasets

Built-in datasets that return a `Population` ready for profiling. No downloads, no preprocessing — just import and go.

```python
import skillinfer

pop = skillinfer.datasets.onet()      # or skillinfer.datasets.esco()
profile = pop.profile()
profile.observe("Skill:Programming", 0.92)
print(profile.predict())
```

All datasets are shipped inside the package: O\*NET and ESCO as gzipped CSV (~510 KB combined), PIAAC as a small `.npz` of summary statistics (~2 KB).

---

## `onet()`

**O\*NET 30.2** — the U.S. Department of Labor's occupational information network.

O\*NET is the most comprehensive public database of occupational skill requirements. It describes what workers in each occupation need to know and be able to do, based on surveys of incumbent workers and occupational analysts.

```python
pop = skillinfer.datasets.onet()
print(pop)
# Population(894 entities x 120 skills, shrinkage=0.0054)
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
# Population(2999 entities x 134 skills, shrinkage=0.0211)
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

---

## `piaac_prior()`

**PIAAC Cycle 2 US** — within-individual prior derived from the OECD Programme for the International Assessment of Adult Competencies. Unlike `onet()` and `esco()`, this dataset ships **only the summary statistics** (population mean and covariance over 9 dimensions). No individual records are bundled.

```python
pop = skillinfer.datasets.piaac_prior()
print(pop)
# Population(1 entities x 9 skills, shrinkage=None)
```

The "1 entity" is a placeholder row holding the population mean — `piaac_prior` is built via [`Population.from_covariance`](population.md#from_covariance) and behaves correctly for everything that uses the prior structure (`profile`, `observe`, `predict`, `match_score`, `rank_agents`).

### What's in it

| | |
|---|---|
| **Entities** | None bundled — only population summary statistics |
| **Features** | 9 dimensions (3 IRT-assessed skills + 6 work-use scales) |
| **Scale** | Min-max scaled to [0, 1] within the PIAAC sample |
| **n (sample size)** | 2,548 adults with complete records |
| **Source** | [PIAAC Cycle 2 US PUF](https://nces.ed.gov/surveys/piaac/datafiles.asp), OECD / NCES |

### Dimensions

| Group | Names | Source columns |
|-------|-------|----------------|
| **IRT-assessed proficiency** | `literacy`, `numeracy`, `problem_solving` | Mean of 10 plausible values (`PVLIT*`, `PVNUM*`, `PVAPS*`) |
| **Skill use at work** | `readwork`, `writwork`, `numwork`, `ictwork`, `influence`, `taskdisc` | Weighted likelihood estimates (`*_WLE_CA*`) |

### When to use it (and when not to)

`piaac_prior()` ships a **within-individual** prior — it tells you how skills covary across people, which is what you want when profiling a specific human from partial observations. Use `onet()` or `esco()` instead when you want a **cross-occupation** prior — how skills covary across job types — for cold-starting an entity by their stated occupation.

### How it was preprocessed

1. Load `prgusap2.csv` (PIAAC Cycle 2 US PUF, semicolon-separated)
2. For each IRT-assessed dimension: take the mean of the 10 plausible values
3. For each work-use scale: take the published weighted likelihood estimate
4. Drop rows with any missing dimension (n=2,548 complete cases)
5. Min-max scale each dimension to [0, 1] (preserves correlations; keeps the package's clipping contract)
6. Compute population mean and Ledoit-Wolf shrunk covariance
7. Save as a 2 KB `.npz`

The full prep script is at [`scripts/prepare_piaac_prior.py`](https://github.com/kostadindev/skillinfer/blob/main/scripts/prepare_piaac_prior.py).

### Why the prior, not the rows?

The PIAAC public-use file is large (~35 MB) and shipping individual-level survey records inside a Python wheel is the wrong threshold. The covariance is the only thing the Kalman update needs from the population. By bundling only the derived prior, the dataset:

- Avoids redistributing OECD individual records
- Sidesteps the "10 plausible values per person" preprocessing question (we make the choice once, document it, and consume the result)
- Adds 2 KB to the wheel instead of tens of MB

### Population statistics

| Statistic | Value |
|-----------|-------|
| Sample size *n* | 2,548 |
| Dimensions *K* | 9 |
| Ledoit-Wolf shrinkage | 0.0030 |
| Top correlation | `literacy` ↔ `numeracy` (r = 0.92) |

### Example

```python
pop = skillinfer.datasets.piaac_prior()
profile = pop.profile()
profile.observe("literacy", 0.85)
profile.observe("numeracy", 0.80)
print(profile.predict())
```

Observing literacy and numeracy moves `problem_solving` from 0.59 (population mean) to ≈0.80 — the lift comes entirely through the off-diagonal covariance with the two assessed cognitive scores.
