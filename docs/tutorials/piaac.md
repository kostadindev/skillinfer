# PIAAC — Profiling individual adults

Profile a single person from a few skill observations using the **within-individual** prior derived from the OECD's PIAAC adult-skills survey.

!!! info "What you'll learn"
    - The difference between a **cross-occupation** prior (`onet`, `esco`) and a **within-individual** prior (`piaac_prior`)
    - Working with a covariance-only Population (no entity rows)
    - Strong block structure in human assessment data: assessed cognitive scores vs. self-reported work-use scales
    - Why one observation is enough for the cognitive block, but the work-use block needs more

## About PIAAC

The [Programme for the International Assessment of Adult Competencies](https://www.oecd.org/skills/piaac/) is the OECD's measurement of skills among working-age adults. The Cycle 2 (2023) US public-use file contains roughly 3,800 adults assessed on:

- **Three IRT-scored proficiencies** — `literacy`, `numeracy`, `problem_solving`. These are direct task-performance measures: respondents read, calculate, and navigate adaptive computer-based items, and the score is the IRT-estimated mean of 10 plausible values.
- **Six self-reported work-use scales** — `readwork`, `writwork`, `numwork`, `ictwork`, `influence`, `taskdisc`. How often a respondent reads, writes, uses numbers, uses ICT, influences others, and exercises task discretion at work. Reported as weighted likelihood estimates.

Where O\*NET and ESCO describe **occupations**, PIAAC describes **individuals**. Two different priors with two different jobs.

## What's bundled

Unlike `onet()` and `esco()`, `piaac_prior()` ships **only summary statistics** — the population mean and Ledoit-Wolf shrunk covariance over 9 dimensions. No individual records leave the OECD distribution. The bundled file is 2 KB.

## Step 1: Load the prior

```python
import skillinfer

pop = skillinfer.datasets.piaac_prior()
print(pop)
```

```text
Population(1 entities x 9 skills)
```

The "1 entity" is a placeholder row holding the population mean — `piaac_prior` is built via [`Population.from_covariance`](../api/population.md#from_covariance). Everything that uses the prior structure (`profile`, `observe`, `predict`, `match_score`) works exactly as on `onet()` or `esco()`. Operations that need rows (`pop.entity(name)`, `pop.profile(prior_entity=...)`, held-out evaluation) do not apply.

## Step 2: Explore the covariance

```python
for _, row in pop.top_correlations(k=8).iterrows():
    print(f"  {row['feature_a']:<18} <-> {row['feature_b']:<18}  r = {row['correlation']:+.3f}")
```

```text
  literacy           <-> numeracy            r = +0.925
  literacy           <-> problem_solving     r = +0.914
  numeracy           <-> problem_solving     r = +0.905
  writwork           <-> ictwork             r = +0.548
  readwork           <-> ictwork             r = +0.527
  readwork           <-> writwork            r = +0.522
  readwork           <-> numwork             r = +0.431
  numwork            <-> ictwork             r = +0.420
```

Two clean blocks emerge:

- **Assessed cognitive block** (literacy, numeracy, problem_solving): pairwise correlations all above 0.90. Knowing one of these tells you almost everything about the other two.
- **Work-use block** (the six WLE scales): moderate positive correlations in the 0.4–0.55 range. People who read at work also tend to write and use ICT, but the signal is weaker than in the assessed block.
- **Cross-block correlations are weak** — assessed proficiency and self-reported usage are partially decoupled. Strong literacy doesn't necessarily mean strong reading-at-work, because work-use depends on the job too.

## Step 3: Few-shot recovery on the cognitive block

A single literacy observation already moves the rest of the cognitive block sharply:

```python
profile = pop.profile()
profile.observe("literacy", 0.85)

print(profile.predict().to_string(index=False))
```

```text
        feature   mean    std  ci_lower  ci_upper
       literacy  0.848  0.014     0.821     0.875
       numeracy  0.805  0.057     0.694     0.916
problem_solving  0.802  0.065     0.674     0.929
       readwork  0.655  0.324     0.020     1.000
       writwork  0.622  0.340     0.000     1.000
        numwork  0.663  0.341     0.000     1.000
        ictwork  0.667  0.382     0.000     1.000
      influence  0.660  0.341     0.000     1.000
       taskdisc  0.660  0.349     0.000     1.000
```

From one observation, `numeracy` and `problem_solving` jumped from the prior mean (0.61, 0.59) to 0.80 — almost matching the literacy reading. The CIs on the cognitive dims tightened sharply (std ≈ 0.06). The work-use dims barely moved and their CIs are still very wide (std ≈ 0.34) — the cross-block covariance just isn't strong enough to constrain them from a single cognitive observation.

## Step 4: Three observations close the cognitive block

```python
profile = pop.profile()
profile.observe("literacy", 0.85)
profile.observe("numeracy", 0.80)
profile.observe("problem_solving", 0.78)

print(profile.predict().to_string(index=False))
```

```text
        feature   mean    std  ci_lower  ci_upper
       literacy  0.848  0.014     0.821     0.875
       numeracy  0.800  0.014     0.773     0.827
problem_solving  0.781  0.014     0.753     0.808
       readwork  0.660  0.318     0.037     1.000
       writwork  0.630  0.337     0.000     1.000
        numwork  0.657  0.334     0.002     1.000
        ictwork  0.666  0.377     0.000     1.000
      influence  0.668  0.337     0.000     1.000
       taskdisc  0.653  0.346     0.000     1.000
```

The cognitive block is now fully constrained (std drops to 0.014, the observation noise floor). The work-use predictions only inched up — they need their own observations.

## Step 5: Which observations matter most next?

```python
profile.most_uncertain(k=4)
```

```text
   feature   mean    std
   ictwork  0.666  0.377
  taskdisc  0.653  0.346
 influence  0.668  0.337
  writwork  0.630  0.337
```

`most_uncertain` ranks the dimensions whose posterior is still the loosest. For this profile, `ictwork` is the highest-leverage next observation — it has the weakest cross-block transfer from the assessed scores and the strongest within-block transfer to the rest of the work-use scales (correlations 0.42–0.55).

## Step 6: When to use which prior

| | `piaac_prior()` | `onet()` | `esco()` |
|---|---|---|---|
| **Prior over** | Individuals | Occupations | Occupations |
| **Use when** | Profiling a specific person from partial assessments | Cold-starting an entity by stated occupation | Cold-starting in a binary skill space |
| **Dimensions** | 9 (3 assessed + 6 work-use) | 120 (skills + knowledge + abilities) | 134 (Level-2 skill groups) |
| **Score type** | Continuous, scaled to [0, 1] | Continuous, [0, 1] | Binary {0, 1} |
| **Data** | Summary statistics only (~2 KB) | Full occupation matrix (~260 KB) | Full occupation matrix (~85 KB) |
| **Bundled rows** | 0 | 894 | 2,999 |

A reasonable workflow combines both: start a new candidate with `onet()` using their stated occupation as `prior_entity`, then cross-walk PIAAC observations into O\*NET feature names (`literacy → Skill:Reading Comprehension`, `numeracy → Skill:Mathematics`, etc.) and call `observe()`. You get the wider coverage of O\*NET as the prior and the fidelity of PIAAC as the data.

## Closing

`piaac_prior()` exists to validate one specific claim: covariance transfer recovers individual human profiles, not just occupational averages. The block structure is sharper than O\*NET's because the dimensions are fewer and well-curated. The trade-off is coverage — 9 dimensions can only describe so much of a person. For routing or team-formation against rich task descriptions, PIAAC is a probe; O\*NET is the workspace.
