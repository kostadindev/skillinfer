# Tutorial: European Skills (ESCO)

Cross-taxonomy validation with the European Skills taxonomy — a completely different data source that validates `skillinfer` generalizes beyond any single taxonomy's design.

!!! info "What you'll learn"
    - Working with binary (0/1) data instead of continuous ratings
    - Cross-taxonomy validation: does the method generalize?
    - Handling sparse, independently curated skill assignments

## About ESCO

[ESCO v1.2.1](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications and Occupations) is curated by the European Commission. It differs from O\*NET in three key ways:

| | O\*NET | ESCO |
|---|--------|------|
| **Source** | U.S. Department of Labor surveys | EU expert panel curation |
| **Feature type** | Continuous ratings (1–5) | Binary assignments (has/doesn't have) |
| **Scale** | 894 occupations x 120 features | ~3,000 occupations x 134 skill groups |

These differences make ESCO a strong cross-taxonomy validation: if `skillinfer` works on both, the method generalizes beyond any single taxonomy's design choices.

## Data preparation

ESCO assigns individual skills (~13,000) to occupations. We aggregate to Level-2 skill groups via the ESCO hierarchy, yielding a binary occupation x skill-group matrix.

!!! note "Data download"
    ESCO CSVs must be downloaded manually from [esco.ec.europa.eu/en/use-esco/download](https://esco.ec.europa.eu/en/use-esco/download). Select "ESCO dataset" → CSV → English.

```python
import skillinfer

# After parsing (see examples/esco.py for the full pipeline)
tax = skillinfer.Taxonomy.from_dataframe(R, normalize=False)
print(tax)
```

```text
Taxonomy(2847 entities x 134 features, shrinkage=0.0089)
```

The density of the binary matrix is typically ~25% — most occupations have about a third of the skill groups.

## Observe and predict

With binary data, observations are 0.0 or 1.0:

```python
state = tax.new_state(obs_noise=0.1)

# Observe 3 skill groups this occupation has
state.observe("using digital tools for collaboration", 1.0)
state.observe("developing objectives and strategies", 1.0)
state.observe("analysing data", 1.0)
```

The model predicts which other skill groups the occupation likely has:

```text
  Skill Group                                True  Pred  ± Std
  managing information                          1  0.721  0.089
  processing information                        1  0.694  0.092
  working with computers                        1  0.687  0.088
  communicating with others through media        1  0.623  0.095
  applying knowledge of human behaviour          0  0.412  0.104
  performing physical activities                 0  0.198  0.112
```

Even with binary data, the covariance structure captures meaningful skill relationships.

## Validation results

```python
results = skillinfer.validation.held_out_evaluation(
    tax, frac_observed=[0.1, 0.3, 0.5], n_splits=10, obs_noise=0.1
)
```

```text
frac_observed  method
0.1            diagonal    0.908
               kalman      0.934  ← transfer helps with binary data too
               prior       0.872
0.3            diagonal    0.956
               kalman      0.971
               prior       0.872
```

The Kalman filter outperforms the diagonal baseline on ESCO just as it does on O\*NET, despite the fundamentally different data characteristics (binary vs. continuous, EU curation vs. U.S. surveys).

## Full example

See [`examples/esco.py`](https://github.com/kostadindev/skillinfer/blob/main/examples/esco.py) for the complete script including ESCO data parsing and hierarchy traversal.

## Key takeaway

`skillinfer` works across different data types (binary and continuous), different curation methodologies (expert panels and surveys), and different scales (134 and 120 features). The covariance transfer mechanism is robust to these variations — **it's a property of the math, not the data format**.
